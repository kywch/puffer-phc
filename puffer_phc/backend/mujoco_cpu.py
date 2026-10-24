"""MuJoCo C (CPU) sim backend for eval and solver inspection, built through Newton's SolverMuJoCo(use_mujoco_cpu=True)
so the model build is the one a GPU Newton backend reuses (mujoco-warp gives the same results, tmp/newton/FINDINGS.md).

MuJoCo C supports one world per model, so every env gets its own SolverMuJoCo on one shared single-world Newton model
(1 humanoid + ground). That is the per-world semantics of mujoco-warp: no inter-env contact and an independent
constraint solve per env. Inspect env i through `solvers[i].mj_model` / `solvers[i].mj_data`.

The Newton state lives on the CPU. The tensors below are copies on `device` in gym layout and conventions, refreshed
after every step. Gym's 3-hinge joints are PhysX spherical joints, so the humanoid is imported with ball joints and
converted at this boundary: dof_pos = rotation vector per joint, dof_vel = child-frame relative angular velocity.
"""

import atexit
import itertools
import os
import xml.etree.ElementTree as ET

import mujoco
import newton
import numpy as np
import torch
import warp as wp

from puffer_phc.backend.base import SimBackend
from puffer_phc.torch_utils import exp_map_to_quat, quat_conjugate, quat_rotate, quat_to_exp_map

# Gym's per-shape self-collision filter bits (humanoid_phc / isaacgym backend): shapes collide iff (f_i & f_j) == 0
FILTER_INTS = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # fmt: skip
PHYSX_DT = 1 / 120  # gym's physics step: PhysX's implicit drive adds kp*PHYSX_DT damping
MJ_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)  # MuJoCo default
# gym TGS (4 position iterations) is unconverged: loaded joints act softer. kp and kd scale per body
GYM_TGS_GAIN_SCALE = {
    "L_Ankle": 0.2, "R_Ankle": 0.2, "L_Toe": 0.2, "R_Toe": 0.2,
    "L_Hip": 0.8, "R_Hip": 0.8, "L_Knee": 0.8, "R_Knee": 0.8,
    "Torso": 0.5, "Spine": 0.5, "Chest": 0.5,
}  # fmt: skip
# name: physics dt, steps per simulate(), contact solref, refsafe on/off, gym-TGS gain scales. Both are eval-parity
# settings for gym-trained checkpoints, not training presets; see docs/physics_solver.md.
SOLVER_PRESETS = {
    # Default: half the physics cost of isaacgym_parity_480, refsafe off. 0.984 success / 39.5 mm MPJPE.
    "isaacgym_parity_240": dict(dt=1 / 240, substeps=4, solref=(0.005, 1.0), refsafe=False, gains=GYM_TGS_GAIN_SCALE),
    # Higher-fidelity reference, refsafe on. 0.977/0.961 success, 38.7/39.1 mm MPJPE (seeds 0/1).
    "isaacgym_parity_480": dict(dt=1 / 480, substeps=8, solref=(0.005, 1.0), refsafe=True, gains=GYM_TGS_GAIN_SCALE),
}


class MujocoCPUBackend(SimBackend):
    def __init__(self, device_type, device_id, headless, solver_preset="isaacgym_parity_240"):
        super().__init__(device_type, device_id)
        self.preset = SOLVER_PRESETS[solver_preset]

    def load_asset(self, asset_file):
        root = ET.parse(asset_file).getroot()
        for a in root.findall("actuator"):  # 69 motors that gym ignores; PD drives are set up in create_envs
            root.remove(a)
        self._xml = ET.tostring(root, encoding="unicode")
        mjm = mujoco.MjModel.from_xml_string(self._xml)  # raw 3-hinge model: names and gym's per-axis limits
        self.body_names = [mjm.body(i).name for i in range(1, mjm.nbody)]
        self.num_bodies, self.num_dof = len(self.body_names), mjm.nv - 6
        self.dof_props = {
            "lower": mjm.jnt_range[1:, 0],
            "upper": mjm.jnt_range[1:, 1],
            "effort": np.full(self.num_dof, np.finfo(np.float32).max),  # gym: unlimited
        }
        # PhysX applies the drive in the child's principal-inertia frame: equilibrium rotvec = Q_j @ target_j
        Q = np.loadtxt(os.path.join(os.path.dirname(asset_file), "smpl_humanoid_drive_frame.txt"))
        self._drive_frame = torch.tensor(Q, dtype=torch.float32, device=self.device).view(-1, 3, 3)

    def create_envs(self, num_envs, env_spacing, kp_scale, kd_scale, has_self_collision, build_extras=None):
        self.num_envs = num_envs
        b = newton.ModelBuilder()
        b.add_mjcf(
            self._xml,
            ignore_names=["floor"],
            parse_sites=False,
            convert_3d_hinge_to_ball_joints=True,
            enable_self_collisions=has_self_collision,  # Newton filters parent-child pairs, like PhysX articulations
        )
        assert list(b.shape_body) == list(range(self.num_bodies)), "expects one shape per body"
        if has_self_collision:
            for i, j in itertools.combinations(range(self.num_bodies), 2):
                if FILTER_INTS[i] & FILTER_INTS[j]:
                    b.add_shape_collision_filter_pair(i, j)

        # The importer reads XML stiffness/damping as a MuJoCo passive spring/damper; gym uses them as PD gains.
        stiff = b.custom_attributes["mujoco:dof_passive_stiffness"]
        gains = self.preset["gains"]
        for d in range(6, b.joint_dof_count):
            kp, kd = kp_scale * float(stiff.values[d]), kd_scale * b.joint_damping[d]
            s = gains.get(self.body_names[1 + (d - 6) // 3], 1.0)
            b.joint_target_ke[d] = s * kp
            b.joint_target_kd[d] = s * (kd + kp * PHYSX_DT)  # PhysX's implicit kp = explicit kp + damping kp*dt
            b.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)
            b.joint_damping[d] = 0.0
            b.joint_armature[d] = 0.02
        stiff.values.clear()  # no passive spring

        # Contact params on the humanoid geoms; solmix 1e6 makes them win over the ground plane's defaults.
        for s in range(b.shape_count):
            b.custom_attributes["mujoco:condim"].values[s] = 3  # XML default condim=1 is frictionless
            b.custom_attributes["mujoco:solref"].values[s] = self.preset["solref"]
            b.custom_attributes["mujoco:geom_solimp"].values[s] = MJ_SOLIMP
            b.custom_attributes["mujoco:geom_solmix"].values[s] = 1e6
            b.shape_material_mu[s] = 1.0

        world = newton.ModelBuilder()
        world.add_world(b)
        world.add_ground_plane()
        self.model = world.finalize(device="cpu")
        self.body_masses = [self.model.body_mass.numpy().tolist()] * num_envs

        # solver_preset picks physics dt/substeps, contact solref, refsafe on/off and PD gain scales
        # (SOLVER_PRESETS above): eval-parity settings for gym-trained checkpoints, not training presets.
        # refsafe clamps contact timeconst >= 2*dt; a 1/120-scale config (tc >= 1/60) makes loaded feet slip
        # 3-6x more than PhysX, hence the finer dt. See docs/physics_solver.md.
        with wp.ScopedDevice("cpu"):
            self.solvers = [
                newton.solvers.SolverMuJoCo(
                    self.model, use_mujoco_cpu=True, solver="newton", integrator="implicitfast", cone="pyramidal"
                )
                for _ in range(num_envs)
            ]
        if not self.preset["refsafe"]:
            for solver in self.solvers:
                solver.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_REFSAFE
        self.states = [self.model.state() for _ in range(num_envs)]
        self.controls = [self.model.control() for _ in range(num_envs)]
        self._q = [wp.to_torch(s.joint_q) for s in self.states]  # [7 + 23*4]: root pos, quat xyzw, joint quats
        self._qd = [wp.to_torch(s.joint_qd) for s in self.states]  # [6 + 69]: world root vels, parent-frame ang vels
        self._body_q = [wp.to_torch(s.body_q) for s in self.states]
        self._body_qd = [wp.to_torch(s.body_qd) for s in self.states]
        self._target = [wp.to_torch(c.joint_target_q) for c in self.controls]

        n, dev = num_envs, self.device
        self.root_states = torch.zeros(n, 13, device=dev)
        self.dof_pos = torch.zeros(n, self.num_dof, device=dev)
        self.dof_vel = torch.zeros(n, self.num_dof, device=dev)
        self.rigid_body_pos = torch.zeros(n, self.num_bodies, 3, device=dev)
        self.rigid_body_rot = torch.zeros(n, self.num_bodies, 4, device=dev)
        self.rigid_body_vel = torch.zeros(n, self.num_bodies, 3, device=dev)
        self.rigid_body_ang_vel = torch.zeros(n, self.num_bodies, 3, device=dev)
        # Applied PD torque (MuJoCo child-frame ball DOFs). Gym's dof_force_tensor is not the drive torque
        # (4-8x smaller RMS, dynamics_report.md), so a power reward from this differs; eval does not use it.
        self.dof_force = torch.zeros(n, self.num_dof, device=dev)
        self.contact_forces = torch.zeros(n, self.num_bodies, 3, device=dev)  # unused
        self._stats_sum, self._stats_max = np.zeros(3), np.zeros(2)  # (steps, niter, ncon), (niter, ncon)
        atexit.register(lambda: print(self.solver_stats()))
        self.refresh()

    def _targets(self, pd_targets):  # gym PD targets -> Newton ball joint targets in PhysX's drive frame
        rotvec = torch.einsum("jab,njb->nja", self._drive_frame, pd_targets.view(-1, self.num_dof // 3, 3))
        return exp_map_to_quat(rotvec).view(len(pd_targets), -1).cpu()

    def refresh(self):
        q = torch.stack(self._q).to(self.device)
        qd = torch.stack(self._qd).to(self.device)
        body_q = torch.stack(self._body_q).to(self.device)
        body_qd = torch.stack(self._body_qd).to(self.device)
        joint_quat = q[:, 7:].view(self.num_envs, -1, 4)
        self.root_states[:, :7] = q[:, :7]
        self.root_states[:, 7:] = qd[:, :6]
        self.dof_pos[:] = quat_to_exp_map(joint_quat).view(self.num_envs, -1)
        self.dof_vel[:] = quat_rotate(quat_conjugate(joint_quat), qd[:, 6:].view(self.num_envs, -1, 3)).flatten(1)
        self.rigid_body_pos[:] = body_q[..., :3]
        self.rigid_body_rot[:] = body_q[..., 3:]
        self.rigid_body_vel[:] = body_qd[..., :3]  # world COM velocity, as gym
        self.rigid_body_ang_vel[:] = body_qd[..., 3:]
        self.dof_force[:] = torch.tensor(np.stack([s.mj_data.qfrc_actuator[6:] for s in self.solvers]))

    def simulate(self):
        dt, substeps = self.preset["dt"], self.preset["substeps"]
        for solver, state, control in zip(self.solvers, self.states, self.controls):
            for _ in range(substeps):
                solver.step(state, state, control, None, dt)
                niter, ncon = solver.mj_data.solver_niter[0], solver.mj_data.ncon
                self._stats_sum += (1, niter, ncon)
                self._stats_max = np.maximum(self._stats_max, (niter, ncon))

    def step(self, pd_targets):
        for target, t in zip(self._target, self._targets(pd_targets)):
            target[7:] = t
        for _ in range(self.control_freq_inv):
            self.simulate()
        self.refresh()

    def reset_envs(self, env_ids):
        """Push the root_states/dof_pos/dof_vel rows of env_ids to the sim; PD targets := dof_pos."""
        root, dof_pos, dof_vel = self.root_states[env_ids], self.dof_pos[env_ids], self.dof_vel[env_ids]
        joint_quat = exp_map_to_quat(dof_pos.view(len(dof_pos), -1, 3))  # [n, 23, 4] xyzw
        q = torch.cat([root[:, :7], joint_quat.flatten(1)], -1).cpu()
        joint_vel = quat_rotate(joint_quat, dof_vel.view(joint_quat.shape[:2] + (3,)))  # Newton ball qd: parent frame
        qd = torch.cat([root[:, 7:], joint_vel.flatten(1)], -1).cpu()
        targets = self._targets(dof_pos)
        for k, i in enumerate(env_ids.tolist()):
            self._q[i][:] = q[k]
            self._qd[i][:] = qd[k]
            self._target[i][7:] = targets[k]
            self.solvers[i].mj_data.qacc_warmstart[:] = 0  # drop the stale warm start; don't use solver.reset()
            newton.eval_fk(self.model, self.states[i].joint_q, self.states[i].joint_qd, self.states[i])

    def solver_stats(self):
        (steps, niter, ncon), (niter_max, ncon_max) = self._stats_sum, self._stats_max
        warnings = {name: sum(s.mj_data.warning[int(w)].number for s in self.solvers)
                    for name, w in mujoco.mjtWarning.__members__.items() if name != "mjNWARNING"}  # fmt: skip
        warnings = {k: v for k, v in warnings.items() if v}
        return (
            f"MuJoCo CPU solver over {int(steps)} env-steps: iterations mean {niter / max(steps, 1):.1f} max "
            f"{int(niter_max)}, contacts mean {ncon / max(steps, 1):.1f} max {int(ncon_max)}, warnings {warnings}"
        )

    def close(self):
        pass
