"""Shared by the Newton-built backends (mujoco_cpu.py, mujoco_gpu.py): model build from the gym MJCF, solver presets
and the gym <-> Newton state conversions.

Gym's 3-hinge joints are PhysX spherical joints, so the humanoid is imported with ball joints and converted at this
boundary: dof_pos = rotation vector per joint, dof_vel = child-frame relative angular velocity.
"""

import itertools
import os
import xml.etree.ElementTree as ET

import mujoco
import newton
import numpy as np
import torch

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
# settings for gym-trained checkpoints, not training presets. refsafe clamps contact timeconst >= 2*dt; a 1/120-scale
# config (tc >= 1/60) makes loaded feet slip 3-6x more than PhysX, hence the finer dt. See docs/physics_solver.md.
SOLVER_PRESETS = {
    # Default: half the physics cost of isaacgym_parity_480, refsafe off. 0.984 success / 39.5 mm MPJPE.
    "isaacgym_parity_240": dict(dt=1 / 240, substeps=4, solref=(0.005, 1.0), refsafe=False, gains=GYM_TGS_GAIN_SCALE),
    # Higher-fidelity reference, refsafe on. 0.977/0.961 success, 38.7/39.1 mm MPJPE (seeds 0/1).
    "isaacgym_parity_480": dict(dt=1 / 480, substeps=8, solref=(0.005, 1.0), refsafe=True, gains=GYM_TGS_GAIN_SCALE),
}


def load_asset(asset_file, device):
    """Returns the MJCF without its actuators, body names, gym dof_props and the PhysX drive frames [23, 3, 3]."""
    root = ET.parse(asset_file).getroot()
    for a in root.findall("actuator"):  # 69 motors that gym ignores; PD drives are set up in build_model
        root.remove(a)
    xml = ET.tostring(root, encoding="unicode")
    mjm = mujoco.MjModel.from_xml_string(xml)  # raw 3-hinge model: names and gym's per-axis limits
    dof_props = {
        "lower": mjm.jnt_range[1:, 0],
        "upper": mjm.jnt_range[1:, 1],
        "effort": np.full(mjm.nv - 6, np.finfo(np.float32).max),  # gym: unlimited
    }
    # PhysX applies the drive in the child's principal-inertia frame: equilibrium rotvec = Q_j @ target_j
    Q = np.loadtxt(os.path.join(os.path.dirname(asset_file), "smpl_humanoid_drive_frame.txt"))
    drive_frame = torch.tensor(Q, dtype=torch.float32, device=device).view(-1, 3, 3)
    return xml, [mjm.body(i).name for i in range(1, mjm.nbody)], dof_props, drive_frame


def build_model(xml, body_names, preset, n_worlds, device, kp_scale, kd_scale, has_self_collision):
    """Newton model: n_worlds humanoids (one world each, no inter-world contact) + a shared ground plane."""
    b = newton.ModelBuilder()
    b.add_mjcf(
        xml,
        ignore_names=["floor"],
        parse_sites=False,
        convert_3d_hinge_to_ball_joints=True,
        enable_self_collisions=has_self_collision,  # Newton filters parent-child pairs, like PhysX articulations
    )
    assert list(b.shape_body) == list(range(len(body_names))), "expects one shape per body"
    if has_self_collision:
        for i, j in itertools.combinations(range(len(body_names)), 2):
            if FILTER_INTS[i] & FILTER_INTS[j]:
                b.add_shape_collision_filter_pair(i, j)

    # The importer reads XML stiffness/damping as a MuJoCo passive spring/damper; gym uses them as PD gains.
    stiff = b.custom_attributes["mujoco:dof_passive_stiffness"]
    for d in range(6, b.joint_dof_count):
        kp, kd = kp_scale * float(stiff.values[d]), kd_scale * b.joint_damping[d]
        s = preset["gains"].get(body_names[1 + (d - 6) // 3], 1.0)
        b.joint_target_ke[d] = s * kp
        b.joint_target_kd[d] = s * (kd + kp * PHYSX_DT)  # PhysX's implicit kp = explicit kp + damping kp*dt
        b.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)
        b.joint_damping[d] = 0.0
        b.joint_armature[d] = 0.02
    stiff.values.clear()  # no passive spring

    # Contact params on the humanoid geoms; solmix 1e6 makes them win over the ground plane's defaults.
    for s in range(b.shape_count):
        b.custom_attributes["mujoco:condim"].values[s] = 3  # XML default condim=1 is frictionless
        b.custom_attributes["mujoco:solref"].values[s] = preset["solref"]
        b.custom_attributes["mujoco:geom_solimp"].values[s] = MJ_SOLIMP
        b.custom_attributes["mujoco:geom_solmix"].values[s] = 1e6
        b.shape_material_mu[s] = 1.0

    world = newton.ModelBuilder()
    world.replicate(b, n_worlds)
    world.add_ground_plane()
    return world.finalize(device=device)


def apply_refsafe(preset, opts):
    """Turn MuJoCo's refsafe off on each opt (MjOption or mujoco-warp Option) when the preset says so."""
    if not preset["refsafe"]:
        for opt in opts:
            opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)


def gym_tensors(n, num_bodies, num_dof, device):
    """The gym-convention tensors of base.SimBackend, zeroed."""
    return dict(
        root_states=torch.zeros(n, 13, device=device),
        dof_pos=torch.zeros(n, num_dof, device=device),
        dof_vel=torch.zeros(n, num_dof, device=device),
        rigid_body_pos=torch.zeros(n, num_bodies, 3, device=device),
        rigid_body_rot=torch.zeros(n, num_bodies, 4, device=device),
        rigid_body_vel=torch.zeros(n, num_bodies, 3, device=device),
        rigid_body_ang_vel=torch.zeros(n, num_bodies, 3, device=device),
        # Applied PD torque (MuJoCo child-frame ball DOFs). Gym's dof_force_tensor is not the drive torque
        # (4-8x smaller RMS, dynamics_report.md), so a power reward from this differs; eval does not use it.
        dof_force=torch.zeros(n, num_dof, device=device),
        contact_forces=torch.zeros(n, num_bodies, 3, device=device),  # unused
    )


def write_gym(t, q, qd, body_q, body_qd, qfrc):
    """Fill t's gym tensors in place from Newton joint_q [N, 76], joint_qd [N, 75], body_q [N, 24, 7],
    body_qd [N, 24, 6] (world COM vel, world ang vel) and MuJoCo qfrc_actuator [N, 75]."""
    n = len(q)
    joint_quat = q[:, 7:].view(n, -1, 4)
    t.root_states[:, :7] = q[:, :7]
    t.root_states[:, 7:] = qd[:, :6]
    t.dof_pos[:] = quat_to_exp_map(joint_quat).view(n, -1)
    t.dof_vel[:] = quat_rotate(quat_conjugate(joint_quat), qd[:, 6:].view(n, -1, 3)).flatten(1)
    t.rigid_body_pos[:] = body_q[..., :3]
    t.rigid_body_rot[:] = body_q[..., 3:]
    t.rigid_body_vel[:] = body_qd[..., :3]  # world COM velocity, as gym
    t.rigid_body_ang_vel[:] = body_qd[..., 3:]
    t.dof_force[:] = qfrc[:, 6:]


def pd_targets_to_newton(drive_frame, pd_targets):
    """Gym PD targets [N, 69] -> Newton ball joint targets [N, 92] (quats xyzw) in PhysX's drive frame."""
    rotvec = torch.einsum("jab,njb->nja", drive_frame, pd_targets.view(len(pd_targets), -1, 3))
    return exp_map_to_quat(rotvec).view(len(pd_targets), -1)


def gym_to_newton(drive_frame, root, dof_pos, dof_vel):
    """Newton joint_q, joint_qd and PD targets for gym root_states/dof_pos/dof_vel rows."""
    joint_quat = exp_map_to_quat(dof_pos.view(len(dof_pos), -1, 3))  # [n, 23, 4] xyzw
    q = torch.cat([root[:, :7], joint_quat.flatten(1)], -1)
    joint_vel = quat_rotate(joint_quat, dof_vel.view(joint_quat.shape[:2] + (3,)))  # Newton ball qd: parent frame
    qd = torch.cat([root[:, 7:], joint_vel.flatten(1)], -1)
    return q, qd, pd_targets_to_newton(drive_frame, dof_pos)
