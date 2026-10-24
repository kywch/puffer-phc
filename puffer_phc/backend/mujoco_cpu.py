"""MuJoCo C (CPU) sim backend for eval and solver inspection, built through Newton's SolverMuJoCo(use_mujoco_cpu=True)
so the model build is the one the mujoco_gpu backend uses (_newton_mujoco.py; mujoco-warp gives the same results).

MuJoCo C supports one world per model, so every env gets its own SolverMuJoCo on one shared single-world Newton model
(1 humanoid + ground). That is the per-world semantics of mujoco-warp: no inter-env contact and an independent
constraint solve per env. Inspect env i through `solvers[i].mj_model` / `solvers[i].mj_data`.

The Newton state lives on the CPU. The gym tensors are copies on `device`, refreshed after every step.
"""

import atexit

import mujoco
import newton
import numpy as np
import torch
import warp as wp

from puffer_phc.backend import _newton_mujoco as nc
from puffer_phc.backend.base import SimBackend


class MujocoCPUBackend(SimBackend):
    def __init__(self, device_type, device_id, headless, solver_preset="isaacgym_parity_240"):
        super().__init__(device_type, device_id)
        self.preset = nc.SOLVER_PRESETS[solver_preset]

    def load_asset(self, asset_file):
        self._xml, self.body_names, self.dof_props, self._drive_frame = nc.load_asset(asset_file, self.device)
        self.num_bodies, self.num_dof = len(self.body_names), len(self.dof_props["effort"])

    def create_envs(self, num_envs, env_spacing, kp_scale, kd_scale, has_self_collision, build_extras=None):
        self.num_envs = num_envs
        self.model = nc.build_model(
            self._xml, self.body_names, self.preset, 1, "cpu", kp_scale, kd_scale, has_self_collision
        )
        self.body_masses = [self.model.body_mass.numpy().tolist()] * num_envs
        with wp.ScopedDevice("cpu"):
            self.solvers = [
                newton.solvers.SolverMuJoCo(
                    self.model, use_mujoco_cpu=True, solver="newton", integrator="implicitfast", cone="pyramidal"
                )
                for _ in range(num_envs)
            ]
        nc.apply_refsafe(self.preset, [s.mj_model.opt for s in self.solvers])
        self.states = [self.model.state() for _ in range(num_envs)]
        self.controls = [self.model.control() for _ in range(num_envs)]
        self._q = [wp.to_torch(s.joint_q) for s in self.states]  # [7 + 23*4]: root pos, quat xyzw, joint quats
        self._qd = [wp.to_torch(s.joint_qd) for s in self.states]  # [6 + 69]: world root vels, parent-frame ang vels
        self._body_q = [wp.to_torch(s.body_q) for s in self.states]
        self._body_qd = [wp.to_torch(s.body_qd) for s in self.states]
        self._target = [wp.to_torch(c.joint_target_q) for c in self.controls]

        vars(self).update(nc.gym_tensors(num_envs, self.num_bodies, self.num_dof, self.device))
        self._stats_sum, self._stats_max = np.zeros(3), np.zeros(2)  # (steps, niter, ncon), (niter, ncon)
        atexit.register(lambda: print(self.solver_stats()))
        self.refresh()

    def refresh(self):
        views = (self._q, self._qd, self._body_q, self._body_qd)
        qfrc = torch.tensor(np.stack([s.mj_data.qfrc_actuator for s in self.solvers]), device=self.device)
        nc.write_gym(self, *(torch.stack(v).to(self.device) for v in views), qfrc)

    def simulate(self):
        dt, substeps = self.preset["dt"], self.preset["substeps"]
        for solver, state, control in zip(self.solvers, self.states, self.controls):
            for _ in range(substeps):
                solver.step(state, state, control, None, dt)
                niter, ncon = solver.mj_data.solver_niter[0], solver.mj_data.ncon
                self._stats_sum += (1, niter, ncon)
                self._stats_max = np.maximum(self._stats_max, (niter, ncon))

    def step(self, pd_targets):
        for target, t in zip(self._target, nc.pd_targets_to_newton(self._drive_frame, pd_targets).cpu()):
            target[7:] = t
        for _ in range(self.control_freq_inv):
            self.simulate()
        self.refresh()

    def reset_envs(self, env_ids):
        """Push the root_states/dof_pos/dof_vel rows of env_ids to the sim; PD targets := dof_pos."""
        rows = (self.root_states[env_ids], self.dof_pos[env_ids], self.dof_vel[env_ids])
        q, qd, targets = (x.cpu() for x in nc.gym_to_newton(self._drive_frame, *rows))
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
