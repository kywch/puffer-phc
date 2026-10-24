"""MuJoCo GPU sim backend: Newton SolverMuJoCo on mujoco-warp, all envs as worlds of one batched model on CUDA.

Same model build, gym conventions and solver presets as mujoco_cpu (_newton_mujoco.py). Each env is a separate MuJoCo
world: no inter-env contact, independent constraint solve. Inspect the solver via `solver.mjw_model` /
`solver.mjw_data` (batched, on GPU); `solver.mj_model` is a host copy for introspection only.
"""

import atexit

import mujoco_warp
import newton
import numpy as np
import torch
import warp as wp

from puffer_phc.backend import _newton_mujoco as nc
from puffer_phc.backend.base import SimBackend

NCONMAX, NJMAX = 200, 800  # per world; a humanoid lying flat has ~60-100 contacts, 4 rows each (pyramidal condim 3)


class MujocoGPUBackend(SimBackend):
    def __init__(self, device_type, device_id, headless, solver_preset="isaacgym_parity_240"):
        super().__init__(device_type, device_id)
        assert device_type == "cuda", "the mujoco_gpu backend needs device_type=cuda"
        self.preset = nc.SOLVER_PRESETS[solver_preset]

    def load_asset(self, asset_file):
        self._xml, self.body_names, self.dof_props, self._drive_frame = nc.load_asset(asset_file, self.device)
        self.num_bodies, self.num_dof = len(self.body_names), len(self.dof_props["effort"])

    def create_envs(self, num_envs, env_spacing, kp_scale, kd_scale, has_self_collision, build_extras=None):
        n = self.num_envs = num_envs
        # One stream for torch and warp keeps their work ordered; CUDA graphs can't capture torch's legacy stream.
        torch.cuda.synchronize(self.device)
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(self.device)))
        self.model = nc.build_model(
            self._xml, self.body_names, self.preset, n, self.device, kp_scale, kd_scale, has_self_collision
        )
        self.body_masses = self.model.body_mass.numpy().reshape(n, -1).tolist()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast", cone="pyramidal", nconmax=NCONMAX, njmax=NJMAX
        )
        nc.apply_refsafe(self.preset, [self.solver.mj_model.opt, self.solver.mjw_model.opt])  # before graph capture
        self.state, self.control = s, c = self.model.state(), self.model.control()
        views = [wp.to_torch(a) for a in (s.joint_q, s.joint_qd, s.body_q, s.body_qd, c.joint_target_q)]
        self._q, self._qd, self._body_q, self._body_qd, self._target = (v.view(n, -1, *v.shape[1:]) for v in views)
        d = self.solver.mjw_data
        self._qfrc, self._warmstart, self._niter, self._nefc = map(
            wp.to_torch, (d.qfrc_actuator, d.qacc_warmstart, d.solver_niter, d.nefc)
        )
        self._graphs = {}  # (dt, substeps) -> CUDA graph of one simulate()

        vars(self).update(nc.gym_tensors(n, self.num_bodies, self.num_dof, self.device))
        self._stats_sum, self._stats_max = torch.zeros(2, 2, device=self.device)  # (niter, nefc)
        self._stats_steps = 0
        atexit.register(lambda: print(self.solver_stats()))
        self.refresh()

    def refresh(self):
        nc.write_gym(self, self._q, self._qd, self._body_q, self._body_qd, self._qfrc)

    def _substeps(self, dt, substeps):
        for _ in range(substeps):
            self.solver.step(self.state, self.state, self.control, None, dt)

    def simulate(self):
        key = (self.preset["dt"], self.preset["substeps"])
        if key in self._graphs:
            wp.capture_launch(self._graphs[key])
        else:
            self._substeps(*key)  # eager first run compiles the kernels
            with wp.ScopedCapture(device=self.device) as cap:
                self._substeps(*key)
            self._graphs[key] = cap.graph
        # ponytail: stats sample only the last substep of each simulate(); a warp kernel in the graph would see all
        sample = torch.stack([self._niter, self._nefc]).float()
        self._stats_sum += sample.sum(1)
        torch.maximum(self._stats_max, sample.amax(1), out=self._stats_max)
        self._stats_steps += self.num_envs

    def step(self, pd_targets):
        self._target[:, 7:] = nc.pd_targets_to_newton(self._drive_frame, pd_targets)
        for _ in range(self.control_freq_inv):
            self.simulate()
        self.refresh()

    def reset_envs(self, env_ids):
        """Push the root_states/dof_pos/dof_vel rows of env_ids to the sim; PD targets := dof_pos."""
        rows = (self.root_states[env_ids], self.dof_pos[env_ids], self.dof_vel[env_ids])
        self._q[env_ids], self._qd[env_ids], self._target[env_ids, 7:] = nc.gym_to_newton(self._drive_frame, *rows)
        self._warmstart[env_ids] = 0  # drop the stale warm start; don't use solver.reset()
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, mask=wp.from_torch(mask))

    def solver_stats(self):
        steps = max(self._stats_steps, 1)
        (niter, nefc), (niter_max, nefc_max) = self._stats_sum.tolist(), self._stats_max.tolist()
        overflow = mujoco_warp.OverflowType(int(np.bitwise_or.reduce(self.solver.mjw_data.overflow.numpy())))
        return (
            f"MuJoCo Warp solver over {self._stats_steps} env-steps (last substep of each simulate()): iterations "
            f"mean {niter / steps:.1f} max {int(niter_max)}, constraint rows mean {nefc / steps:.1f} max "
            f"{int(nefc_max)} (njmax {NJMAX}), overflow {overflow!r}"
        )

    def close(self):
        pass
