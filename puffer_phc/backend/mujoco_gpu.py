"""MuJoCo GPU sim backend: Newton SolverMuJoCo on mujoco-warp, all envs as worlds of one batched model on CUDA.

Same model build, gym conventions and solver presets as mujoco_cpu (_newton_mujoco.py). Each env is a separate MuJoCo
world: no inter-env contact, independent constraint solve. Inspect the solver via `solver.mjw_model` /
`solver.mjw_data` (batched, on GPU); `solver.mj_model` is a host copy for introspection only.
"""

import atexit
import warnings

import mujoco_warp
import newton
import numpy as np
import torch
import warp as wp

from puffer_phc.backend import _newton_mujoco as nc

# per world. Rows: 4 per contact (pyramidal, condim 3) + active joint limits; peak 96 over all substeps (random actions).
# Grows with falls/lying poses, self-collision and model/geom changes; overflow drops rows (bodies sink), is flagged.
NCONMAX, NJMAX = 200, 160


class MujocoGPUBackend(nc.NewtonMujocoBackend):
    def __init__(self, device_type, *args, **kwargs):
        assert device_type == "cuda", "the mujoco_gpu backend needs device_type=cuda"
        super().__init__(device_type, *args, **kwargs)

    def create_envs(self, num_envs, env_spacing, kp_scale, kd_scale, has_self_collision, build_extras=None):
        n = self.num_envs = num_envs
        # One stream for torch and warp keeps their work ordered; CUDA graphs can't capture torch's legacy stream.
        torch.cuda.synchronize(self.device)
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(self.device)))
        self.model = nc.build_model(
            self._xml, self.body_names, self.preset, n, self.device, kp_scale, kd_scale, has_self_collision
        )
        self.body_masses = self.model.body_mass.numpy().reshape(n, -1).tolist()
        # ponytail: private mujoco-warp API (process-global const; BlockDim is experimental) — re-check after each
        # mujoco-warp bump. Values tuned on RTX 3090 @4096 envs for our nv=75 kinematic tree.
        mujoco_warp._src.types.M_BLOCK_DENSE_MAX = 80  # read in put_model(); default 64 < nv=75 trips sparse LDL
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast", cone="pyramidal", nconmax=NCONMAX, njmax=NJMAX
        )
        bd = self.solver.mjw_model.block_dim
        bd.cholesky_factorize = bd.cholesky_factorize_solve = 128  # default 32 is ~20% slower for this block size
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
        self._open_viewer(self.state)  # one N-world model/state covers all envs; the viewer lays worlds out itself

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

    def reset_envs(self, mask):
        """Push the root_states/dof_pos/dof_vel rows where mask is set to the sim; PD targets := dof_pos."""
        q, qd, target = nc.gym_to_newton(self._drive_frame, self.root_states, self.dof_pos, self.dof_vel)
        m = mask[:, None]  # torch.where over all envs: boolean indexing would sync
        self._q[:], self._qd[:] = torch.where(m, q, self._q), torch.where(m, qd, self._qd)
        self._target[:, 7:] = torch.where(m, target, self._target[:, 7:])
        self._warmstart.masked_fill_(m, 0)  # drop the stale warm start; don't use solver.reset()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, mask=wp.from_torch(mask))

    def overflow(self):
        """Solver overflow flags (sticky), all worlds OR-ed. Syncs."""
        return mujoco_warp.OverflowType(int(np.bitwise_or.reduce(self.solver.mjw_data.overflow.numpy())))

    def warn_overflow(self):
        if overflow := self.overflow():
            warnings.warn(f"MuJoCo Warp {overflow!r}: constraint rows or contacts dropped; raise NJMAX/NCONMAX")

    def solver_stats(self):
        steps = max(self._stats_steps, 1)
        (niter, nefc), (niter_max, nefc_max) = self._stats_sum.tolist(), self._stats_max.tolist()
        return (
            f"MuJoCo Warp solver over {self._stats_steps} env-steps (last substep of each simulate()): iterations "
            f"mean {niter / steps:.1f} max {int(niter_max)}, constraint rows mean {nefc / steps:.1f} max "
            f"{int(nefc_max)} (njmax {NJMAX}), overflow {self.overflow()!r}"
        )
