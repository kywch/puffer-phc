from abc import ABC, abstractmethod


class SimBackend(ABC):
    """Simulator seam used by HumanoidPHC. Gym conventions throughout: N envs, 24 bodies, 69 DOFs (23 joints x 3).

    Call order: __init__ -> load_asset(asset_file) -> create_envs(...) -> step / reset_envs / refresh.
    load_asset sets num_bodies, num_dof, body_names (== SMPL_MUJOCO_NAMES) and dof_props ("lower", "upper",
    "effort" arrays of length 69). create_envs sets body_masses (N lists of 24) and these torch tensors on `device`:
      root_states [N,13]: pos, quat xyzw, world lin vel, world ang vel
      dof_pos, dof_vel [N,69]: per-joint rotation vectors (gym spherical joints) and their rates
      rigid_body_pos [N,24,3], rigid_body_rot [N,24,4] xyzw, rigid_body_vel (world COM), rigid_body_ang_vel (world)
      dof_force [N,69]: applied PD torque; contact_forces [N,24,3] (may stay zero)
    HumanoidPHC keeps references to these tensors: update them in place, never rebind them.
    """

    def __init__(self, device_type, device_id):
        self.device = f"cuda:{device_id}" if device_type == "cuda" else "cpu"
        self.control_freq_inv = 2
        self.dt = self.control_freq_inv / 60  # control step; physics runs at 1/120 s

    @abstractmethod
    def load_asset(self, asset_file): ...

    @abstractmethod
    def create_envs(self, num_envs, env_spacing, kp_scale, kd_scale, has_self_collision, build_extras=None):
        """build_extras(env_id, env_ptr) is a gym-only hook (render env markers); other backends ignore it."""

    @abstractmethod
    def step(self, pd_targets):
        """Set PD targets [N,69], run control_freq_inv simulate() calls, then refresh()."""

    @abstractmethod
    def simulate(self):
        """One sim step (1/60 s), no refresh."""

    @abstractmethod
    def reset_envs(self, env_ids):
        """Push rows env_ids of root_states/dof_pos/dof_vel (written by the caller) to the sim; PD targets := dof_pos.
        The caller then calls refresh(), which must leave consistent body state for those rows."""

    @abstractmethod
    def refresh(self): ...

    @abstractmethod
    def close(self): ...

    def render(self):
        pass
