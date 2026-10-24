import sys

from isaacgym import gymapi

try:
    import gymtorch
except ImportError:
    from isaacgym import gymtorch

import numpy as np
import torch

from puffer_phc.backend.base import FILTER_INTS, SimBackend
from puffer_phc.torch_utils import torch_rand_float


class IsaacGymBackend(SimBackend):
    def __init__(self, device_type, device_id, headless):
        assert device_type in ["cpu", "cuda"], "Device type must be cpu or cuda"

        if device_type == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"
        super().__init__(device_type, device_id)
        compute_device = -1 if "cuda" not in self.device else device_id
        graphics_device = -1 if headless else compute_device

        # Sim params: keep these hardcoded here for now
        sim_params = gymapi.SimParams()

        sim_params.dt = self.dt / self.control_freq_inv  # 1/60, with PhysX's default substeps=2
        self.dt = self.control_freq_inv * sim_params.dt  # keep the pre-refactor float32-rounded control dt

        sim_params.use_gpu_pipeline = "cuda" in self.device
        sim_params.num_client_threads = 0

        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 10.0

        sim_params.physx.use_gpu = "cuda" in self.device
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.num_subscenes = 0

        # Set gravity based on up axis and return axis index
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # Create sim and viewer
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        assert self.sim is not None, "Failed to create sim"
        self.sim_params = sim_params

        self.enable_viewer_sync = True
        self.viewer = None

        if not headless:  # Set up a minimal viewer
            # Subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # Set the camera position (Z axis up)
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_asset(self, asset_file):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self.humanoid_asset = self.gym.load_asset(self.sim, "/", asset_file, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(self.humanoid_asset)
        self.body_names = [self.gym.get_asset_rigid_body_name(self.humanoid_asset, i) for i in range(self.num_bodies)]
        self.dof_props = self.gym.get_asset_dof_properties(self.humanoid_asset)

    def create_envs(self, num_envs, env_spacing, kp_scale, kd_scale, has_self_collision, build_extras=None):
        """Create ground, envs with one PD-driven humanoid each, then prepare the sim and wrap its tensors.
        build_extras(env_id, env_ptr) lets the render env add marker actors to each env."""
        self.num_envs = num_envs

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        self.env_origins = []
        self.humanoid_handles = []
        self.body_masses = []
        max_agg_bodies, max_agg_shapes = 160, 160

        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        num_per_row = int(np.sqrt(num_envs))

        # Since the same humanoid is used for all the envs ...
        dof_prop = self.dof_props
        dof_prop["driveMode"] = gymapi.DOF_MODE_POS
        dof_prop["stiffness"] *= kp_scale
        dof_prop["damping"] *= kd_scale

        # NOTE: (from Joseph) You get a small perf boost (~4%) by putting all the actors in the same env
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._build_humanoid(i, env_ptr, dof_prop, has_self_collision)
            if build_extras is not None:
                build_extras(i, env_ptr)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

            # Save the env origins for the camera work (render_env)
            row = i // num_per_row
            col = i % num_per_row
            self.env_origins.append((col * 2 * env_spacing, row * 2 * env_spacing, 0.0))

        self.gym.prepare_sim(self.sim)
        self._setup_tensors()

    def _build_humanoid(self, env_id, env_ptr, dof_prop, has_self_collision):
        # Collision settings: probably affect speed a lot
        col_group = env_id  # no inter-environment collision
        col_filter = 0 if has_self_collision else 1

        assert self.sim_params.up_axis == gymapi.UP_AXIS_Z
        pos = torch.tensor((0, 0, 0.89)).to(self.device)  # NOTE: char_h (0.89) hard coded
        pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
            1
        )  # ZL: segfault if we do not randomize the position

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # NOTE: Domain randomization code was here. Search for self.cfg.domain_rand.has_domain_rand in the original repos.

        humanoid_handle = self.gym.create_actor(
            env_ptr, self.humanoid_asset, start_pose, f"humanoid_{env_id}", col_group, col_filter, 0
        )
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        self.body_masses.append(
            [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
        )

        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        if has_self_collision:
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
            assert len(FILTER_INTS) == len(props)

            for p_idx in range(len(props)):
                props[p_idx].filter = FILTER_INTS[p_idx]
            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

    def _setup_tensors(self):
        ### get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)  # Keep this as reference
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.refresh()

        # NOTE: all_root_states also holds the render env's marker actors
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.all_root_states.shape[0] // self.num_envs
        self.root_states = self.all_root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self.actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        self.dof_pos = self.dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 1]

        self.all_rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self.all_rigid_body_state.shape[0] // self.num_envs
        rigid_body_state = self.all_rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self.rigid_body_pos = rigid_body_state[..., : self.num_bodies, 0:3]
        self.rigid_body_rot = rigid_body_state[..., : self.num_bodies, 3:7]
        self.rigid_body_vel = rigid_body_state[..., : self.num_bodies, 7:10]
        self.rigid_body_ang_vel = rigid_body_state[..., : self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., : self.num_bodies, :]

    def refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def simulate(self):
        self.gym.simulate(self.sim)

    def step(self, pd_targets):
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pd_targets))
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.refresh()

    def reset_envs(self, mask):
        """Push the root_states/dof_pos/dof_vel rows where mask is set to the sim; PD targets := dof_pos."""
        env_ids_int32 = self.actor_ids[mask]
        if len(env_ids_int32) == 0:  # PhysX's indexed setters reject empty index tensors
            return

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.all_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_pos.contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def render(self):
        if not self.viewer:
            return

        # Check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # Check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # Step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
