import os
import copy
from enum import Enum
from types import SimpleNamespace

from gym import spaces
import torch
import numpy as np

from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES

from puffer_phc import ASSET_DIR
from puffer_phc.backend import make_backend
from puffer_phc.poselib_skeleton import SkeletonTree
from puffer_phc.motion_lib import MotionLibSMPL, FixHeightMode
from puffer_phc.sampling import MotionSampler, StartSampler
from puffer_phc.torch_utils import (
    to_torch,
    exp_map_to_quat,
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_conjugate,
    quat_to_tan_norm,
    quat_to_angle_axis,
)


class StateInit(Enum):
    Start = 1
    Random = 2


class HumanoidPHC:
    def __init__(
        self,
        cfg,
        device_type="cuda",
        device_id=0,  # Allow multi-gpu setting
        headless=True,
    ):
        backend_name = cfg["env"]["backend"]
        backend_kwargs = dict(device_type=device_type, device_id=device_id, headless=headless)
        if backend_name in ("mujoco_cpu", "mujoco_gpu"):
            backend_kwargs["solver_preset"] = cfg["env"]["solver_preset"]
        self.backend = make_backend(backend_name, **backend_kwargs)
        self.device = self.backend.device
        self.dt = self.backend.dt

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        ##########################
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.all_env_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_file = cfg["env"]["motion_file"]  # Must be provided

        ### Robot
        self._config_robot()  #  All robot configs should be here

        ### Env
        self._config_env()  # All env configs should be here
        self._create_envs()

        self._define_gym_spaces()
        self._setup_gym_tensors()
        self._setup_env_buffers()

        self.flag_test = False  # eval mode: see toggle_eval_mode()
        self._load_motion(self.motion_file)

    def reset(self):
        everyone = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_masked(everyone)

        # ZL: This way it will simulate one step, then get reset again, squashing any remaining wiredness. Temporary fix
        self.backend.simulate()
        self._reset_masked(everyone)
        torch.cuda.empty_cache()

        self._reset_ref_motion_times = self._motion_start_times.clone()
        self._compute_observations()
        if self.use_amp_obs:
            self._init_amp_obs(self.all_env_ids)
        return self.obs_buf

    def step(self, actions):
        self.backend.step(self._pd_action_scale * actions)

        ### Compute observations, rewards, resets, which was self.post_physics_step()
        # This is after stepping, so progress buffer got + 1. Compute reset/reward do not need to forward 1 timestep since they are for "this" frame now.
        self.progress_buf += 1

        # reward, reset and eval all compare against the reference at this frame
        motion_times = self.progress_buf * self.dt + self._motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.all_env_ids, motion_times)
        self._compute_reward(motion_res)
        self._compute_reset(motion_res)

        self.extras["terminate"] = self._terminate_buf.clone()
        self.extras["reward_raw"] = self.reward_raw.detach()

        if self.use_amp_obs:
            self._update_hist_amp_obs()  # One step for the amp obs
            self._compute_amp_observations()
            self.extras["amp_obs"] = self.amp_obs  ## ZL: hooks for adding amp_obs for training

        if self.flag_test:
            self.extras["body_pos"] = self._rigid_body_pos.clone()  # _reset_masked below overwrites it in place
            self.extras["body_pos_gt"] = motion_res["rg_pos"]

        done = self.reset_buf.clone()
        if not self.flag_test:
            self.start_sampler.record(self._motion_lib._curr_motion_ids, self.progress_buf, done, self._terminate_buf)
        assert not self.use_amp_obs, "the masked reset has no AMP init"
        self._reset_masked(done)
        self._compute_observations()  # observation for the next step, after the reset

        self.render()  # no-op for headless backends; HumanoidRenderEnv overrides render(), not step()

        # obs, reward, done, info
        return self.obs_buf, self.rew_buf, done, self.extras

    def render(self):
        self.backend.render()

    def close(self):
        self.backend.close()

    #####################################################################
    ### __init__()
    #####################################################################

    def _config_robot(self):
        # Currently only supporting SMPL neutral humanoids
        # The PHC code can load/create differently-shaped SMPL humanoids, or unitree ones.
        self.humanoid_type = "smpl"

        # For SMPL PHC, the below are different from the default
        self._has_self_collision = self.cfg["robot"]["has_self_collision"]  # is True
        self._has_upright_start = True
        self._has_dof_subset = True

        # NOTE: These are used in the AMP obs computation.
        self._has_shape_obs_disc = False
        self._has_limb_weight_obs_disc = False

        # NOTE: To customize SMPL, see below links
        # https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid.py#L270
        # https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid.py#L782

        ### Define body, joints, dof
        self._body_names = SMPL_MUJOCO_NAMES

        # Following UHC as hand and toes does not have realiable data.
        self.remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]

        self.joint_groups = [
            ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
            ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
            ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
            ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
            ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
        ]
        self.limb_weight_group = [
            [self._body_names.index(joint_name) for joint_name in joint_group] for joint_group in self.joint_groups
        ]

        disc_idxes = []
        self._dof_names = self._body_names[1:]
        for idx, name in enumerate(self._dof_names):
            if name not in self.remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = (
            torch.from_numpy(np.concatenate(disc_idxes)) if len(disc_idxes) > 0 else torch.tensor([]).long()
        )

        ### Load the Neutral SMPL humanoid asset only
        self.gender_beta = np.zeros(17)  # NOTE: gender (1) + betas (16)

        # And we use the same humanoid shapes for all the agents.
        self.humanoid_shapes = torch.tensor(np.array([self.gender_beta] * self.num_envs)).float().to(self.device)

        # NOTE: The below SMPL assets must be present.
        asset_file_real = str(ASSET_DIR / "smpl_humanoid.xml")
        assert os.path.exists(asset_file_real)

        sk_tree = SkeletonTree.from_mjcf(asset_file_real)
        self.skeleton_trees = [sk_tree] * self.num_envs

        self.backend.load_asset(asset_file_real)
        self.num_bodies = self.backend.num_bodies
        self.num_dof = self.backend.num_dof

        assert self.num_bodies == len(
            self._body_names
        ), "Number of bodies in asset file does not match number of SMPL bodies"
        assert self.num_dof == len(self._dof_names) * 3, "Number of DOF in asset file does not match number of SMPL DOF"

        # Check if the body ids are consistent between the sim asset and self._body_names (SMPL_MUJOCO_NAMES)
        assert self.backend.body_names == self._body_names, f"Body order mismatch: {self.backend.body_names}"

    def _build_body_ids_tensor(self, body_names):
        body_ids = [self._body_names.index(name) for name in body_names]
        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _config_env(self):
        env_config = self.cfg["env"]

        ### Overall env-related
        self.max_episode_length = self._step_cap = 300  # training episode cap
        self._termination_distances = to_torch(
            np.array([env_config["termination_distance"]] * self.num_bodies), device=self.device
        )
        # NOTE: _termination_distances is changed between train/eval, so keep a backup
        self._termination_distances_backup = self._termination_distances.clone()

        self.env_spacing = 5

        ### Obs/action-related
        self._local_root_obs = True

        self.key_bodies = ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]
        self._key_body_ids = self._build_body_ids_tensor(self.key_bodies)

        # All bodies are tracked and reset on; eval drops the hands and toes (following UHC)
        self._reset_bodies_id = self._build_body_ids_tensor(self._body_names)
        # NOTE: reset_bodies_id is changed between train/eval, so keep a backup
        self._reset_bodies_id_backup = self._reset_bodies_id
        eval_bodies = [name for name in self._body_names if name not in self.remove_names]
        self._eval_track_bodies_id = self._build_body_ids_tensor(eval_bodies)

        ### Control-related
        self._kp_scale = env_config["kp_scale"]
        self._kd_scale = env_config["kd_scale"]

        ### Motion/AMP-related
        self._min_motion_len = 5

        self._state_init = StateInit[env_config.get("state_init", "Random")]

        self.use_amp_obs = env_config["use_amp_obs"]
        self._num_amp_obs_steps = 10
        self._amp_root_height_obs = True

        ### Reward-related
        self.power_coefficient = env_config["rew_power_coef"]

        # NOTE: body pos reward, body rot reward, body vel reward, body ang vel reward
        self._imitation_reward_dim = 4
        self.reward_specs = {
            "k_pos": 100,
            "k_rot": 10,
            "k_vel": 0.1,
            "k_ang_vel": 0.1,
            "w_pos": 0.5,
            "w_rot": 0.3,
            "w_vel": 0.1,
            "w_ang_vel": 0.1,
        }

    def _create_envs(self):
        self.backend.create_envs(
            self.num_envs, self.env_spacing, self._kp_scale, self._kd_scale, self._has_self_collision
        )

        # The same humanoid for all envs: limb lengths + masses per limb group, attached to every env
        masses = torch.tensor(self.backend.body_masses[0])
        limb_lengths = torch.norm(self.skeleton_trees[0].local_translation, dim=-1)
        limb_lengths = [limb_lengths[group].sum() for group in self.limb_weight_group]
        masses_per_group = [masses[group].sum() for group in self.limb_weight_group]
        limb_weight = torch.tensor(limb_lengths + masses_per_group)
        self.humanoid_limb_and_weights = limb_weight.repeat(self.num_envs, 1).to(self.device)
        print("Humanoid Weight", masses.sum().item())

        ### PD action scale: 1.2x the largest joint limit of each joint, capped at pi
        dof_prop = self.backend.dof_props
        lower, upper = (np.asarray(dof_prop[k], dtype=np.float32) for k in ("lower", "upper"))
        limit = np.where((lower == 0) & (upper == 0), np.pi, np.maximum(np.abs(lower), np.abs(upper)))
        scale = np.repeat(np.minimum(1.2 * limit.reshape(-1, 3).max(axis=1), np.pi), 3).astype(np.float32)
        self._pd_action_scale = to_torch(scale, device=self.device)

        # ZL: Modified SMPL to give stronger knee
        for name in ("L_Knee", "R_Knee"):
            self._pd_action_scale[self._dof_names.index(name) * 3 + 1] = 5

        # Hands and toes are frozen at 0: no reliable motion data (see remove_names)
        for name in self.remove_names:
            i = self._dof_names.index(name) * 3
            self._pd_action_scale[i : i + 3] = 0

    def _define_gym_spaces(self):
        ### Observations
        # Self obs: height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        self._num_self_obs = 1 + self.num_bodies * (3 + 6 + 3 + 3) - 3

        # Task obs: 24 per body (pos, rot, vel, ang vel diffs + ref pos, rot), see compute_imitation_observations_v6
        self._task_obs_size = self.num_bodies * 24

        self.num_obs = self._num_self_obs + self._task_obs_size  # = 934
        assert self.num_obs == 934

        # AMP obs
        self._dof_obs_size = len(self._dof_names) * 6
        self._num_amp_obs_per_step = (
            13 + self._dof_obs_size + self.num_dof + 3 * len(self.key_bodies)
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        if self._has_dof_subset:
            self._num_amp_obs_per_step -= (6 + 3) * int((self.num_dof - len(self.dof_subset)) / 3)

        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step

        self.num_actions = self.num_dof

        ### Gym/puffer spaces
        self.single_observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf, dtype=np.float32
        )
        self.amp_observation_space = spaces.Box(
            np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf, dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0, dtype=np.float32
        )

    def _setup_gym_tensors(self):
        b = self.backend
        self.dof_force_tensor = b.dof_force

        self._humanoid_root_states = b.root_states

        self._dof_pos = b.dof_pos
        self._dof_vel = b.dof_vel

        self._rigid_body_pos = b.rigid_body_pos
        self._rigid_body_rot = b.rigid_body_rot
        self._rigid_body_vel = b.rigid_body_vel
        self._rigid_body_ang_vel = b.rigid_body_ang_vel

    def _setup_env_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # NOTE: store indiviaul reward components: the 4 imitation terms + power
        self.reward_raw = torch.zeros((self.num_envs, self._imitation_reward_dim + 1)).to(self.device)

        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.short)

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)  # This is dones
        # _terminate_buf records early termination
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        self.extras = {}  # Stores info

        # AMP/Motion-related
        self._reset_default_env_ids = []
        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        # reset() resets every env to its motion's start state: what _init_amp_obs and the tests read
        self._reset_ref_env_ids = self._reset_ref_motion_ids = self.all_env_ids

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        # amp_obs_demo_buf is fed into the discriminator training as the real motion data
        # This replaces the demo replay buffer in the original PHC code
        # amp_batch_size is fixed to the number of envs
        self._amp_obs_demo_buf = torch.zeros_like(self._amp_obs_buf)

    def _load_motion(self, motion_train_file):
        motion_lib_cfg = SimpleNamespace(
            motion_file=motion_train_file,
            device=self.device,
            fix_height=FixHeightMode.full_fix,
            min_length=self._min_motion_len,
            im_eval=False,
            smpl_type=self.humanoid_type,
            step_dt=self.dt,
        )
        shapes, limb_weights = self.humanoid_shapes.cpu(), self.humanoid_limb_and_weights
        self._motion_train_lib = MotionLibSMPL(motion_lib_cfg, self.skeleton_trees[0], shapes, limb_weights)
        self._motion_lib = self._motion_train_lib

        # Shares the train lib's processed motions. A cfg copy: the train lib keeps a reference to motion_lib_cfg,
        # and im_eval turns off its heading augmentation
        self._motion_eval_lib = copy.copy(self._motion_train_lib)
        self._motion_eval_lib.m_cfg = SimpleNamespace(**{**vars(motion_lib_cfg), "im_eval": True})

        cfg, M = self.cfg["env"], self.num_unique_motions
        self.motion_sampler = MotionSampler(M, cfg["motion_sampling"], self.device, cfg["pmcp_alpha"])
        target = cfg["start_sampling_target"]
        self.start_sampler = StartSampler(M, cfg["start_sampling"], self.num_envs, self.device, target=target)
        self._motion_key_ids = {k: i for i, k in enumerate(self._motion_train_lib._motion_data_keys)}
        self._motion_lib.load_motions(self.motion_sampler.sample(self.num_envs))

    #####################################################################
    ### reset()
    #####################################################################

    def _reset_masked(self, mask):
        """Reset the envs where mask is set to their motion's start state, as full-batch ops: no host sync."""
        ids = self.all_env_ids
        if self._state_init == StateInit.Random:
            times = self._sample_time(ids, mask)
        else:
            times = torch.zeros_like(self._motion_start_times)
        res = self._motion_lib.get_motion_state(ids, times)
        m = mask[:, None]
        root = torch.cat([res["root_pos"], res["root_rot"], res["root_vel"], res["root_ang_vel"]], -1)
        self._humanoid_root_states[:] = torch.where(m, root, self._humanoid_root_states)
        self._dof_pos[:] = torch.where(m, res["dof_pos"], self._dof_pos)
        self._dof_vel[:] = torch.where(m, res["dof_vel"], self._dof_vel)
        self._motion_start_times[:] = torch.where(mask, times, self._motion_start_times)
        self.backend.reset_envs(mask)
        for buf in (self.progress_buf, self.reset_buf, self._terminate_buf):
            buf.masked_fill_(mask, 0)
        self.backend.refresh_reset(mask, res)  # rigid body state of the reset envs comes from the sim's FK

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            raise NotImplementedError("Not tested yet")

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times)

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -self.dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        amp_obs_demo = self._get_amp_obs(motion_ids, motion_times)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)

        # amp_obs_demo_buf is fed into the discriminator training as the real motion data
        self._amp_obs_demo_buf[env_ids] = self._amp_obs_buf[env_ids]

    def _get_amp_obs(self, motion_ids, motion_times):
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times)
        key_pos = motion_res["rg_pos"][:, self._key_body_ids]
        key_vel = motion_res["body_vel"][:, self._key_body_ids]

        return self._compute_amp_observations_from_state(
            motion_res["root_pos"],
            motion_res["root_rot"],
            motion_res["root_vel"],
            motion_res["root_ang_vel"],
            motion_res["dof_pos"],
            motion_res["dof_vel"],
            key_pos,
            key_vel,
            motion_res["motion_bodies"],
            motion_res["motion_limb_weights"],
            self.dof_subset,
        )

    def _sample_time(self, motion_ids, mask=None):
        if self.flag_test:
            return torch.zeros_like(motion_ids, dtype=torch.float)
        return self.start_sampler.start_times(self._motion_lib, motion_ids, mask)

    #####################################################################
    ### compute observations
    #####################################################################

    def _compute_observations(self):
        # This is the normalized state of the humanoid
        state = self._compute_humanoid_obs()

        # This is the difference of state with the demo, but it is
        # called "state" in the paper.
        imitation = self._compute_task_obs()

        # Possible the original paper only uses imitation
        obs = torch.cat([state, imitation], dim=-1)
        self.obs_buf[:] = obs

        return obs

    def _compute_humanoid_obs(self):
        with torch.no_grad():
            return compute_humanoid_observations_smpl_max(
                self._rigid_body_pos,
                self._rigid_body_rot,
                self._rigid_body_vel,
                self._rigid_body_ang_vel,
            )

    def _compute_task_obs(self):
        # Next frame, so +1
        motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times

        motion_res = self._motion_lib.get_motion_state(self.all_env_ids, motion_times)

        self._ref_body_pos = motion_res["rg_pos"]  # HumanoidRenderEnv draws these
        body_pos = self._rigid_body_pos
        return compute_imitation_observations_v6(
            body_pos[:, 0, :],
            self._rigid_body_rot[:, 0, :],
            body_pos,
            self._rigid_body_rot,
            self._rigid_body_vel,
            self._rigid_body_ang_vel,
            motion_res["rg_pos"],  # ref_rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]

        # assert self.humanoid_type == "smpl"
        if self.dof_subset is None:
            # ZL hack
            (
                self._dof_pos[:, 9:12],
                self._dof_pos[:, 21:24],
                self._dof_pos[:, 51:54],
                self._dof_pos[:, 66:69],
            ) = 0, 0, 0, 0
            (
                self._dof_vel[:, 9:12],
                self._dof_vel[:, 21:24],
                self._dof_vel[:, 51:54],
                self._dof_vel[:, 66:69],
            ) = 0, 0, 0, 0

        if env_ids is None:
            # TODO: revisit constant args
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_from_state(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                key_body_vel,
                self.humanoid_shapes,
                self.humanoid_limb_and_weights,
                self.dof_subset,
            )
        else:
            if len(env_ids) == 0:
                return

            self._curr_amp_obs_buf[env_ids] = self._compute_amp_observations_from_state(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                key_body_vel[env_ids],
                self.humanoid_shapes[env_ids],
                self.humanoid_limb_and_weights[env_ids],
                self.dof_subset,
            )

    def _compute_amp_observations_from_state(
        self,
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        key_body_pos,
        key_body_vels,
        smpl_params,
        limb_weight_params,
        dof_subset,
    ):
        smpl_params = smpl_params[:, :-6]

        # TODO: revisit constant args
        return build_amp_observations_smpl(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_body_pos,
            smpl_params,
            limb_weight_params,
            dof_subset,
            self._local_root_obs,  # Constant: True
            self._amp_root_height_obs,  # Constant: True
            self._has_dof_subset,  # Constant: True
            self._has_shape_obs_disc,  # Constant: False
            self._has_limb_weight_obs_disc,  # Constant: False
            self._has_upright_start,  # Constant: True
        )

    #####################################################################
    ### step() -- pre_physics_step(), post_physics_step()
    #####################################################################

    def _compute_reward(self, motion_res):
        self.rew_buf[:], self.reward_raw[:, : self._imitation_reward_dim] = compute_imitation_reward(
            self._rigid_body_pos,
            self._rigid_body_rot,
            self._rigid_body_vel,
            self._rigid_body_ang_vel,
            motion_res["rg_pos"],  # ref_rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
            self.reward_specs,
        )

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
        power_reward = -self.power_coefficient * power
        # First 3 frame power reward should not be counted. since they could be dropped.
        power_reward.masked_fill_(self.progress_buf <= 3, 0)

        self.rew_buf[:] += power_reward
        self.reward_raw[:, -1] = power_reward

    def _compute_reset(self, motion_res):
        # Reset is also called after the progress_buf is updated.
        time = self.progress_buf * self.dt + self._motion_start_times
        # Reaching the step cap is a truncation, not a failure
        pass_time = (time >= self._motion_lib._motion_lengths) | (self.progress_buf >= self._step_cap)

        body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :]
        ref_body_pos = motion_res["rg_pos"][..., self._reset_bodies_id, :]

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_im_reset(
            self.reset_buf,
            self.progress_buf,
            body_pos,
            ref_body_pos,
            pass_time,
            self._termination_distances[..., self._reset_bodies_id],
            self.flag_test,
        )

    # NOTE: Training/eval code changes the termination distances.
    def set_termination_distances(self, termination_distances):
        self._termination_distances[:] = termination_distances

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            # Got RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0 : (self._num_amp_obs_steps - 1)]
            except:  # noqa
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0 : (self._num_amp_obs_steps - 1)].clone()

        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0 : (self._num_amp_obs_steps - 1)]

    #####################################################################
    ### Motion/AMP
    #####################################################################

    @property
    def amp_obs(self):
        return self._amp_obs_buf.view(-1, self.num_amp_obs) if self.use_amp_obs else None

    def fetch_amp_obs_demo(self):
        return self._amp_obs_demo_buf.view(-1, self.num_amp_obs) if self.use_amp_obs else None

    def resample_motions(self):
        if self.flag_test:
            self.forward_motion_samples()

        else:
            self._motion_lib.load_motions(self.motion_sampler.sample(self.num_envs))

            self.reset()

    def forward_motion_samples(self, first=False):
        """For evaluation: load the next (or with first, the first) num_envs motions, shortest first."""
        self._motion_sample_start_idx = 0 if first else self._motion_sample_start_idx + self.num_envs
        lib = self._motion_lib
        order = torch.from_numpy(np.argsort(lib._frames / lib._fps, kind="stable")).to(self.device)
        ids = torch.arange(self.num_envs, device=self.device) + self._motion_sample_start_idx
        lib.load_motions(order[ids % self.num_unique_motions])
        self.reset()

    @property
    def num_unique_motions(self):
        return self._motion_lib._num_unique_motions

    @property
    def current_motion_ids(self):
        return self._motion_lib._curr_motion_ids

    @property
    def motion_sample_start_idx(self):
        return self._motion_sample_start_idx

    @property
    def motion_data_keys(self):
        return self._motion_lib._motion_data_keys

    def get_motion_steps(self):
        return self._motion_lib.get_motion_num_steps().clamp(max=self._step_cap)

    #####################################################################
    ### Toggle train/eval model. Used in the training/eval code
    #####################################################################
    def toggle_eval_mode(self):
        self.flag_test = True

        # Relax the early termination condition for evaluation
        self.set_termination_distances(0.5)  # NOTE: hardcoded
        self._step_cap = 1200  # eval runs each motion from frame 0 to its end or this cap

        self._motion_lib = self._motion_eval_lib
        self.forward_motion_samples(first=True)  # using _motion_eval_lib
        self._reset_bodies_id = self._eval_track_bodies_id  # Following UHC

        # Return the number of motions
        return self._motion_lib._num_unique_motions

    def untoggle_eval_mode(self, failed_keys):
        self.flag_test = False

        self.set_termination_distances(self._termination_distances_backup)
        self._step_cap = self.max_episode_length
        self._motion_lib = self._motion_train_lib
        self._reset_bodies_id = self._reset_bodies_id_backup
        self.motion_sampler.record_eval([self._motion_key_ids[k] for k in failed_keys])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))  # SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))


def compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel):
    root_pos = body_pos[:, 0, :]
    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(body_rot[:, 0, :])

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    return torch.cat([root_h, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim=-1)


@torch.jit.script
def compute_imitation_observations_v6(
    root_pos,
    root_rot,
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    B, J, _ = body_pos.shape

    heading_inv_rot = calc_heading_quat_inv(root_rot)
    heading_rot = calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).view(-1, 4)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).view(-1, 4)

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos - body_pos
    diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand, diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = quat_mul(ref_body_rot, quat_conjugate(body_rot))
    diff_local_body_rot_flat = quat_mul(
        quat_mul(heading_inv_rot_expand, diff_global_body_rot.view(-1, 4)),
        heading_rot_expand,
    )  # Need to be change of basis

    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel - body_vel
    diff_local_vel = my_quat_rotate(heading_inv_rot_expand, diff_global_vel.view(-1, 3))

    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_local_ang_vel = my_quat_rotate(heading_inv_rot_expand, diff_global_ang_vel.view(-1, 3))

    ##### body pos + Dof_pos This part will have proper futures.
    local_ref_body_pos = ref_body_pos - root_pos.view(B, 1, 3)  # preserves the body position
    local_ref_body_pos = my_quat_rotate(heading_inv_rot_expand, local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = quat_mul(heading_inv_rot_expand, ref_body_rot.view(-1, 4))
    local_ref_body_rot = quat_to_tan_norm(local_ref_body_rot)

    obs = [
        diff_local_body_pos_flat.view(B, -1),  # 24 * 3
        quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1),  # 24 * 6
        diff_local_vel.view(B, -1),  # 24 * 3
        diff_local_ang_vel.view(B, -1),  # 24 * 3
        local_ref_body_pos.view(B, -1),  # 24 * 3
        local_ref_body_rot.view(B, -1),  # 24 * 6
    ]
    return torch.cat(obs, dim=-1)


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = quat_to_tan_norm(exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert (num_joints * joint_obs_size) == joint_dof_obs.shape[1]

    return joint_dof_obs


@torch.jit.script
def build_amp_observations_smpl(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    shape_params,
    limb_weight_params,
    dof_subset,
    local_root_obs,
    root_height_obs,
    has_dof_subset,
    has_shape_obs_disc,
    has_limb_weight_obs,
    upright,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, bool) -> Tensor
    B, N = root_pos.shape
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot_inv, root_rot)
    else:
        root_rot_obs = root_rot

    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot_inv, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot_inv, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2]
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2]
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2]
    )

    if has_dof_subset:
        dof_vel = dof_vel[:, dof_subset]
        dof_pos = dof_pos[:, dof_subset]

    dof_obs = dof_to_obs_smpl(dof_pos)
    obs_list = []
    if root_height_obs:
        obs_list.append(root_h)
    obs_list += [
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos,
    ]
    # 1? + 6 + 3 + 3 + 114 + 57 + 12
    if has_shape_obs_disc:
        obs_list.append(shape_params)
    if has_limb_weight_obs:
        obs_list.append(limb_weight_params)
    obs = torch.cat(obs_list, dim=-1)

    return obs


@torch.jit.script
def compute_imitation_reward(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
    rwd_specs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = (
        rwd_specs["k_pos"],
        rwd_specs["k_rot"],
        rwd_specs["k_vel"],
        rwd_specs["k_ang_vel"],
    )
    w_pos, w_rot, w_vel, w_ang_vel = (
        rwd_specs["w_pos"],
        rwd_specs["w_rot"],
        rwd_specs["w_vel"],
        rwd_specs["w_ang_vel"],
    )

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = quat_mul(ref_body_rot, quat_conjugate(body_rot))
    diff_global_body_angle = quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)

    return reward, reward_raw


@torch.jit.script
def compute_humanoid_im_reset(
    reset_buf,
    progress_buf,
    rigid_body_pos,
    ref_body_pos,
    pass_time,
    termination_distance,
    use_mean,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor]
    # NOTE: When evaluating, using mean is relaxed vs. max is strict.
    if use_mean:
        has_fallen = torch.any(
            torch.norm(rigid_body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0],
            dim=-1,
        )  # using average, same as UHC"s termination condition
    else:
        has_fallen = torch.any(
            torch.norm(rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1
        )  # using max

    # first timestep can sometimes still have nonzero contact forces
    # so only check after first couple of steps
    terminated = has_fallen * (progress_buf > 1)
    reset = pass_time | terminated

    return reset, terminated
