import functools

from puffer_phc.humanoid_phc import HumanoidPHC

import torch

import pufferlib


def env_creator(name="puffer_phc"):
    return functools.partial(make, name)


def make(name, **kwargs):
    """Backend "isaacgym" needs puffer_phc.backend.preload_isaacgym() called before torch is imported."""
    return PHCPufferEnv(name, **kwargs)


class PHCPufferEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        name,
        motion_file,
        has_self_collision,
        num_envs=32,
        device_type="cuda",
        exp_name="puffer_phc",
        clip_actions=True,
        device_id=0,
        headless=True,
        log_interval=32,
        rew_power_coef=0.0005,
        use_amp_obs=False,
        motion_sampling="pmcp",
        pmcp_alpha=0.2,
        start_sampling="sgs",
        start_sampling_target=0.5,
        termination_distance=0.25,
        kp_scale=1.0,
        kd_scale=1.0,
        backend="mujoco_gpu",
        solver_preset="isaacgym_parity_240",
    ):
        self.render_mode = "native"
        cfg = {
            "env": {
                "num_envs": num_envs,
                "motion_file": motion_file,
                "rew_power_coef": rew_power_coef,
                "use_amp_obs": use_amp_obs,
                "motion_sampling": motion_sampling,
                "pmcp_alpha": pmcp_alpha,
                "start_sampling": start_sampling,
                "start_sampling_target": start_sampling_target,
                "termination_distance": termination_distance,
                "kp_scale": kp_scale,
                "kd_scale": kd_scale,
                "backend": backend,
                "solver_preset": solver_preset,
            },
            "robot": {
                "has_self_collision": has_self_collision,
            },
            "exp_name": exp_name,
        }
        if headless or backend == "isaacgym":  # isaacgym play uses its backend viewer
            self.env = HumanoidPHC(cfg, device_type=device_type, device_id=device_id, headless=headless)
        else:
            from puffer_phc.render_env import HumanoidRenderEnv  # imports pyglet: play mode only

            self.env = HumanoidRenderEnv(cfg, device_type=device_type, device_id=device_id, headless=headless)

        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.num_agents = self.num_envs = self.env.num_envs
        self.clip_actions = clip_actions
        self.device = self.env.device

        self.use_amp_obs = use_amp_obs
        self.amp_observation_space = self.env.amp_observation_space if use_amp_obs else None

        # Check the buffer data types, match them to puffer
        buffers = pufferlib.namespace(
            observations=self.env.obs_buf,
            rewards=self.env.rew_buf,
            terminals=torch.zeros(self.num_agents, dtype=torch.bool, device=self.device),
            truncations=torch.zeros_like(self.env.reset_buf),
            masks=torch.ones_like(self.env.reset_buf),
            actions=torch.zeros(
                (self.num_agents, *self.single_action_space.shape), dtype=torch.float, device=self.device
            ),
        )

        super().__init__(buffers)

        self.log_interval = log_interval
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._episode_count = torch.zeros((), dtype=torch.int64, device=self.device)
        # sums over finished episodes since the last log: return, length, truncations, count
        self._ep_stats = torch.zeros(4, dtype=torch.float64, device=self.device)

        self.raw_rewards = torch.zeros(5, dtype=torch.float32, device=self.device)

    def reset(self, seed=None):
        self.tick = 0
        self.env.reset()

        self.amp_obs = self.env.amp_obs if self.use_amp_obs else None

        # Clear the buffers
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        self.actions[:] = 0
        self.raw_rewards[:] = 0
        self._ep_stats[:] = 0

        return self.observations, []

    def step(self, actions):
        """actions: torch tensor (any device) or numpy array."""
        self.actions[:] = torch.as_tensor(actions)
        if self.clip_actions:
            self.actions.clamp_(-1, 1)

        # obs, reward are put into the buffers; the done envs are reset in place
        done = self.env.step(self.actions)[2]

        self.amp_obs = self.env.amp_obs if self.use_amp_obs else None

        rew = self.rewards.clone()

        # Extract reward-related info for logging
        self.raw_rewards += self.env.extras["reward_raw"].mean(dim=0)

        # reset_buf flags the envs that are (early-) terminated or truncated.
        # Early-terminated envs are in self.env.extras["terminate"]
        # NOTE: Truncated does NOT mean all the all parts of the motion has been played out because
        # during reset, the initial frame is randomly selected, so it could start from the very end.
        self.terminals[:] = self.env.extras["terminate"]
        self.truncations[:] = done & ~self.terminals
        self.masks[:] = ~self.truncations
        stats = torch.stack([self.episode_returns, self.episode_lengths, self.truncations, done]).double()
        self._ep_stats += stats @ done.double()

        self._episode_count += done.sum()
        self.episode_returns.masked_fill_(done, 0)
        self.episode_lengths.masked_fill_(done, 0)

        self.episode_returns += self.rewards * ~self.env.reset_buf
        self.episode_lengths += ~self.env.reset_buf

        # TODO: self.env.extras has infos. Extract useful info?
        info = []
        self.tick += 1
        if self.tick % self.log_interval == 0:
            info = self.mean_and_log()

            # Extract reward-related info
            reward_info = {
                "rew_body_pos": self.raw_rewards[0].item() / self.log_interval,
                "rew_body_rot": self.raw_rewards[1].item() / self.log_interval,
                "rew_lin_vel": self.raw_rewards[2].item() / self.log_interval,
                "rew_ang_vel": self.raw_rewards[3].item() / self.log_interval,
                "rew_power": self.raw_rewards[4].item() / self.log_interval,
            }

            self.raw_rewards[:] = 0

            if len(info) > 0:
                info[0].update(reward_info)
            else:
                info.append(reward_info)

        return self.observations, rew, self.terminals, self.truncations, info

    @property
    def episode_count(self):
        return int(self._episode_count)  # syncs: read at log time only

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def mean_and_log(self):
        ret, length, trunc_rate, _ = (self._ep_stats / self._ep_stats[3]).tolist()  # nan if no episode ended
        info = {"episode_return": ret, "episode_length": length, "epi_trunc_rate": trunc_rate}
        self._ep_stats[:] = 0
        counts = self.env.start_sampler.counts
        info["sampling/probe_success"] = (counts[1] / counts[0]).item()  # nan if no probe ended
        info["sampling/pmcp_fail_mass"] = self.env.motion_sampler.fail_mass
        counts.zero_()
        self.env.backend.warn_overflow()  # once per log interval (= once per epoch at the default config)

        return [info]

    def fetch_amp_obs_demo(self):
        return self.env.fetch_amp_obs_demo()
