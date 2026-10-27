# ruff: noqa: E402  # isaacgym's backend-peek must run before any heavier (torch-importing) import below
import os
import gc
import ast
import sys
import uuid
import time
import math
import json
import argparse
import configparser
from datetime import datetime

import joblib
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

from puffer_phc.backend import preload_isaacgym


def _peek_backend(argv):
    """Peek at [env] backend (config.ini / --config / --env.backend) before any heavier import runs:
    isaacgym must be imported before torch (see puffer_phc.backend.preload_isaacgym)."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default="config.ini")
    peek.add_argument("--env.backend", dest="backend", default=None)
    known = peek.parse_known_args(argv)[0]
    if known.backend is not None:
        return known.backend
    p = configparser.ConfigParser()
    p.read(known.config)
    return p.get("env", "backend", fallback="mujoco_gpu").strip("\"'")


preload_isaacgym(_peek_backend(sys.argv[1:]))

import torch
import numpy as np

import pufferlib
import pufferlib.cleanrl
import pufferlib.vector

from puffer_phc import clean_pufferl
from puffer_phc.environment import make as env_creator
import puffer_phc.policy as policy_module


METRICS = ("mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist")  # smpl_sim compute_metrics_lite's keys


def p_mpjpe(pred, gt):
    """Batched torch port of smpl_sim's p_mpjpe (MPJPE after scale, rotation and translation alignment): (B, J)."""
    mu_x, mu_y = gt.mean(1, keepdim=True), pred.mean(1, keepdim=True)
    x0, y0 = gt - mu_x, pred - mu_y
    nx, ny = x0.square().sum((1, 2), keepdim=True).sqrt(), y0.square().sum((1, 2), keepdim=True).sqrt()
    U, s, Vt = torch.linalg.svd((x0 / nx).transpose(1, 2) @ (y0 / ny))
    V = Vt.transpose(1, 2)
    sign = torch.linalg.det(V @ U.transpose(1, 2)).sign()
    V = torch.cat([V[..., :2], V[..., 2:] * sign[:, None, None]], -1)
    s = torch.cat([s[:, :2], s[:, 2:] * sign[:, None]], -1)
    R = V @ U.transpose(1, 2)
    a = s.sum(1)[:, None, None] * nx / ny
    return (a * (pred @ R) + (mu_x - a * (mu_y @ R)) - gt).norm(dim=-1)


class EvalMetrics:
    """Per-env sums and frame counts of compute_metrics_lite's errors (m), fed in chunks of frames."""

    def __init__(self, num_envs, device):
        self.sums = torch.zeros(num_envs, len(METRICS), dtype=torch.float64, device=device)
        self.counts = torch.zeros(num_envs, len(METRICS), dtype=torch.long, device=device)
        self.prev = None  # last two (pred, gt) frames, for velocity and acceleration
        self.t = 0

    def update(self, pred, gt, valid):
        """pred, gt: (T, num_envs, J, 3); valid: (T, num_envs), the frames compute_metrics_lite gets."""
        if self.prev is None:
            self.prev = pred[[0, 0]], gt[[0, 0]]
        P, G = torch.cat([self.prev[0], pred]), torch.cat([self.prev[1], gt])
        self.prev = P[-2:], G[-2:]
        pl, gl = pred - pred[:, :, :1], gt - gt[:, :, :1]
        err = torch.stack([
            (gt - pred).norm(dim=-1),
            (pl - gl).norm(dim=-1),
            p_mpjpe(pl.flatten(0, 1), gl.flatten(0, 1)).view(pl.shape[:3]),
            ((P[:-2] - 2 * P[1:-1] + P[2:]) - (G[:-2] - 2 * G[1:-1] + G[2:])).norm(dim=-1),
            ((P[2:] - P[1:-1]) - (G[2:] - G[1:-1])).norm(dim=-1),
        ], -1).mean(-2)  # fmt: skip
        t = torch.arange(self.t, self.t + len(valid), device=valid.device)[:, None]
        self.t += len(valid)
        mask = torch.stack([valid, valid, valid, valid & (t >= 2), valid & (t >= 1)], -1)
        self.sums += torch.where(mask, err.double(), 0).sum(0)
        self.counts += mask.sum(0)


def reduce_metrics(sums, counts):
    """Frame-weighted means in mm, as np.mean over compute_metrics_lite's concatenated frames."""
    return dict(zip(METRICS, (sums.sum(0) / counts.sum(0) * 1000).tolist()))


class EvalStats:
    def __init__(self, vec_env, failed_save_path=None):
        self.task_env = vec_env.env
        self.num_envs = self.task_env.num_envs
        self.failed_save_path = failed_save_path

        # Prep the env for evaluation
        self.num_unique_motions = self.task_env.toggle_eval_mode()

        self.batches = []
        self.success_rate = 0
        self.failed_keys = []
        self.results = None
        self.results_by_motion = None
        self.new_batch()

        self.pbar = tqdm(total=-(-self.num_unique_motions // self.num_envs), desc="Eval batches")

    def new_batch(self):
        device = self.task_env.device
        self.curr_steps = 0
        self.running = torch.ones((), dtype=torch.bool, device=device)
        self.terminate_state = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.played_steps_buf = torch.zeros(self.num_envs, dtype=torch.short, device=device)
        self.metrics = EvalMetrics(self.num_envs, device)
        self.frames = []

    def post_step_eval(self):
        """Tracks the batch on the GPU and reads it back every 16 steps. Steps past the batch's end change nothing
        for the counted envs: none is alive with steps left."""
        env, t = self.task_env, self.curr_steps
        motion_num_steps = env.get_motion_steps()
        info = env.extras
        n = self.num_unique_motions - env.motion_sample_start_idx  # envs past n repeat earlier motions

        # A motion's frames count up to and including the one it terminates on
        self.frames.append((info["body_pos"], info["body_pos_gt"], ~self.terminate_state & (t < motion_num_steps - 1)))
        # If terminate after the last frame, then it is not a termination. curr_step is one step behind simulation.
        self.terminate_state |= (t < motion_num_steps) & info["terminate"]
        self.played_steps_buf += ~self.terminate_state & (t < motion_num_steps)
        # The batch ends once no counted env is alive with frames left to play
        self.running &= (~self.terminate_state[:n] & (motion_num_steps[:n] > t + 1)).any()
        self.curr_steps += 1

        if self.curr_steps % 16:
            return False, False
        self.metrics.update(*map(torch.stack, zip(*self.frames)))
        self.frames = []
        if self.running.item():
            return False, False

        m = self.metrics
        cols = env.current_motion_ids, self.terminate_state, motion_num_steps, self.played_steps_buf, m.sums, m.counts
        self.batches.append([c[:n] for c in cols])
        self.pbar.update(1)

        # All motions have been fully evaluated
        if env.motion_sample_start_idx + self.num_envs >= self.num_unique_motions:
            return self.get_final_stats(), False

        # Move on to the next motion
        env.forward_motion_samples()
        self.new_batch()
        return False, True

    def get_final_stats(self):
        self.pbar.clear()
        ids, *cols = (torch.cat(c).cpu().numpy() for c in zip(*self.batches))
        terminated, motion_length, played_steps, sums, counts = (c[np.argsort(ids)] for c in cols)  # dataset order

        self.success_rate = 1 - terminated.mean()
        self.failed_keys = self.task_env.motion_data_keys[terminated]

        metrics_all_print = reduce_metrics(sums, counts)
        if terminated.all():
            print("No success!!!")
            metrics_succ_print = metrics_all_print
        else:
            metrics_succ_print = reduce_metrics(sums[~terminated], counts[~terminated])

        print("------------------------------------------")
        print(f"Success Rate: {self.success_rate:.10f}")
        print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
        print("Succ: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
        print(f"Failed: {len(self.failed_keys)} motions (keys in failed_*.pkl)")

        self.results = {
            "eval/success_rate": float(self.success_rate),
            "eval/mpjpe_all": metrics_all_print["mpjpe_g"],
            "eval/mpjpe_succ": metrics_succ_print["mpjpe_g"],
            "eval/accel_dist": metrics_succ_print["accel_dist"],
            "eval/vel_dist": metrics_succ_print["vel_dist"],
            "eval/mpjpel_all": metrics_all_print["mpjpe_l"],
            "eval/mpjpel_succ": metrics_succ_print["mpjpe_l"],
            "eval/mpjpe_pa": metrics_succ_print["mpjpe_pa"],
        }

        self.results_by_motion = {
            "motion_keys": self.task_env.motion_data_keys.tolist(),
            "motion_length": motion_length,
            "played_steps": played_steps,
            "success": ~terminated,
        }

        return True

    def update_env_and_close(self):
        # NOTE: Assuming that resampling motion will happen right after the eval,
        # so not resetting the env here.
        self.task_env.untoggle_eval_mode(self.failed_keys)

        torch.cuda.empty_cache()
        gc.collect()

        if self.failed_save_path:
            env = self.task_env
            m, s, keys = env.motion_sampler, env.start_sampler, env._motion_train_lib._motion_data_keys
            joblib.dump(dict(failed_keys=self.failed_keys, motion_keys=keys, h=m.h, last=m.last, load_count=m.load_count,
                             att=s.att.cpu().numpy(), succ=s.succ.cpu().numpy()), self.failed_save_path)  # fmt: skip

        return self.results


def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args["policy"])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args["rnn"])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args["train"]["device"])


def init_run(args):
    import aim

    exp_id = args["env_name"] + "-" + str(uuid.uuid4())[:8]
    run = aim.Run(
        repo=args["train"]["data_dir"],
        experiment=args["track_project"],
        system_tracking_interval=None,
        log_system_params=False,
    )
    run.name = exp_id
    run["hparams"] = args
    if args["tag"] is not None:
        run.add_tag(args["tag"])
    return run, exp_id


def train(args, vec_env, policy, run=None, exp_id=None, skip_resample=False, final_eval=False):
    if run is None and args["track"]:
        run, exp_id = init_run(args)

    if exp_id is None:
        exp_id = args["env_name"] + "-" + str(uuid.uuid4())[:8]

    train_config = pufferlib.namespace(**args["train"], env=args["env_name"], exp_id=exp_id)
    data = clean_pufferl.create(train_config, vec_env, policy, run=run)

    data_dir = os.path.join(train_config.data_dir, exp_id)
    os.makedirs(data_dir, exist_ok=True)

    while data.global_step < train_config.total_timesteps:
        if not skip_resample and data.epoch > 0 and data.epoch % train_config.motion_resample_interval == 0:
            # Evaluate all motions, then resample with the updated failure weights
            eval_stats = EvalStats(vec_env, failed_save_path=os.path.join(data_dir, f"failed_{data.epoch:06d}.pkl"))
            rollout(vec_env, policy, eval_stats)
            eval_results = eval_stats.update_env_and_close()
            if data.run:
                eval_results["0verview/agent_steps"] = data.global_step
                eval_results["0verview/epoch"] = data.epoch
                clean_pufferl.log_dict(run, eval_results, data.global_step)

            vec_env.env.resample_motions()

            # Reset the envs and lstm hidden states
            vec_env.reset()
            if data.experience.lstm_h is not None:
                data.experience.lstm_h[:] = 0
                data.experience.lstm_c[:] = 0

        # Collect data
        results, _ = clean_pufferl.evaluate(data)

        # Update obs running mean and std
        # During evaluate() and train(), the obs_norm is NOT updated.
        rms_update_fn = getattr(data.policy.policy, "update_obs_rms", None)
        if rms_update_fn:
            rms_update_fn(data.experience.obs)

        amp_rms_update_fn = getattr(data.policy.policy, "update_amp_obs_rms", None)
        if data.use_amp_obs and amp_rms_update_fn:
            amp_rms_update_fn(data.experience.amp_obs)

        # Update policy
        clean_pufferl.train(data)

        # Apply learning rate exp decay
        if data.config.lr_decay_rate > 0:
            decay = math.exp(-data.config.lr_decay_rate * data.epoch)
            if decay < data.config.lr_decay_floor:
                decay = data.config.lr_decay_floor
            for g in data.optimizer.param_groups:
                g["lr"] = data.config.learning_rate * decay

    uptime = data.profile.uptime

    # Final evaluation
    if final_eval:
        eval_stats = EvalStats(vec_env)
        rollout(vec_env, policy, eval_stats)
        results.update(eval_stats.update_env_and_close())
        if data.run:
            results["0verview/agent_steps"] = data.global_step
            results["0verview/epoch"] = data.epoch
            clean_pufferl.log_dict(run, results, data.global_step)

    clean_pufferl.close(data)

    return results, uptime


def rollout(vec_env, policy, eval_stats=None):
    # NOTE (Important): Using deterministic action for evaluation
    policy.policy.set_deterministic_action(True)  # Ugly... but...

    obs, _ = vec_env.reset()
    state = None

    ep_cnt = 0
    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device)
            if hasattr(policy, "lstm"):
                action, _, _, _, state = policy(obs, state)
            else:
                action, _, _, _ = policy(obs)

            action = action.reshape(vec_env.action_space.shape)

        obs, _, done, trunc, info = vec_env.step(action)

        if hasattr(policy, "lstm"):
            # Reset lstm states for the reset
            reset_envs = torch.logical_or(done, trunc)
            if reset_envs.any():
                state[0][:, reset_envs] = 0
                state[1][:, reset_envs] = 0

        # Get episode-related info here
        if len(info) > 0 and not eval_stats:
            ep_ret = info[0]["episode_return"]
            ep_len = info[0]["episode_length"]
            print(f" Episode cnt: {vec_env.episode_count - ep_cnt}, Reward: {ep_ret:.3f}, Length: {ep_len:.3f}")
            ep_cnt = vec_env.episode_count

        if eval_stats:
            is_done, next_batch = eval_stats.post_step_eval()
            if is_done:
                policy.policy.set_deterministic_action(False)
                break

            if next_batch and state is not None:
                # Reset the states
                state[0][:] = 0
                state[1][:] = 0


### CARBS Sweeps
def sweep_carbs(args, sweep_count=500, max_suggestion_cost=3600):
    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam

    # from carbs import ParamDictType
    from carbs import Param

    def carbs_param(
        group,
        name,
        space,
        sweep_params,
        mmin=None,
        mmax=None,
        search_center=None,
        is_integer=False,
        rounding_factor=1,
        scale=1,
    ):
        sweep_param = sweep_params[group]["parameters"][name]
        if "values" in sweep_param:
            values = sweep_param["values"]
            mmin = min(values)
            mmax = max(values)

        if mmin is None:
            mmin = float(sweep_param["min"])
        if mmax is None:
            mmax = float(sweep_param["max"])

        if space == "log":
            Space = LogSpace
            if search_center is None:
                search_center = 2 ** (np.log2(mmin) + np.log2(mmax) / 2)
        elif space == "linear":
            Space = LinearSpace
            if search_center is None:
                search_center = (mmin + mmax) / 2
        elif space == "logit":
            Space = LogitSpace
            assert mmin == 0
            assert mmax == 1
            assert search_center is not None
        else:
            raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

        return Param(
            name=f"{group}/{name}",
            space=Space(
                min=mmin,
                max=mmax,
                is_integer=is_integer,
                rounding_factor=rounding_factor,
                scale=scale,
            ),
            search_center=search_center,
        )

    target_metric = args["sweep"]["metric"]["name"].split("/")[-1]
    sweep_parameters = args["sweep"]["parameters"]

    # Must be hardcoded and match the sweep space in config.ini for now
    param_spaces = []
    if "total_timesteps" in sweep_parameters["train"]["parameters"]:
        time_param = sweep_parameters["train"]["parameters"]["total_timesteps"]
        min_timesteps = time_param["min"]
        param_spaces.append(
            carbs_param(
                "train", "total_timesteps", "log", sweep_parameters, search_center=min_timesteps, is_integer=True
            )
        )

    param_spaces += [
        carbs_param("train", "learning_rate", "log", sweep_parameters, search_center=args["ssc_lr"]),
        carbs_param("train", "gae_lambda", "logit", sweep_parameters, search_center=0.50),
        carbs_param("train", "update_epochs", "linear", sweep_parameters, search_center=3, is_integer=True),
        carbs_param("train", "clip_coef", "logit", sweep_parameters, search_center=0.1),
        carbs_param("train", "vf_coef", "linear", sweep_parameters, search_center=2.0),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=5,
        num_random_samples=len(param_spaces),
        max_suggestion_cost=max_suggestion_cost,
        is_saved_on_every_observation=False,
    )
    carbs = CARBS(carbs_params, param_spaces)

    def main():
        # set torch and pytorch seeds to current time
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))

        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print("Suggestion:", suggestion)
        train_suggestion = {k.split("/")[1]: v for k, v in suggestion.items() if k.startswith("train/")}
        env_suggestion = {k.split("/")[1]: v for k, v in suggestion.items() if k.startswith("env/")}
        args["train"].update(train_suggestion)

        args["env"].update(env_suggestion)
        args["track"] = True
        run, exp_id = init_run(args)

        print(args["train"])
        print(args["env"])
        print(args["policy"])

        try:
            vec_env = pufferlib.vector.make(env_creator, env_kwargs=args["env"])
            policy_cls = getattr(policy_module, args["policy_name"])
            rnn_cls = None
            if "rnn_name" in args:
                rnn_cls = getattr(policy_module, args["rnn_name"])
            policy = make_policy(vec_env.driver_env, policy_cls, rnn_cls, args)

            stats, uptime = train(
                args, vec_env, policy, run, exp_id, skip_resample=args["skip_resample"], final_eval=args["final_eval"]
            )

        except Exception:
            import traceback

            traceback.print_exc()

        else:
            observed_value = 0
            for k, v in stats.items():
                if k.endswith(target_metric):
                    observed_value = v
                    break

            print("Observed value:", observed_value)
            print("Uptime:", uptime)

            carbs.observe(
                ObservationInParam(
                    input=orig_suggestion,
                    output=observed_value,
                    cost=uptime,
                )
            )

    for _ in range(sweep_count):
        main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--mode", type=str, default="train", choices="train eval play sweep".split())
    parser.add_argument(
        "-m", "--motion-file", type=str, default="sample_data/cmu_mocap_05_06.pkl", help="Path to motion file"
    )
    parser.add_argument("-c", "--checkpoint-path", type=str, default=None, help="Path to a pretrained checkpoint")
    # default="ckpt_amass_800m_acc97.pt")
    parser.add_argument("--track", action="store_true", help="Track with Aim")
    parser.add_argument("--track-project", type=str, default="pufferlib")
    parser.add_argument("--tag", type=str, default=None, help="Aim tag; required for --mode sweep")
    parser.add_argument("--ssc-lr", type=float, default=0.0001, help="Sweep search center for learning rate")
    parser.add_argument("--skip-resample", action="store_true", help="Skip resampling motions")
    parser.add_argument("--final-eval", action="store_true", help="Final evaluation")

    args = parser.parse_known_args()[0]

    if args.mode == "sweep" and not args.tag:
        parser.error("--tag is required for --mode sweep")

    p = configparser.ConfigParser()
    p.read(args.config)
    # [<section>.<backend>] overrides [<section>] for that backend, e.g. gym-tuned values for isaacgym
    backend = _peek_backend(sys.argv[1:])
    for section in ("env", "train"):
        if p.has_section(f"{section}.{backend}"):
            p[section].update(p[f"{section}.{backend}"])

    for section in p.sections():
        if section.startswith(("env.", "train.")):
            continue
        for key in p[section]:
            if section == "base":
                argparse_key = f"--{key}".replace("_", "-")
            else:
                argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    parsed = parser.parse_args().__dict__
    args = {"env": {}, "policy": {}, "rnn": {}}
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        if key == "tag":
            prev[subkey] = value
            continue
        try:
            prev[subkey] = ast.literal_eval(value)
        except:  # noqa
            prev[subkey] = value

    device = args["train"]["device"]

    # Create the environment
    args["env"]["name"] = args["env_name"]
    args["env"]["device_type"] = device
    if args["motion_file"]:
        args["env"]["motion_file"] = args["motion_file"]

    args["env"]["use_amp_obs"] = args["train"]["amp_reward_coef"] > 0

    # If play, change these env args
    if args["mode"] == "play":
        args["env"]["num_envs"] = 16
        args["env"]["headless"] = False

    # If sweep, run sweep here and exit
    if args["mode"] == "sweep":
        sweep_carbs(args, sweep_count=500)
        sys.exit(0)

    # Create the env and policy (seeded first, so motion sampling and initial weights are reproducible)
    clean_pufferl.seed_everything(args["train"]["seed"], args["train"]["torch_deterministic"])
    vec_env = pufferlib.vector.make(env_creator, env_kwargs=args["env"])
    policy_cls = getattr(policy_module, args["policy_name"])
    rnn_cls = None
    if "rnn_name" in args:
        rnn_cls = getattr(policy_module, args["rnn_name"])
    policy = make_policy(vec_env.driver_env, policy_cls, rnn_cls, args)

    if args["checkpoint_path"]:
        checkpoint = torch.load(args["checkpoint_path"], map_location=device, weights_only=False)
        res = policy.load_state_dict(checkpoint["state_dict"], strict=False)
        # only the discriminator may differ between an AMP and a non-AMP run
        assert all("disc" in k or "amp_obs_norm" in k for k in res.missing_keys + res.unexpected_keys), res
        print(f"Loaded checkpoint from {args['checkpoint_path']}")

    # Train or evaluate
    if args["mode"] == "train":
        train(args, vec_env, policy)

    elif args["mode"] == "play":
        # Just to play and render without collecting stats
        vec_env.env.set_termination_distances(10)
        rollout(vec_env, policy)

    elif args["mode"] == "eval":
        import polars as pl

        eval_stats = EvalStats(vec_env)
        rollout(vec_env, policy, eval_stats)

        out_dir = os.path.dirname(args["checkpoint_path"] or "")
        ts = datetime.now().strftime("%m%d_%H%M")
        with open(os.path.join(out_dir, f"eval_summary_{ts}.json"), "w") as f:
            json.dump(eval_stats.results, f, indent=4)

        df = pl.DataFrame(eval_stats.results_by_motion)
        df.write_csv(os.path.join(out_dir, f"results_by_motion_{ts}.tsv"), separator="\t")

        eval_stats.update_env_and_close()
