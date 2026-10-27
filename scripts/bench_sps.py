# ruff: noqa: E402  # isaacgym's backend-peek must run before any heavier (torch-importing) import below
"""Env-steps/sec harness and profiler, mainly for mujoco_gpu (isaacgym runs the plain SPS modes).

Modes (--mode, comma list; one step = one control step of all envs):
  physics  backend.simulate() x control_freq_inv (mujoco_gpu: graph launches + solver-stats reductions)
  backend  backend.step(targets): PD-target conversion + physics + refresh
  env      PHCPufferEnv.step on pre-generated seeded GPU action tensors (no numpy)
  env_np   PHCPufferEnv.step(action_space.sample()): numpy actions, H2D copy per step
  rollout  policy forward + vecenv recv/send as in clean_pufferl.evaluate, no PPO update (PPO side: train.py's Profile)
Each (N, preset) of the comma lists runs in its own process (backends are never freed) and appends one JSON line per
mode to --out. gpu_mem_gb: device memory taken since env creation, read after each mode (needs an idle GPU).
--profile, after all timing rows: per-phase GPU (CUDA event span) and CPU ms/step by call path, torch syncs by call
site (not warp's), then last (it slows the rest of the process) a 10-step torch.profiler trace per mode to
tmp/speed/trace_<mode>_<preset>_<N>.json. Phases are NVTX ranges for nsys (torch.profiler is empty under nsys):
    nsys profile --cuda-graph-trace=node -t cuda,nvtx,osrt -o tmp/speed/nsys_newton_120 \\
      python scripts/bench_sps.py --mode env --num-envs 4096 --solver-preset newton_120 --repeats 1 --profile
    nsys stats --report cuda_gpu_kern_sum tmp/speed/nsys_newton_120.nsys-rep
--record PATH / --check PATH (one config): obs/rew/done/root_states/dof_pos every 10 of 200 env-mode steps, saved
with the config (a mismatched --check is refused). mujoco_gpu is not bitwise deterministic run to run (fp32 noise
that chaos grows to ~0.1 by step 200): PASS iff each field's max abs diff at step 10 <= --tol, else exit 1.
"""

import argparse
import time

from puffer_phc.backend import preload_isaacgym


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["isaacgym", "mujoco_gpu", "mujoco_cpu"], default="mujoco_gpu")
    p.add_argument("--mode", default="physics,env")
    p.add_argument("--num-envs", default="4096")
    p.add_argument("--solver-preset", default="newton_120", help="mujoco backends only")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--record")
    p.add_argument("--check")
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--fixed-reset-times", action="store_true", help="one reset time per motion: RNG-free --record")
    p.add_argument("--checkpoint", help="policy weights for rollout mode (default: random init)")
    p.add_argument("--motion-file", default="data/amass_train_take6_upright.pkl")
    p.add_argument("--out", default="tmp/speed/bench.jsonl")
    return p.parse_args()


args = parse_args()
preload_isaacgym(args.backend)

import collections
import json
import os
import random
import subprocess
import sys
import warnings

import numpy as np
import pufferlib.cleanrl
import pufferlib.vector
import torch

from puffer_phc.environment import env_creator
from puffer_phc.humanoid_phc import HumanoidPHC
from puffer_phc.policy import PHCPolicy


class PhaseTimer:
    """Times wrapped methods per call path: CUDA event pairs (read after the loop, no sync inside) + CPU wall."""

    def __init__(self, env):
        e, b = env.env, env.env.backend
        nc = sys.modules.get("puffer_phc.backend._newton_mujoco")
        targets = [(env, "step", "PHCPufferEnv.step"), (e, "step", "HumanoidPHC.step")]
        targets += [(e, m, m) for m in ("_action_to_pd_targets", "_compute_reward", "_compute_reset", "_reset_actors")]
        targets += [(e, m, m) for m in ("_compute_observations", "_compute_humanoid_obs", "_compute_task_obs")]
        targets += [(e, "reset", "HumanoidPHC.reset"), (e, "_get_state_from_motionlib_cache", "_motion_cache")]
        targets += [(e._motion_lib, "get_motion_state", "motion_lib.get_motion_state")]
        targets += [(b, m, f"backend.{m}") for m in ("step", "simulate", "refresh", "reset_envs")]
        targets += [(nc, "pd_targets_to_newton", "nc.pd_targets_to_newton")] if nc else []
        self.calls, self.stack, restore = {}, [], [(o, a, getattr(o, a)) for o, a, _ in targets]
        self.unwrap = lambda: [setattr(*r) for r in restore]
        for owner, attr, name in targets:
            setattr(owner, attr, self._timed(getattr(owner, attr), name))

    def _timed(self, fn, name):
        def timed(*a, **k):
            self.stack.append(name)
            rec = self.calls.setdefault("/".join(self.stack), [])
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(name)
            e0.record()
            t0 = time.perf_counter()
            with torch.profiler.record_function(name):
                out = fn(*a, **k)
            e1.record()
            rec.append((e0, e1, time.perf_counter() - t0))
            torch.cuda.nvtx.range_pop()
            self.stack.pop()
            return out

        return timed

    def report(self, steps):
        cpu = {k: 1e3 * sum(c for *_, c in v) / steps for k, v in self.calls.items()}
        gpu = {k: sum(e0.elapsed_time(e1) for e0, e1, _ in v) / steps for k, v in self.calls.items()}
        order = {k: i for i, k in enumerate(self.calls)}
        keys = sorted(self.calls, key=lambda k: [order["/".join(k.split("/")[:j])] for j in range(1, k.count("/") + 2)])
        print(f"{'phase (inclusive; indent = nested in the row above)':58} calls/step gpu_ms cpu_ms self_cpu_ms")
        for k in keys:
            kids = sum(v for c, v in cpu.items() if c.rsplit("/", 1)[0] == k and c.count("/") == k.count("/") + 1)
            name = "  " * k.count("/") + k.rsplit("/", 1)[-1]
            print(f"{name:58} {len(self.calls[k]) / steps:10.2f} {gpu[k]:6.2f} {cpu[k]:6.2f} {cpu[k] - kids:11.2f}")
        return cpu, gpu


def timed(step_fn, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step_fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def step_fns(env, policy):
    b, n = env.env.backend, env.num_envs
    gen = torch.Generator(env.device).manual_seed(args.seed)
    acts = torch.rand(16, n, env.single_action_space.shape[0], generator=gen, device=env.device) * 2 - 1
    target = acts[0, :, : b.num_dof] * 0.3
    i = iter(range(sys.maxsize))
    return dict(
        physics=lambda: [b.simulate() for _ in range(b.control_freq_inv)],
        backend=lambda: b.step(target),
        env=lambda: env.step(acts[next(i) % 16]),
        env_np=lambda: env.step(env.action_space.sample()),
        rollout=torch.no_grad()(lambda: (env.send(policy(env.observations)[0]), env.recv())),
    )


def profile_phases(env, fn, row, steps=args.steps):
    timer = PhaseTimer(env)
    timed(fn, steps)
    cpu, gpu = timer.report(steps)
    pct = 100 * sum(v for k, v in cpu.items() if "/" not in k) / row["ms_per_step"]
    print(f"top-level phases sum to {pct:.1f}% of the unprofiled step's {row['ms_per_step']:.2f} ms (CPU wall)")
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        timed(fn, steps)
    torch.cuda.set_sync_debug_mode(0)
    timer.unwrap()
    caught = [w for w in caught if "synchroniz" in str(w.message)]
    sites = collections.Counter(f"{os.path.relpath(w.filename)}:{w.lineno}" for w in caught)
    syncs, graph_ms = sum(sites.values()) / steps, sum(v for k, v in gpu.items() if k.endswith("backend.simulate"))
    print(f"syncs/step {syncs:.2f}; top sites: {sites.most_common(8)}")
    return dict(syncs_per_step=syncs, phase_sum_pct=pct, graph_ms=graph_ms)  # graph_ms: event span of the graphs


def profile_trace(env, fn, row):
    tag, step_ms = f"{row['mode']}_{row['preset']}_{row['num_envs']}", row["ms_per_step"]
    timer = PhaseTimer(env)  # phases show up as ranges in the trace
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        timed(fn, 10)
    timer.unwrap()
    prof.export_chrome_trace(f"tmp/speed/trace_{tag}.json")
    phases = {k.split("/")[-1] for k in timer.calls}  # their GPU-side ranges are not kernels
    kernels = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    kernels = sorted((e for e in kernels if e.key not in phases), key=lambda e: -e.self_device_time_total)
    busy = sum(e.self_device_time_total for e in kernels) / 1e4  # us over 10 steps -> ms/step
    # GPU-bound iff kernels outside the simulate() graphs + the graphs' event span (gaps inside a graph count as busy)
    # cover > 80% of the step. ponytail: warp kernels count as in-graph; eager eval_fk in reset_envs is missed.
    eager = sum(e.self_device_time_total for e in kernels if not e.key.endswith("_cuda_kernel_forward")) / 1e4
    verdict = "GPU-bound" if eager + row["graph_ms"] > 0.8 * step_ms else "launch/CPU-bound"
    print(f"{tag}: kernels {busy:.2f}, eager {eager:.2f} + graph {row['graph_ms']:.2f} of {step_ms:.2f} ms: {verdict}")
    for e in kernels[:12]:
        print(f"  {e.self_device_time_total / 1e4:7.3f} ms/step {e.count / 10:6.1f}/step  {e.key[:90]}")
    return dict(gpu_busy_ms=busy, verdict=verdict)


def record_run(env, step):
    b, snaps = env.env.backend, []
    for i in range(200):
        step()
        if i % 10 == 9:
            fields = env.observations, env.rewards, env.terminals | env.truncations, b.root_states, b.dof_pos
            snaps.append([t.cpu().numpy() for t in fields])
    return dict(zip(("obs", "rew", "done", "root_states", "dof_pos"), map(np.stack, zip(*snaps))))


def main():
    presets = ["-"] if args.backend == "isaacgym" else args.solver_preset.split(",")
    configs = [(p, int(n)) for p in presets for n in args.num_envs.split(",")]
    if len(configs) > 1:  # one process per config: backends are never freed (atexit hook), profiling taints a process
        if args.record or args.check:
            sys.exit("--record/--check take a single (num-envs, solver-preset)")
        for p, n in configs:
            subprocess.run([sys.executable, *sys.argv, "--num-envs", str(n), "--solver-preset", p], check=True)
        return
    ((preset, n),) = configs
    if args.fixed_reset_times:  # a golden-ratio phase per motion id instead of torch.rand

        def fixed_time(self, ids, mask=None):
            return ((ids * 0.618034) % 1 * self._motion_lib._motion_lengths[ids] * 30).floor() / 30

        HumanoidPHC._sample_time = fixed_time
    sha = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for seed in (random.seed, np.random.seed, torch.manual_seed):
        seed(args.seed)
    device = "cpu" if args.backend == "mujoco_cpu" else "cuda"
    kwargs = dict(motion_file=args.motion_file, has_self_collision=True, num_envs=n, device_type=device)
    kwargs.update(backend=args.backend, headless=True, solver_preset=preset)  # isaacgym ignores the preset
    free0 = torch.cuda.mem_get_info()[0]
    env = pufferlib.vector.make(env_creator(), env_kwargs=kwargs)
    env.async_reset(args.seed)
    env.action_space.seed(args.seed)
    policy = pufferlib.cleanrl.Policy(PHCPolicy(env, hidden_size=512)).to(env.device)
    if args.checkpoint:
        policy.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)["state_dict"])
    fns = step_fns(env, policy)
    meta = dict(backend=args.backend, preset=preset, num_envs=n, seed=args.seed)
    if args.record or args.check:
        env.clip_actions = False
        snaps = record_run(env, fns["env"])
        if args.record:
            np.savez(args.record, meta=json.dumps(meta), **snaps)
        if args.check:
            ref = np.load(args.check)
            if json.loads(str(ref["meta"])) != meta:
                sys.exit(f"{args.check} was recorded with {ref['meta']}")
            diffs = {k: np.abs(snaps[k].astype(float) - ref[k]).reshape(20, -1).max(1) for k in snaps}
            for k, d in diffs.items():
                first = (10 * np.flatnonzero(d > args.tol)[:1] + 10).tolist() or "-"
                print(f"{k}: step 10 {d[0]:.2g}, max {d.max():.2g}, first step > tol {first}")
            print("PASS") if max(d[0] for d in diffs.values()) <= args.tol else sys.exit("FAIL")
        return
    rows = []
    for mode in args.mode.split(","):
        env.clip_actions = mode != "env"
        timed(fns[mode], args.warmup)
        ep0 = env.episode_count
        sps = sorted(args.steps * n / timed(fns[mode], args.steps) for _ in range(args.repeats))
        rows.append(dict(mode=mode, sps_median=sps[len(sps) // 2], sps_min=sps[0], sps_max=sps[-1],
                         ms_per_step=1e3 * n / sps[len(sps) // 2], gpu_mem_gb=(free0 - torch.cuda.mem_get_info()[0]) / 1e9,
                         resets_per_step=(env.episode_count - ep0) / (args.steps * args.repeats),
                         steps=args.steps, repeats=args.repeats, checkpoint=args.checkpoint, git=sha, **meta))  # fmt: skip
        print(f"{mode:8} {args.backend} {preset} N={n}: sps {sps[len(sps) // 2]:.0f} [{sps[0]:.0f}, {sps[-1]:.0f}] "
              f"{rows[-1]['ms_per_step']:.2f} ms/step resets/step {rows[-1]['resets_per_step']:.1f}")  # fmt: skip
    for prof in (profile_phases, profile_trace) if args.profile else ():  # torch.profiler last: it slows the process
        for row in rows:
            env.clip_actions = row["mode"] != "env"
            row.update(prof(env, fns[row["mode"]], row))
    with open(args.out, "a") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


if __name__ == "__main__":
    main()
