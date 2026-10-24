# ruff: noqa: E402  # isaacgym's backend-peek must run before any heavier (torch-importing) import below
"""Measure env-steps/sec for a sim backend, to compare isaacgym vs mujoco_gpu vs mujoco_cpu.

    pixi run -e default python scripts/bench_sps.py --backend isaacgym
    pixi run -e newton python scripts/bench_sps.py --backend mujoco_gpu --num-envs 4096
"""
import argparse
import time

from puffer_phc.backend import preload_isaacgym


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["isaacgym", "mujoco_gpu", "mujoco_cpu"], default="mujoco_gpu")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--solver-preset", default=None, help="mujoco backends only")
    p.add_argument("--motion-file", default="data/amass_train_take6_upright.pkl")
    return p.parse_args()


args = parse_args()
preload_isaacgym(args.backend)

import torch

from puffer_phc.environment import make


def main():
    kwargs = dict(
        motion_file=args.motion_file,
        has_self_collision=True,
        num_envs=args.num_envs,
        device_type="cpu" if args.backend == "mujoco_cpu" else "cuda",
        headless=True,
        backend=args.backend,
    )
    if args.solver_preset and args.backend != "isaacgym":
        kwargs["solver_preset"] = args.solver_preset
    env = make("puffer_phc", **kwargs)
    env.reset()

    backend = env.env.backend
    is_cuda = env.device.startswith("cuda")
    control_steps = backend.control_freq_inv * backend.preset["substeps"] if hasattr(backend, "preset") else None

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    def timed_loop(step_fn, n):
        sync()
        start = time.perf_counter()
        for _ in range(n):
            step_fn()
        sync()
        return time.perf_counter() - start

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    timed_loop(lambda: env.step(env.action_space.sample()), args.warmup)
    env_elapsed = timed_loop(lambda: env.step(env.action_space.sample()), args.steps)
    env_sps = args.steps * env.num_envs / env_elapsed

    target = (torch.rand(env.num_envs, backend.num_dof, device=env.device) * 2 - 1) * 0.3
    timed_loop(lambda: backend.step(target), args.warmup)
    phys_elapsed = timed_loop(lambda: backend.step(target), args.steps)
    phys_sps = args.steps * env.num_envs / phys_elapsed

    mem_peak = f"{torch.cuda.max_memory_allocated(env.device) / 1e9:.2f}GB" if is_cuda else "-"
    print(
        f"backend={args.backend} preset={args.solver_preset or '-'} num_envs={env.num_envs} "
        f"physics_steps_per_control={control_steps or '-'} env_sps={env_sps:.0f} "
        f"physics_sps={phys_sps:.0f} gpu_mem_peak={mem_peak}"
    )


if __name__ == "__main__":
    main()
