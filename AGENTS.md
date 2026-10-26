# puffer_phc

PHC (Perpetual Humanoid Control) motion imitation trained with PufferLib, by default on the MuJoCo GPU backend (Newton + mujoco-warp). The MuJoCo CPU and GPU backends run in the `default` pixi env (Python 3.12, newton main-branch pin / mujoco-warp 3.16, torch 2.12+cu130); the legacy Isaac Gym backend runs in the `isaacgym` pixi env (Python 3.8).

## Where it runs

Run inside the devcontainer (`modest_ritchie`), where this repo is `/workspace/puffer_phc` (host: `/workspace/docker/puffer_phc`). The pixi env at `.pixi/` has absolute container paths baked in, so do not use it from the host.

From the host, run commands with:

```bash
docker exec -w /workspace/puffer_phc -e PYTHONPATH=/workspace/puffer_phc modest_ritchie bash -lc '~/.pixi/bin/pixi run <cmd>'
```

## Environment

- pixi lives in the container home (`~/.pixi/bin/pixi`) and is lost on container rebuild. Reinstall: `curl -fsSL https://pixi.sh/install.sh | bash`. The env itself (`.pixi/`) is on the bind mount and survives.
- `pixi.lock` is gitignored. It was freshly resolved; the upstream lock had a stale `cuda-version-12.9-3` hash.
- `isaacgym` env stack: pytorch 2.3.1 + CUDA 12.1, Isaac Gym 1.0rc4 (`pip install -e ./isaacgym/python`), PufferLib fork `kywch/PufferLib@isaacgym`, smpl_sim.
- `pixi run -e isaacgym build_gymtorch` must have been run; it builds `third_party/gymtorch` (develop mode), which the isaacgym backend imports as `gymtorch`.
- `PYTHONPATH` must be the repo root; the package is not installed. pixi activation sets it.
- Check: `pixi run -e isaacgym test_deps` prints `True`.

Rebuild from scratch (inside the container):

```bash
pixi install --all
pixi run -e isaacgym pip install -e ./isaacgym/python
pixi run -e isaacgym build_gymtorch
pixi run -e isaacgym test_deps
```

## Local files (not in git, ignored via `.gitignore`)

| Path | What |
|---|---|
| `data/amass_train_take6_upright.pkl` | Small AMASS subset (129 motions, 32 MB), copied from `/windows/dataset/AMASS` |
| `ckpt_amass_800m_acc97.pt` | Pretrained AMASS policy; load with `-c` |
| `smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` | SMPL body models |
| `isaacgym/` | Isaac Gym 1.0rc4 |

The devcontainer cannot see `/windows`. Copy any other datasets into `data/`.

Processed motion libraries are cached in `<motion_file>.motion_cache/` (gitignored). The cache rebuilds automatically when the motion file or the processing code changes; delete the directory to force a rebuild. Details in `docs/optimization.md` (section 3.5).

The `smpl/SMPL_*.pkl` files were re-saved without chumpy (numpy `shapedirs`) so they load on Python 3.12. On other machines, run `uv run smpl/strip_chumpy.py` first (see `smpl/README.md`).

## Commands

From `/workspace/puffer_phc` inside the container:

```bash
export PATH=~/.pixi/bin:$PATH PYTHONPATH=$PWD

# Train on mujoco_gpu with newton_120 (config.ini defaults: 4096 envs, 800M steps; checkpoints every 2100 epochs, full eval + motion resample every 500 under experiments/<exp_id>/)
pixi run python scripts/train.py -m data/amass_train_take6_upright.pkl

# Smoke test (~1 min)
pixi run python scripts/train.py -m data/amass_train_take6_upright.pkl \
  --env.num-envs 512 --train.batch-size 16384 --train.minibatch-size 4096 --train.total-timesteps 200000

# Legacy Isaac Gym: train, fine-tune from the pretrained policy, smoke test
pixi run -e isaacgym python scripts/train.py --env.backend isaacgym -m data/amass_train_take6_upright.pkl
pixi run -e isaacgym python scripts/train.py --env.backend isaacgym -m data/amass_train_take6_upright.pkl -c ckpt_amass_800m_acc97.pt
pixi run -e isaacgym python scripts/train.py --env.backend isaacgym -m data/amass_train_take6_upright.pkl \
  --env.num-envs 512 --train.batch-size 16384 --train.minibatch-size 4096 --train.total-timesteps 200000

# Eval on Isaac Gym (headless; writes eval_summary_MMDD_HHMM.json and results_by_motion_MMDD_HHMM.tsv next to the checkpoint)
pixi run -e isaacgym python scripts/train.py --env.backend isaacgym --mode eval -c ckpt_amass_800m_acc97.pt -m data/amass_train_take6_upright.pkl

# Eval on MuJoCo CPU (Newton SolverMuJoCo, one CPU solver per env; ~3.5 min for 129 motions with the
# isaacgym_parity_240 preset, ~6 min with isaacgym_parity_480)
pixi run python scripts/train.py --mode eval -c ckpt_amass_800m_acc97.pt -m data/amass_train_take6_upright.pkl --env.backend mujoco_cpu --env.solver-preset isaacgym_parity_240 --env.num-envs 129

# Same eval on the GPU (mujoco-warp, all envs batched in one solver; ~40 s)
pixi run python scripts/train.py --mode eval -c ckpt_amass_800m_acc97.pt -m data/amass_train_take6_upright.pkl --env.backend mujoco_gpu --env.solver-preset isaacgym_parity_240 --env.num-envs 129

# Tests (both envs define `test`, so -e is required)
pixi run -e default test
pixi run -e isaacgym test
```

Any `[section] key` in `config.ini` can be overridden as `--section.key-name`. `[<section>.<backend>]` overrides the `[env]` and `[train]` keys for that backend, e.g. `[env.isaacgym]` (`rew_power_coef = 0.0005`) and `[train.isaacgym]` (`optimizer = "adam"`, because the isaacgym env's torch 2.3.1 has no `torch.optim.Muon`; `[train] optimizer = "muon"` is the default). CLI flags still win. Add `--track` to log locally to Aim under `experiments/.aim`; view with `pixi run aim up --repo experiments --host 127.0.0.1 --port 43800`. Merge runs from another machine: `pixi run aim runs --repo <other>/experiments cp --destination experiments <run_hash>`. `--mode sweep` requires `--tag <name>` (every trial is tracked and tagged); filter in the Aim UI/search with `"<name>" in run.tags`.

`--mode play` opens an Isaac Gym viewer and needs X11; it has not been set up here. On the mujoco backends, `--mode play` instead opens a Newton ViewerGL window (see `puffer_phc/backend/_newton_mujoco.py`); it also needs a display.

The training dashboard is a rich live table (CPU is this process, top-style, 100 = one core; DRAM is the process RSS share; VRAM is device-wide used). When stdout is redirected, the log shows only the frames that were printed, and the startup motion-key dump can look like a hang. Check GPU use or `experiments/` before assuming it is stuck.

## Sim backends

- `[env] backend` in `config.ini` selects `mujoco_gpu` (default, with `solver_preset = newton_120`), `mujoco_cpu` or `isaacgym`.
- Code in `puffer_phc/backend/` (`base.py` docstring is the interface contract: tensor layouts and conventions).
- `mujoco_cpu` (MuJoCo C, one solver per env) and `mujoco_gpu` (mujoco-warp, one batched solver, CUDA graph per `simulate()`) share the model build, presets and gym conversions in `backend/_newton_mujoco.py`. Both convert gym's per-joint rotation-vector dof_pos and map PD targets through the measured PhysX drive frame (`puffer_phc/assets/smpl_humanoid_drive_frame.txt`).
- Eval of the provided Isaac Gym checkpoint uses the `[env] solver_preset` (`--env.solver-preset`) `isaacgym_parity_240` (8 x 1/240 s per control step, refsafe off), or `isaacgym_parity_480` (16 x 1/480 s, refsafe on, higher fidelity, ~2x the cost). Both use near-rigid contacts (solref 0.005) and per-body kp/kd gain scales (ankle/toe 0.2, hip/knee 0.8, torso/spine/chest 0.5) to emulate Isaac Gym's unconverged TGS solver. See `docs/physics_solver.md` for the settings and why.
- Eval of `ckpt_amass_800m_acc97.pt` on mujoco_cpu (`isaacgym_parity_240` preset): success 0.984, MPJPE 39.5 mm (vs gym's 0.992 / 37.4; remaining gap is physics, not a bug; eval has ±0.02 success run-to-run noise, see docs/physics_solver.md). On mujoco_gpu: 0.977 / 0.953, 39.4 / 40.8 mm (seeds 0/1; float32, solver tolerance 1e-6 vs 1e-8 on CPU; mujoco-warp >= 3.16 wraps ball-joint PD errors across pi natively, matching MuJoCo C), ~40 s wall vs ~3.5 min; raw ~55k env-steps/s at 4096 envs on an RTX 3090 (predates the env-step work in `docs/optimization.md`).
- Inspect solver state via `backend.solvers[i].mj_model` / `.mj_data` (mujoco_cpu) or `backend.solver.mjw_model` / `.mjw_data` (mujoco_gpu, batched).
- Newton-native training presets (no gym gain scales, contact tc 1/120): `newton_120` (4 x 1/120 s per control step, the config.ini default) and `newton_180` (6 x 1/180 s, higher fidelity, slower). See `docs/newton_mujoco_physics.md`.

## Reference results

- Smoke test (512 envs, RTX 3090): 10.3k steps per second on Isaac Gym.
- Full config (4096 envs, newton_120, RTX 3090, GPU clock unlocked): ~71.5k SPS with Adam; Muon (the default) is ~8% lower. See `docs/optimization.md`.
- Power penalty: mujoco's `dof_force` is the applied PD torque, ~5-9x Isaac Gym's. newton_120, 129 motions, eval at epoch 1500: `rew_power_coef` 0.0001 gives success 0.992 / MPJPE 43.5 mm, the gym-tuned 0.0005 gives 0.039. mujoco backends use 0.0001; `[env.isaacgym]` keeps 0.0005.
- Eval of `ckpt_amass_800m_acc97.pt` on `amass_train_take6_upright.pkl`: success rate 0.992, MPJPE 37.1 mm, PA-MPJPE 20.1 mm.
- Isaac Gym eval is not bit-deterministic run to run (MPJPE spread ~37.1–37.8 mm, runs: 37.09–37.81, success 0.984–0.992).

## Known issues

- **Shared memory:** the devcontainer has the Docker default of 64 MB `/dev/shm` (`--shm-size=16g` in `/workspace/docker/.devcontainer/devcontainer.json` applies after a rebuild). Motion loading is single-process; its multi-worker path, which hung on this limit, was removed.
- **No debugger in the container:** ptrace is blocked, so `py-spy dump` fails. Inspect hung processes through `/proc/<pid>/wchan` and the log instead.

## Converting more AMASS data

`scripts/convert_amass_data.py --path <AMASS root>` reads `**/*.npz` and writes `amass_train_<N>_upright.pkl` into the input directory. Run it where the data is reachable (host paths need a host env, or copy the raw AMASS data into the container first).

- Optional: `--occulusion_file amass_copycat_occlusion_v3.pkl` filters the set to about 11.3k motions. The file is not downloaded.
- A larger pre-converted set exists at `/windows/dataset/AMASS/amass_train_hard_1k.pkl` (628 MB).
