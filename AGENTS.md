# puffer_phc

PHC (Perpetual Humanoid Control) motion imitation trained with PufferLib on Isaac Gym. Python 3.8 only (Isaac Gym constraint).

## Where it runs

Run inside the devcontainer (`modest_ritchie`), where this repo is `/workspace/puffer_phc` (host: `/workspace/docker/puffer_phc`). The pixi env at `.pixi/` has absolute container paths baked in, so do not use it from the host.

From the host, run commands with:

```bash
docker exec -w /workspace/puffer_phc -e PYTHONPATH=/workspace/puffer_phc modest_ritchie bash -lc '~/.pixi/bin/pixi run <cmd>'
```

## Environment

- pixi lives in the container home (`~/.pixi/bin/pixi`) and is lost on container rebuild. Reinstall: `curl -fsSL https://pixi.sh/install.sh | bash`. The env itself (`.pixi/`) is on the bind mount and survives.
- `pixi.lock` is gitignored. It was freshly resolved; the upstream lock had a stale `cuda-version-12.9-3` hash.
- Stack: pytorch 2.3.1 + CUDA 12.1, Isaac Gym 1.0rc4 (`pip install -e ./isaacgym/python`), PufferLib fork `kywch/PufferLib@isaacgym`, smpl_sim.
- `pixi run build_gymtorch` must have been run. Otherwise the repo's `gymtorch/` dir shadows Isaac Gym's and training fails with `module 'gymtorch' has no attribute 'wrap_tensor'`.
- `PYTHONPATH` must be the repo root; the package is not installed.
- Check: `pixi run test_deps` prints `True`.

Rebuild from scratch (inside the container):

```bash
pixi install
pixi run pip install -e ./isaacgym/python
pixi run build_gymtorch
pixi run test_deps
```

## Local files (not in git, excluded via `.git/info/exclude`)

| Path | What |
|---|---|
| `data/amass_train_take6_upright.pkl` | Small AMASS subset (129 motions, 32 MB), copied from `/windows/dataset/AMASS` |
| `ckpt_amass_800m_acc97.pt` | Pretrained AMASS policy; load with `-c` |
| `smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` | SMPL body models |
| `isaacgym/` | Isaac Gym 1.0rc4 |

The devcontainer cannot see `/windows`. Copy any other datasets into `data/`.

## Commands

From `/workspace/puffer_phc` inside the container:

```bash
export PATH=~/.pixi/bin:$PATH PYTHONPATH=$PWD

# Train (config.ini defaults: 4096 envs, 500M steps; checkpoints and full eval every 1500 epochs under experiments/<exp_id>/)
pixi run python scripts/train.py -m data/amass_train_take6_upright.pkl

# Fine-tune from the pretrained policy
pixi run python scripts/train.py -m data/amass_train_take6_upright.pkl -c ckpt_amass_800m_acc97.pt

# Smoke test (~1 min)
pixi run python scripts/train.py -m data/amass_train_take6_upright.pkl \
  --env.num-envs 512 --train.batch-size 16384 --train.minibatch-size 4096 --train.total-timesteps 200000

# Eval (headless; writes eval_summary_MMDD_HHMM.json and results_by_motion_MMDD_HHMM.tsv next to the checkpoint)
pixi run python scripts/train.py --mode eval -c ckpt_amass_800m_acc97.pt -m data/amass_train_take6_upright.pkl
```

Any `[section] key` in `config.ini` can be overridden as `--section.key-name`. Add `--track` to log locally to Aim under `experiments/.aim`; view with `pixi run aim up --repo experiments --host 127.0.0.1 --port 43800`. Merge runs from another machine: `pixi run aim runs --repo <other>/experiments cp --destination experiments <run_hash>`. `--mode sweep` requires `--tag <name>` (every trial is tracked and tagged); filter in the Aim UI/search with `"<name>" in run.tags`.

`--mode play` opens an Isaac Gym viewer and needs X11; it has not been set up here.

The training dashboard is a rich live table. When stdout is redirected, the log shows only the frames that were printed, and the startup motion-key dump can look like a hang. Check GPU use or `experiments/` before assuming it is stuck.

## Reference results

- Smoke test: about 10.3k steps per second on an RTX 3090 with 512 envs.
- Eval of `ckpt_amass_800m_acc97.pt` on `amass_train_take6_upright.pkl`: success rate 0.992, MPJPE 37.1 mm, PA-MPJPE 20.1 mm.

## Known issues

- **Shared memory:** the devcontainer has the Docker default of 64 MB `/dev/shm`. With multiple workers, motion loading (`puffer_phc/motion_lib.py`) passes tensors through shared memory; a worker dies with `No space left on device` and the main process hangs forever in `queue.get()`.
  - Current workaround: `num_thread=1` in `puffer_phc/humanoid_phc.py` (local, uncommitted change).
  - Permanent fix: `--shm-size=16g` was added to `/workspace/docker/.devcontainer/devcontainer.json`, but it takes effect only after a devcontainer rebuild. After the rebuild, `num_thread` can go back to 4.
- **No debugger in the container:** ptrace is blocked, so `py-spy dump` fails. Inspect hung processes through `/proc/<pid>/wchan` and the log instead.

## Converting more AMASS data

`scripts/convert_amass_data.py --path <AMASS root>` reads `**/*.npz` and writes `amass_train_<N>_upright.pkl` into the input directory. Run it where the data is reachable (host paths need a host env, or copy the raw AMASS data into the container first).

- Optional: `--occulusion_file amass_copycat_occlusion_v3.pkl` filters the set to about 11.3k motions. The file is not downloaded.
- A larger pre-converted set exists at `/windows/dataset/AMASS/amass_train_hard_1k.pkl` (628 MB).
