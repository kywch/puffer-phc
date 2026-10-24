# mujoco_gpu speed: what changed, how to measure, what's left

Scope: training-time throughput of the `mujoco_gpu` backend (Newton `SolverMuJoCo` on mujoco-warp), measured with
`scripts/bench_sps.py` on one RTX 3090, GPU otherwise idle. Not about eval fidelity — see
[physics_solver.md](physics_solver.md) and [newton_mujoco_physics.md](newton_mujoco_physics.md) for that. `tmp/` is
local scratch (gitignored); paths under it are not in the repo and may be pruned.

## 1. Summary

SPS = env-steps/sec = num_envs x control-steps/sec, 4096 envs unless noted. "physics" = backend `simulate()` only;
"env" = full `PHCPufferEnv.step`; "rollout" = policy forward + vecenv recv/send with `ckpt_amass_800m_acc97.pt`.

| preset | metric | start (3.15) | 383ceff | 6151f02 | cb8ad4a | now |
|---|---|---|---|---|---|---|
| newton_120 | physics | 105801 | 106926 | 141367 | - | 141367⁶ |
| newton_120 | env | 75653 | 76474 | 93710 | 106360 | 106360 |
| newton_120 | rollout (ckpt) | n/a¹ | 64963² | 76804³ | 86608 | 86608 |
| newton_180 | physics | 74933 | 75676 | 101333 | n/m⁴ | 101333 |
| newton_180 | env | 55775 | 56220 | 70677 | n/m⁴ | 70677 |
| isaacgym_parity_240 | physics | 55780 | 56291 | 75038 | n/m⁴ | 75038 |
| isaacgym_parity_240 | env | 44849 | 45176 | 57662 | n/m⁴ | 57662 |
| isaacgym (PhysX), ref⁵ | env / physics | 38754 / 100918 | - | - | - | - |

Sources: start = `tmp/speed/old_315/baseline.md`; 383ceff, 6151f02, cb8ad4a = their commit messages / `tmp/speed/baseline.md`.
¹ `old_315/baseline.md`'s rollout(ckpt) figure (61019) is flagged profiler-contaminated by `tmp/speed/baseline.md`
("its 57.6 ms 'unprofiled' and 61k ckpt-rollout figures were profiler-contaminated"); no clean unprofiled ckpt-rollout
number exists at that point. ² `tmp/speed/profile.jsonl`'s checkpoint rollout row, measured *under* `--profile`
(profiling adds overhead), not directly comparable to the unprofiled 76804/86608 cells; no unprofiled ckpt-rollout
figure was recorded at 383ceff. ³ `tmp/speed/b1/bench.jsonl`, tag `6151f02` (working tree right before cb8ad4a).
⁴ not re-benched after cb8ad4a — the host-sync fixes are in the env/torch layer, not the solver, so similar relative
gains are expected but unverified. ⁵ `tmp/speed/old_315/sps_4096.txt`, pre-harness `8616ce2` script,
`other_gpu_procs=1` (GPU not idle) — order-of-magnitude reference only, not comparable. ⁶ an independent
re-measurement right after cb8ad4a saw 139754 (`tmp/speed/b1/bench.jsonl`, tag `6151f02-dirty`); cb8ad4a didn't touch
the solver, so the ~1% gap to 141367 is run-to-run noise, not a regression.

## 2. How to measure

`scripts/bench_sps.py` (its docstring is the full reference). Modes:

| mode | what it runs |
|---|---|
| `physics` | `backend.simulate()` x `control_freq_inv` |
| `backend` | `backend.step(targets)`: PD-target conversion + physics + refresh |
| `env` | `PHCPufferEnv.step` on pre-generated seeded GPU action tensors (always random actions) |
| `env_np` | same, but `action_space.sample()` (numpy actions, H2D copy per step, as training used to send them) |
| `rollout` | policy forward + vecenv recv/send, no PPO update (random-init policy unless `--checkpoint` is passed) |

```
pixi run python scripts/bench_sps.py --mode physics,env --num-envs 1024,4096 \
  --solver-preset newton_120,newton_180 --profile
```

Add `--checkpoint ckpt_amass_800m_acc97.pt` for checkpoint actions in `rollout` mode (the SPS numbers in this doc
labeled "ckpt" used it; plain "rollout" numbers are the random-init policy). `PYTHONPATH=$PWD` is only needed if not
already exported in the shell (see `AGENTS.md`).

Comma lists sweep every (mode, N, preset) combo, one subprocess per (N, preset) (backends are never freed).
`--profile`, after all timing rows: per-phase CUDA-event/CPU ms by call path, host sync counts by call site
(`torch.cuda.set_sync_debug_mode`), a 10-step `torch.profiler` trace, and a GPU- vs launch-bound verdict.
`--record PATH` / `--check PATH` take a single (num-envs, solver-preset) config — a comma-swept invocation refuses to
run them. They save/compare a seeded 200-step trace (obs/reward/done/root_states/dof_pos every 10 steps), and a
`--check` whose recorded config doesn't match the current one is refused. `mujoco_gpu` is **not bitwise
deterministic** run to run (fp32 noise grows to ~0.1 by step 200), so `--check` only requires each field's max diff
at step 10 within `--tol` (default 1e-2).

```
pixi run nsys profile --cuda-graph-trace=node -t cuda,nvtx,osrt -o tmp/speed/nsys_newton_120 \
  python scripts/bench_sps.py --mode env --num-envs 4096 --solver-preset newton_120 --repeats 1 --profile
nsys stats --report cuda_gpu_kern_sum tmp/speed/nsys_newton_120.nsys-rep
```

Per-kernel breakdown inside the CUDA graph (phases are NVTX ranges; `torch.profiler` is empty under nsys).

Rules used for every change below: GPU otherwise idle, one config per process; median of 3 repeats (300 steps, 50
warmup, seed 0); keep a change only if it gives >=3% SPS at the measured config (the rule that dropped the
motion-lib dedup in section 4). Every kept change passes `bench_sps.py --check` against a reference trace and a
129-motion eval (`ckpt_amass_800m_acc97.pt`); the full test suite was additionally run for `cb8ad4a`
(`tmp/speed/b1/test_default.log`, `test_newton.log`) and is part of `383ceff`'s own verification (newton 21 passed,
default 4 passed 1 skipped, per its commit message) — not separately cited here for `6151f02`, whose commit message
says "Tests: none added."

## 3. What was done

### 3.1 Measurement harness (`8616ce2`, `bf3437a`)

**Problem.** No way to measure env/backend throughput beyond ad hoc scripts.
**Change.** `8616ce2` added a minimal SPS script; `bf3437a` extended it into the harness in section 2.
**Result.** Everything else here is measured with it. `old_315/baseline.md` is the first full baseline
(mujoco-warp 3.15); `tmp/speed/baseline.md` repeats it on 3.16.
**Caveat.** Profiling mutates the process (kernel load timings, extra wrappers), so `--profile` rows run after all
plain timing rows, and the sync counting is the harness's own (`torch.cuda.set_sync_debug_mode`), not warp's.

### 3.2 Newton pin for mujoco-warp 3.16 (`383ceff`)

**Problem.** mujoco-warp < 3.16 didn't wrap ball-joint actuator setpoints like MuJoCo C: near +-pi, the PD error
wraps the long way, driving the joint ~2*pi the wrong way. `mujoco_gpu.py` carried a per-step `_wrap_ball_ctrl`
workaround kernel for it.
**Change.** Pinned `newton` to main commit `7edb1bb3` (newton 1.8.0.dev0, mujoco/mujoco-warp 3.16.0, native wrap)
and removed the workaround kernel and its hook.
**Result.** `test_gpu_matches_cpu` (crosses pi on a ball joint) passes unaided. 129-motion GPU eval 0.992 success /
38.5 mm MPJPE, `running_on_spot` passes. SPS within ~1% of 3.15 everywhere — not a speed change itself, but what
made it safe to keep optimizing on current mujoco-warp.
**Risk.** Pinned to a main-branch commit, not a release; re-pin to 1.8.0 when it ships.

### 3.3 Dense tile Cholesky for the mass matrix (`6151f02`)

**Problem.** mujoco-warp tile-Cholesky-factors kinematic-tree blocks of M only up to `M_BLOCK_DENSE_MAX=64` dofs;
this model's tree is nv=75, so it fell back to sparse LDL — 58 serial launches/substep, 15.6 of 38 ms per control
step at 4096 envs (`old_315/baseline.md` in-graph breakdown).
**Change**, in `mujoco_gpu.py`: `mujoco_warp._src.types.M_BLOCK_DENSE_MAX = 80` (process-global constant, read in
`put_model()`); `solver.mjw_model.block_dim.cholesky_factorize = cholesky_factorize_solve = 128`; `NJMAX` 800 -> 256
(`NCONMAX` unchanged at 200). At N=4096 (`probe_physics/run_4096.log`): sparse baseline ~107k SPS; `dense80` alone
(default block_dim=32) drops to 85.2k (-21%); `dense80+bd128` reaches 134.9k (+26%) — the block_dim bump turns the
dense path from a regression into a win, not just a further gain.
**Result**, 4096 envs, physics / env SPS: newton_120 107k->141k / 76.5k->93.7k; newton_180 75.7k->101k / 56.2k->70.7k;
isaacgym_parity_240 56.3k->75.0k / 45.2k->57.7k. `bench_sps.py --check` PASS; eval 0.992 / 39.4 mm, overflow NONE;
compute-sanitizer memcheck 0 errors.
**Risks/caveats.**
- `M_BLOCK_DENSE_MAX` / `block_dim` are private mujoco-warp internals (`BlockDim` is experimental upstream) —
  re-check both after every mujoco-warp bump.
- `njmax=256` is sized off an observed peak of 108 constraint rows at `njmax=800` (`tmp/speed/baseline_run.log`) and
  100 rows once capped to 256 (`tmp/speed/b2/base_bench.log`, `after_bench.log`) — ~2.4x headroom over the observed
  peak, not a hard bound. `solver_stats()` samples only the last substep of each `simulate()`, so the true peak may
  be higher. (The `6151f02` commit message and an in-code comment cite a measured peak of 84 rows; that number is
  being corrected separately.) An overflow is recorded in `solver.mjw_data.overflow` but only surfaced when
  `solver_stats()` runs, which happens at process exit here (backend is never freed mid-run) — a mid-run overflow is
  silent until then.
- `jacobian="dense"` errors outright for nv>60 (`probe_physics/screen_2048.log`:
  `ValueError('Dense is unsupported for nv > 60 (nv = 75)')`); this change uses the dense tile-Cholesky
  *factorization* path, not `jacobian="dense"`.

### 3.4 Remove host syncs from the env step (`cb8ad4a`)

**Problem.** `--profile` sync counting showed 78 host syncs per env step, 85 per rollout step
(`tmp/speed/b1/base_profile.log`), from TorchScript scalar writes, boolean-mask indexing in `slerp`, and `Normal`'s
validating constructor — each forces a CPU-GPU round trip.
**Change** (`environment.py`, `humanoid_phc.py`, `torch_utils.py`, `policy.py`, `clean_pufferl.py`, `train.py`):
`slerp` uses `torch.where` instead of a boolean-mask write; JIT scalar writes (`x[..., -1] = 1`) use `.fill_()`;
per-env zeroing uses `index_fill_` (its index tensor must be int64, on the target device); the motion-lib cache and
its sync-y `(...).abs().sum() > 0` hit check are dropped (costs one extra motion query/step until lookups are
merged); `env.step` takes GPU tensors directly and builds terminals/truncations/episode-stat accumulation with
tensor ops, read only at log time; the training loop sends GPU actions without an explicit sync/numpy round trip;
`Normal(validate_args=False)` skips a syncing constructor check.
**Result**, 4096 envs, newton_120, SPS: env 93.7k->106.4k, env_np 82.2k->93.4k, rollout(ckpt) 76.8k->86.6k.
`bench --check` PASS; eval 0.992 / 38.4 mm.
**Risks/caveats.**
- Syncs/step: env 78->1 (reset-id `nonzero()` remains — data-dependent), rollout 85->3 (`torch.normal`'s std check
  remains).
- `index_fill_` needs int64 device ids; a caller passing int32 or CPU ids will error or silently sync.
- `Normal(validate_args=False)`: a NaN `mu`/`std` no longer raises at construction, surfaces later instead.
- Dropped motion-lib cache costs one extra `get_motion_state`/step; kept because the net effect was still a large
  win (see section 5 for revisiting this).

## 4. Tried and not taken

| idea | result | why not |
|---|---|---|
| `jacobian="dense"` | errors | unsupported for nv>60 (nv=75); `probe_physics/screen_2048.log` |
| `M_BLOCK_DENSE_MAX=80` alone, `block_dim=32` default | -26% (102020->75504 @ N=2048) | block_dim 32 is slower than sparse LDL at this size; needed the block_dim bump too |
| `opt.iterations` capped 3-6 | SPS flat; drift at 2s: 683 mm median @ it3 (455 it4, 189 it5, 84 it6) vs 68 mm run-to-run noise floor; it3/it4 also flag `OverflowType.ITERATIONS` | solver already exits at ~1.4 mean iters; cap only bites worst-case worlds and changes physics |
| `opt.tolerance` loosened to 1e-5/1e-4 (base 1e-6) | no SPS change | not the bottleneck |
| `ls_iterations` 5/10/20 | no SPS change | same |
| `solver="cg"` | -63% (102020->37450 @ N=2048) | needs far more iterations (mean 7.9 vs 1.4) to converge |
| `graph_conditional=False` | -54% (102020->46497 @ N=2048) | loses the conditional graph's early exit |
| SAP broadphase (`bp_tile`, `bp_seg`) vs NxN | flat to -16% | no win at this contact count; `bp_seg` notably slower |
| Smaller `nconmax` (48/64, with nj128/nj200) | flat SPS, `OverflowType.NARROWPHASE` flagged | drops real contacts |
| `njmax=128` alone (vs 256) | +2% SPS @ N=4096 (142.4k vs 139.3k dense; 111.1k vs 107k sparse), no overflow | 256 kept for headroom over the observed 100-108 row peak (section 3.3) |
| Motion-lib query dedup (cache next step's task-obs state for reward) | +1.5% env / +1.0% rollout @ 4096 (`tmp/speed/b2/bench.jsonl`) | below the 3% bar; patch kept at `tmp/speed/b2/b2_motion_memo.patch` |

## 5. Where time goes now, and next candidates

In-graph kernel time, `physics` mode, newton_120 @ 4096, from `nsys stats --report cuda_gpu_kern_sum`
(`tmp/speed/probe_physics/nsys_winner_cuda_gpu_kern_sum.csv`, ~350 control steps in the trace). This run was taken at
`383ceff` with the `6151f02` solver settings applied on top via the `probe_physics/check.py` scratch harness (variant
`dense80+bd128+nj256c`); `cb8ad4a` didn't touch the solver, so this breakdown should still hold, but `physics` mode
has no env-side torch kernels in it (section 5's "outside physics" number below is separate):

| kernel | ms/control-step (est. from nsys totals) |
|---|---|
| `_tile_cholesky_factorize_solve_block` (dense Cholesky of M) | ~9.8 |
| `_update_gradient_cholesky_blocked` + `..._skip_unchanged` (constraint-solver Hessian factor) | ~3.9 |
| `_primitive_narrowphase` | 1.52 |
| `_efc_contact_jac_sparse` | 0.83 |
| `_update_gradient_init_h_sparse` | 0.81 |
| `_M` (mass matrix assembly) | 1.06 |
| total in-graph | 29.3 |

From `tmp/speed/b1/bench.jsonl` (post-cb8ad4a): physics 139754 SPS (29.31 ms/step) vs env 106360 SPS (38.51 ms/step)
at N=4096 -> about **9.2 ms outside physics** (env - physics; includes PD-target conversion + refresh plus
obs/reward/reset bookkeeping; random actions, ~half the envs reset each step).

Next candidates, roughly ranked, all estimates pending measurement:

1. **Dense Cholesky is now the largest single kernel (~9.8 ms).** Further gains need a faster mujoco-warp
   factorization path or exploiting tree sparsity more than today's block-dense trade; not attempted here.
2. **Fuse the ~9.2 ms of post-physics env work into the CUDA graph**, or `torch.compile` obs+reward+reset, so launch
   overhead (not GPU compute) stops dominating that part of the step — extending the pattern from 3.4.
3. **Fixed-shape resets.** Reset recomputes obs for the reset subset separately from the main pass; a fixed-shape
   (always-full-batch) reset would let it share the fused/compiled path in (2) instead of branching.
4. **Merge motion-lib queries.** `get_motion_state` runs multiple times/step (reward, task-obs, reset); the dedup
   patch in section 4 was correct but under the 3% bar alone — revisit combined with (2) or (3).
5. **More envs.** Physics plateaus by N=2048 in `old_315/baseline.md`, but that's 3.15, pre-dense-Cholesky,
   pre-host-sync-removal data — re-measure before relying on it. If the plateau still holds and the overhead in
   item 2 is roughly fixed in N, env SPS should keep rising with N until GPU memory or launch count becomes the
   limit — not verified past N=8192 either way.
