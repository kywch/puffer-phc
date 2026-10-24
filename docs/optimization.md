# Training speed: what changed, how to measure, what's left

Scope: training throughput, not just the env step — the `mujoco_gpu` backend (Newton `SolverMuJoCo` on mujoco-warp),
motion loading, and the PPO loop in `clean_pufferl.py` — measured with `scripts/bench_sps.py` and `scripts/train.py`
on one RTX 3090, GPU otherwise idle. Not about eval fidelity — see [physics_solver.md](physics_solver.md) and
[newton_mujoco_physics.md](newton_mujoco_physics.md) for that.

## 1. Summary

SPS = env-steps/sec = num_envs x control-steps/sec, 4096 envs unless noted. "physics" = backend `simulate()` only;
"env" = full `PHCPufferEnv.step`; "rollout" = policy forward + vecenv recv/send with a checkpoint policy;
"training" = full `scripts/train.py` loop (rollout + PPO update), `config.ini`'s full config unless noted.

| preset | metric | start (3.15) | 383ceff | 6151f02 | cb8ad4a | 05a3d68 | now⁷ |
|---|---|---|---|---|---|---|---|
| newton_120 | physics | 105801 | 106926 | 141367 | - | 141367⁶ | 142760 |
| newton_120 | env | 75653 | 76474 | 93710 | 106360 | 106360 | 129155 |
| newton_120 | rollout (ckpt) | n/a¹ | 64963² | 76804³ | 86608 | 86608 | 97332 |
| newton_120 | **training** | - | - | - | - | **36.6k→53.1–53.6k⁸** | **57900** |
| newton_180 | physics | 74933 | 75676 | 101333 | n/m⁴ | n/m⁴ | n/m⁴ |
| newton_180 | env | 55775 | 56220 | 70677 | n/m⁴ | n/m⁴ | n/m⁴ |
| isaacgym_parity_240 | physics | 55780 | 56291 | 75038 | n/m⁴ | n/m⁴ | n/m⁴ |
| isaacgym_parity_240 | env | 44849 | 45176 | 57662 | n/m⁴ | n/m⁴ | n/m⁴ |
| isaacgym (PhysX), ref⁵ | env / physics | 38754 / 100918 | - | - | - | - | - |

Motion loading (startup, not steady-state SPS): 4096 envs, 129-motion file, ~4 min -> 4.9 s cold, 4.5 s warm (§3.5).

Sources: start = the pre-harness mujoco-warp 3.15 baseline; 383ceff, 6151f02, cb8ad4a, 82b9364, 05a3d68 = their
commit messages, cross-checked against reruns at the same tree.
¹ the 3.15 baseline's rollout(ckpt) figure (61019) was later found to be profiler-contaminated; no clean unprofiled
ckpt-rollout number exists at that point. ² measured *under* `--profile` (profiling adds overhead), not directly
comparable to the unprofiled 76804/86608 cells; no unprofiled ckpt-rollout figure was recorded at 383ceff.
³ measured on the working tree right before cb8ad4a, tagged 6151f02. ⁴ not re-benched after cb8ad4a — the host-sync
fixes are in the env/torch layer, not the solver, so similar relative gains are expected but unverified; still true
after 05a3d68 (training-loop only) and the env step work in §3.7. ⁵ measured with the pre-harness `8616ce2`
script, one other GPU process running (GPU not idle) — order-of-magnitude reference only, not comparable.
⁶ an independent re-measurement right after cb8ad4a saw 139754 on the same (6151f02) solver settings; cb8ad4a didn't
touch the solver, so the ~1% gap to 141367 is run-to-run noise, not a regression. ⁷ `now` =
05a3d68 plus the env step work in §3.7 (E1+E2+E3); training is the median SPS over epochs 3-12 of a
`scripts/train.py` run at that state. ⁸ training SPS: 36.6k pre-05a3d68 (a synced-timer trace of the pre-bf16 loop)
to 53084-53627 at 05a3d68 (a `scripts/train.py` run at that commit; 53627 is the median over 150 epochs / 20.1M
steps in a longer run); see §3.6.

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

Add `--checkpoint <path-to-a-checkpoint>` for checkpoint actions in `rollout` mode (the SPS numbers in this doc
labeled "ckpt" used a trained checkpoint; plain "rollout" numbers are the random-init policy). `PYTHONPATH=$PWD` is
only needed if not already exported in the shell (see `AGENTS.md`).

Comma lists sweep every (mode, N, preset) combo, one subprocess per (N, preset) (backends are never freed).
`--profile`, after all timing rows: per-phase CUDA-event/CPU ms by call path, host sync counts by call site
(`torch.cuda.set_sync_debug_mode`), a 10-step `torch.profiler` trace, and a GPU- vs launch-bound verdict.
`--record PATH` / `--check PATH` take a single (num-envs, solver-preset) config — a comma-swept invocation refuses to
run them. `--record` saves a seeded 200-step trace (obs/reward/done/root_states/dof_pos every 10 steps) to `PATH`;
`--check PATH` replays the same seed and compares against it, and refuses if the recorded config doesn't match the
current one. Record a fresh reference before a change and `--check` against it after. `mujoco_gpu` is **not bitwise
deterministic** run to run (fp32 noise grows to ~0.1 by step 200), so `--check` only requires each field's max diff
at step 10 within `--tol` (default 1e-2).

```
pixi run nsys profile --cuda-graph-trace=node -t cuda,nvtx -o nsys_newton_120 \
  python scripts/bench_sps.py --mode env --num-envs 4096 --solver-preset newton_120 --repeats 1
nsys stats --report cuda_gpu_kern_sum nsys_newton_120.nsys-rep
```

Per-kernel breakdown inside the CUDA graph (phases are NVTX ranges; `torch.profiler` is empty under nsys). **Caveat:**
`nsys` combined with `bench_sps.py`'s `--profile` hangs — the process sits idle with no output. Run nsys without
`--profile` and without `-t osrt` instead, as above.

`bench_sps.py --fixed-reset-times` replaces the per-step `torch.rand` reset-time draw with a deterministic
golden-ratio phase per motion id, so `--record`/`--check` stay meaningful across changes to the reset path (used to
verify §3.7's E3).

Training-loop SPS (not a `bench_sps.py` mode) is read off `scripts/train.py`'s own TUI (SPS and per-phase percent),
or measured with a small wrapper that calls `clean_pufferl.evaluate`/`train` directly and wraps `vecenv.send`,
`policy.forward`, GAE, backward and optimizer-step with CUDA-synced timers and `torch.cuda.set_sync_debug_mode` sync
counting, median of several epochs after 2 warmup epochs.

Rules used for every change below: GPU otherwise idle, one config per process; median of 3 repeats (300 steps, 50
warmup, seed 0); keep a change only if it gives >=3% SPS at the measured config (the rule that dropped the
motion-lib query dedup in section 4). Every kept change passes `bench_sps.py --check` against a freshly recorded
reference trace and an eval on the full motion set, plus `pixi run -e default test` (and `-e isaacgym test` for that
backend) — run for `cb8ad4a` and `383ceff` (newton 21 passed, default 4 passed 1 skipped, per `383ceff`'s commit
message), not separately re-run for `6151f02` ("Tests: none added" per its commit message). Absolute SPS drifted
5-10% within a single measurement session (host load), so E1-E3's keep/drop calls used interleaved A/B runs against
a HEAD worktree rather than single absolute numbers.

## 3. What was done

### 3.1 Measurement harness (`8616ce2`, `bf3437a`)

**Problem.** No way to measure env/backend throughput beyond ad hoc scripts.
**Change.** `8616ce2` added a minimal SPS script; `bf3437a` extended it into the harness in section 2.
**Result.** Everything else here is measured with it. The first full baseline was recorded on mujoco-warp 3.15, then
repeated on 3.16.
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
step at 4096 envs (the mujoco-warp 3.15 in-graph breakdown).
**Change**, in `mujoco_gpu.py`: `mujoco_warp._src.types.M_BLOCK_DENSE_MAX = 80` (process-global constant, read in
`put_model()`); `solver.mjw_model.block_dim.cholesky_factorize = cholesky_factorize_solve = 128`; `NJMAX` 800 -> 256
(`NCONMAX` unchanged at 200). At N=4096: sparse baseline ~107k SPS; `dense80` alone (default block_dim=32) drops to
85.2k (-21%); `dense80+bd128` reaches 134.9k (+26%) — the block_dim bump turns the dense path from a regression into
a win, not just a further gain.
**Result**, 4096 envs, physics / env SPS: newton_120 107k->141k / 76.5k->93.7k; newton_180 75.7k->101k / 56.2k->70.7k;
isaacgym_parity_240 56.3k->75.0k / 45.2k->57.7k. `bench_sps.py --check` PASS; eval 0.992 / 39.4 mm, overflow NONE;
compute-sanitizer memcheck 0 errors.
**Risks/caveats.**
- `M_BLOCK_DENSE_MAX` / `block_dim` are private mujoco-warp internals (`BlockDim` is experimental upstream) —
  re-check both after every mujoco-warp bump.
- `njmax=256` is sized off an observed peak of 108 constraint rows at `njmax=800`, and 100 rows once capped to 256 —
  ~2.4x headroom over the observed peak, not a hard bound. `solver_stats()` samples only the last substep of each
  `simulate()`, so the true peak may be higher. (The `6151f02` commit message and an in-code comment cite a measured
  peak of 84 rows; that number was corrected by the later measurement in §3.7's E2.) An overflow is recorded in
  `solver.mjw_data.overflow` but only surfaced when `solver_stats()` runs, which happens at process exit here
  (backend is never freed mid-run) — a mid-run overflow is silent until then.
- `jacobian="dense"` errors outright for nv>60 (observed directly: `ValueError('Dense is unsupported for nv > 60
  (nv = 75)')`); this change uses the dense tile-Cholesky *factorization* path, not `jacobian="dense"`.

### 3.4 Remove host syncs from the env step (`cb8ad4a`)

**Problem.** `--profile` sync counting showed 78 host syncs per env step, 85 per rollout step, from TorchScript
scalar writes, boolean-mask indexing in `slerp`, and `Normal`'s validating constructor — each forces a CPU-GPU round
trip.
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

### 3.5 Motion loading: dedup, batched FK, GPU height curve, disk cache (`82b9364`)

**Problem.** `load_motions` ran the SMPL height fix and FK once per *env slot*, not once per unique motion, on a
single CPU thread. At 4096 envs with a 129-motion file this took ~4 min, with a 4096-item progress bar stalling
around 13-14 items/s. A profiling run with a single unique motion repeated across 300 env slots shows where it went:
~23 s for 300 redundant `fix_trans_height`/FK calls (~13/s) — batched matrix multiplies (8400 calls, ~4.8 s total)
and the skeleton forward-kinematics/rotation ops (global transform ~4.5 s, local rotation ~2.3 s) dominate.
**Change** (`motion_lib.py`): process each unique motion once; batch FK across motions; run the SMPL height curve on
GPU; cache the per-motion height curve on disk next to the motion file, in `*.motion_cache/`, keyed on the motion
file's path/mtime/size, the SMPL settings, `max_length`, the SMPL model file (path/mtime/size), and
`_MOTION_CACHE_VERSION`. Per-env crop and heading draws keep the old RNG order. Removed `motion_aa` (unused,
misaligned under cropping, missing the per-env heading). `num_thread` stays 1 — Docker `/dev/shm` limits it.
**Result**, 4096 envs: 129-motion file ~4 min -> 4.9 s cold, 4.5 s warm; all-distinct-motions cold throughput 13.7 ->
120 motions/s (commit message; re-verified by diffing the old and new motion-lib output tensors, matching within
1.9e-5 on the same device). Also removes a stall that recurs during training: the motion resample every
`motion_resample_interval=500` epochs took ~270 s against a ~3.6 s/epoch baseline measured on the pre-bf16 loop,
i.e. roughly 270/(500x3.6+270) ≈ 13% of wall time lost to that one `resample_motions+reset` call every 500 epochs.
**Risks/caveats.**
- Cache correctness depends on the key covering everything the height curve reads; Codex review (3 rounds) added
  atomic cache writes, handling for read-only cache dirs, and the SMPL model file in the key.
- `num_thread=1` is unchanged — multi-threaded loading is blocked on the Docker shm limit, not on this change.

### 3.6 Train the policy in bf16, drop host syncs from the PPO loop (`05a3d68`)

**Problem.** A synced-timer trace of the pre-bf16 loop (CUDA-synced timers plus `torch.cuda.set_sync_debug_mode`)
showed the PPO update at 56.2% of wall time (vs 43.8% rollout) and 1079 host syncs per epoch, from per-`.item()`
grad-norm accumulation (512 calls), CPU-side experience buffers, and a Python sort of the training data each epoch.
**Change** (`clean_pufferl.py`, `policy.py`, `config.ini`'s `[train] bf16`): `torch.autocast("cuda", bfloat16)`
around the policy MLP forward in both rollout and update; the mu/value heads, losses, GAE and Adam optimizer stay
fp32. Experience buffers live on GPU instead of being copied to numpy each step. A stable `argsort` replaces the
per-epoch Python sort. Grad norm comes from `clip_grad_norm_`'s return value instead of 512 `.item()` calls per
update (this also changes `before_clip_grad_norm` to the global L2 norm, not a sum of per-parameter norms).
Explained variance now pairs sorted values with sorted advantages.
**Result**, full config, 4096 envs: 36.6k -> 53.1k-53.6k SPS (36470 SPS on the pre-bf16 loop; 53084 and a 150-epoch
median of 53627 over 20.1M steps at 05a3d68). Host syncs 1079 -> 186 per epoch. Over 20M steps the learning curves
(episode_return, episode_length, losses) stay within the spread of two fp32 baseline seeds — e.g. episode_return at
20M steps is 2.4546/2.4435 for the two baseline seeds vs 2.4453 for bf16. LSTM policies smoke-train with bf16 on.
**Risks/caveats.**
- `mean_bound_loss` is logged but contributes no gradient: it's read once from a `torch.no_grad()` rollout forward
  (pre-existing, not introduced here — see section 5).
- `before_clip_grad_norm` and `loss/explained_variance` are not numerically comparable to pre-05a3d68 runs (the
  computation changed, not just the dtype).
- LSTM hidden/cell state is kept fp32 across steps (`.float()` after the autocast block); only the MLP body runs in
  bf16.

### 3.7 Env step E1-E3: cheaper rotations, smaller NJMAX, syncless masked reset

On top of `05a3d68`.

**Problem.** With bf16 training in place, the env step was still 39.26 ms at 4096 envs / newton_120:
`my_quat_rotate`'s `torch.bmm` became 51.5 cuBLAS `gemvx` calls/step (2.12 ms); `NJMAX=256` was sized for a peak that
was never re-measured after `cb8ad4a`; and the post-physics reset path still ran one `torch.nonzero(done)` host sync
plus a second, subset-only observation pass and a second, redundant time-t motion query.
**Change.**
- **E1**: `my_quat_rotate`'s `torch.bmm` -> elementwise mul-sum; `quat_to_tan_norm` rewritten in closed form
  (`torch_utils.py`, -6 net LOC). Identical outputs to HEAD on 100k random unit quats.
- **E2**: `NJMAX` 256 -> 160, plus a per-log-interval overflow warning. All-substep peak constraint rows, measured
  with a warp kernel after every solver step over 1.23M env-steps: max 96 / p99 68 / median 44 with random actions,
  max 64 / p99 60 / median 44 with checkpoint actions. 160 is what drives the row count: 1.67x the measured 96-row
  peak (vs. 1.48x the 108-row peak cited in the `6151f02`-era logs). `warn_overflow()` reads `mjw_data.overflow`
  once per log interval (= once per epoch at the default 4096-env/131072-batch config) and warns if it's non-zero;
  overflow was NONE in every run below.
- **E3**: one time-t `get_motion_state` shared by reward, reset flags and eval mpjpe; a masked full-batch reset
  (`_reset_masked`, `torch.where`/`masked_fill_` over all envs, Random/Start ref-state init only) replacing the
  indexed reset on the hot path; observations computed once, after the reset; no `nonzero` in the wrapper
  (`episode_count` becomes a GPU counter). The indexed `reset(env_ids)` stays for eval/play/resample.
  `bench_sps.py --fixed-reset-times` (§2) makes the reset-time draw deterministic so `--check` still means something
  across this change.
**Result**, 4096 envs, newton_120, SPS: env 39.26 -> 31.71 ms (104339 -> 129155, +24%); physics 138695 -> 142760;
rollout(ckpt) 84958 -> 97332; training 53084 -> 57900 (+9%). Syncs/step: env ~1.2 -> 0 (log-interval reads only).
`bench --check` PASS at each step; 129-motion eval unchanged within noise (HEAD 0.984/38.64 mm vs E1+E2+E3
0.992/39.42 mm); `pixi run -e default test` passes (21 default, 4 isaacgym + 1 skipped).
**Risks/caveats.**
- Absolute SPS drifted 5-10% within this measurement session (host load), so the per-change keep/drop calls used
  interleaved A/B runs against a HEAD worktree, not the single-process absolute numbers quoted above (see §2).
- E3 only supports Random/Start reference-state init on the masked path; anything else is asserted. `mujoco_cpu`'s
  `reset_envs` had to be fixed to handle an all-false mask.
- The RNG stream for reset times changed (draws N per step instead of k), so an old `--check` reference needs a
  fresh recording with `--fixed-reset-times` to stay meaningful.
- The time-t "light" query that would skip `dof_pos`/`dof_vel` for reward+reset was not done; `get_motion_state` is
  unchanged, just called once instead of twice.

## 4. Tried and not taken

| idea | result | why not |
|---|---|---|
| `jacobian="dense"` | errors | unsupported for nv>60 (nv=75) |
| `M_BLOCK_DENSE_MAX=80` alone, `block_dim=32` default | -26% (102020->75504 @ N=2048) | block_dim 32 is slower than sparse LDL at this size; needed the block_dim bump too |
| `opt.iterations` capped 3-6 | SPS flat; drift at 2s: 683 mm median @ it3 (455 it4, 189 it5, 84 it6) vs 68 mm run-to-run noise floor; it3/it4 also flag `OverflowType.ITERATIONS` | solver already exits at ~1.4 mean iters; cap only bites worst-case worlds and changes physics |
| `opt.tolerance` loosened to 1e-5/1e-4 (base 1e-6) | no SPS change | not the bottleneck |
| `ls_iterations` 5/10/20 | no SPS change | same |
| `solver="cg"` | -63% (102020->37450 @ N=2048) | needs far more iterations (mean 7.9 vs 1.4) to converge |
| `graph_conditional=False` | -54% (102020->46497 @ N=2048) | loses the conditional graph's early exit |
| SAP broadphase (`bp_tile`, `bp_seg`) vs NxN | flat to -16% | no win at this contact count; `bp_seg` notably slower |
| Smaller `nconmax` (48/64, with nj128/nj200) | flat SPS, `OverflowType.NARROWPHASE` flagged | drops real contacts |
| `njmax=128` alone (vs 256) | +2% SPS @ N=4096 (142.4k vs 139.3k dense; 111.1k vs 107k sparse), no overflow | 256 kept for headroom over the observed 100-108 row peak (section 3.3); later corrected to 160 in §3.7's E2 |
| Motion-lib query dedup (cache next step's task-obs state for reward) | +1.5% env / +1.0% rollout @ 4096 | below the 3% bar on its own; superseded by §3.7's E3, which merges the queries directly |
| The "~7 ms in-graph gap" between `torch.profiler`'s kernel-sum and the measured step time | not real: nsys shows the physics CUDA graph ~97% busy, ~0.9 ms idle per control step | `torch.profiler` misses kernels launched inside the solver's conditional (while-loop) child graphs; since the gap isn't real, a two-stream split of the world (to fill it) isn't worth doing |
| `torch.compile(policy, mode="reduce-overhead")` | crashes | its CUDA graph overwrites `policy.py`'s `mean_bound_loss` output before it's read; needs `clone()` or `torch.compiler.cudagraph_mark_step_begin()` |
| `torch.compile(policy, mode="default")` | no SPS gain (fp32, 512 envs) | not retested with bf16 autocast on, or at full (4096-env) scale |
| 8192 envs on 05a3d68 | physics 146k, env 119k SPS; ~10.5 GB VRAM (vs ~6 GB at 4096) | env scaled +14% over 4096, physics +5.5% — some headroom left, not pursued further this round |
| Custom physics kernels (patching newton or mujoco-warp internals beyond the existing `M_BLOCK_DENSE_MAX`/`block_dim` overrides) | not attempted | out of scope by user decision |

## 5. Where time goes now, and next candidates

In-graph kernel time, `physics` mode, newton_120 @ 4096 envs, from an `nsys stats --report cuda_gpu_kern_sum` run
over several hundred control steps, 05a3d68 + E1+E2+E3 (§3.7): total in-graph time is about
28.8 ms of the 31.71 ms env step, of which the dense tile Cholesky of the mass matrix
(`_tile_cholesky_factorize_solve_block`) is the largest single kernel at ~9.9 ms, followed by the constraint-solver
Hessian Cholesky at ~2.85 + 1.93 ms, `mul_m` at ~1.3 ms, and narrowphase at ~1.2 ms. The remaining ~2.9 ms of the env
step is motion queries (~0.65, down from 5 calls to 2), reward/reset-flag/reset-env/refresh/obs bookkeeping (~2.0),
GPU-bound rather than launch-bound now that E3 removed the CPU-launch-bound reset tail.

Training-epoch time split after bf16 but before E3 (§3.6's commit, 4096 envs, full config, wall/epoch ~2.45 s):
env-step send/recv ~53% of wall, policy forward (rollout + update combined) ~17%, and the PPO update's own
backward+optimizer-step work ~1 s of that ~2.45 s. E3 then cut the env-step share further (see §3.7's result).

Next candidates, roughly ranked, all estimates pending measurement:

1. **Compile or CUDA-graph the post-physics env work** (obs/reward/reset), now that E3 made every shape fixed and
   sync-free — extends the pattern from §3.4/§3.7.
2. **One graph per control step** with the Newton gym-state conversions fused and done once per control step instead
   of every substep, or a whole-step graph covering physics + post-physics work together.
3. **Fused Adam and `torch.compile` under bf16 for the update.** The one tried `torch.compile` attempt (see section
   4) predates bf16 and was fp32-only; worth retrying now that the update runs mixed precision.
4. **GAE on the GPU**, removing the remaining CPU round trip in the update's advantage computation.
5. **Re-test 8192 envs** on the current tree (last measured on 05a3d68, before E1-E3); env and physics both had
   headroom at that point (section 4).
6. **Fewer PPO update epochs or a smaller critic** — would change learning, not just speed; left to a deliberate
   call rather than bundled into a throughput change.
7. **Dense Cholesky is still the single largest in-graph kernel (~9.9 ms).** Further gains there need a faster
   mujoco-warp factorization path or exploiting tree sparsity more than today's block-dense trade; not attempted.

**Open doubt, not yet acted on:** `mean_bound_loss` is read once from a `torch.no_grad()` rollout forward
(`policy.py`), so it never contributes a gradient to the loss it's added to in `clean_pufferl.train` — this predates
both bf16 and the env-step work and hasn't been fixed.
