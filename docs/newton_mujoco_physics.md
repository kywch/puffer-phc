# Newton/MuJoCo-native physics settings for training (mujoco_gpu)

These settings are for training on `mujoco_gpu` (Newton `SolverMuJoCo` on mujoco-warp). They do not try to match
Isaac Gym: no gym TGS gain scales and no gym-matched contact stiffness. For the gym-parity eval presets
(`isaacgym_parity_*`) and why they look the way they do, see [physics_solver.md](physics_solver.md).

All numbers below come from one RTX 3090, mujoco 3.15.0, mujoco-warp 3.15.0 and newton 1.7.0. The scripts and raw
results are in `tmp/newton_gpu/compare/`, which is local only (`REPORT.md` and `table.md` there).

## Ball-joint PD error wrap (fixed in mujoco_gpu.py)

MuJoCo C (>= 3.11) servos a ball-joint position actuator toward the setpoint nearest the current angle: it wraps
`ctrl - length` per axis to [-pi, pi]. mujoco-warp only does so from 3.16; newton 1.7.0 pins mujoco-warp 3.15. Newton
turns each target quaternion into a rotation vector with |angle| <= pi, so when a target crosses |angle| = pi,
mujoco-warp 3.15 applies close to a full 2*pi torque the wrong way (R_Knee: ctrl +2.72 -> -2.77 at length 1.63;
MuJoCo C +443 Nm, mujoco-warp -3578 Nm). On the gym checkpoint this failed `50020/50021 running_on_spot` in 6 of 7
GPU evals. `mujoco_gpu.py` now wraps `ctrl` against the current joint angle before every mujoco-warp step, as MuJoCo
C does.

Every GPU number in sections 1-3, except the `newton_180` replay paragraph, was measured with an earlier scratch
version of this fix (`knobs.py: wrapfix`), which wrapped against the previous substep's `actuator_length`. With the
shipped fix, `isaacgym_parity_240` on GPU scores 0.977 / 0.953 success and 39.4 / 40.8 mm MPJPE (seeds 0/1);
`running_on_spot` 50021 no longer fails, and the failed motions are mostly ones that also fail on CPU
(`one_leg_jump`, `running_on_spot` 50009/50020).

## 1. Candidate settings

Unless a row says otherwise, every setting uses: implicitfast, Newton solver, iterations 100 / ls_iterations 50,
pyramidal cone, impratio 1, solimp (0.9, 0.95, 0.001, 0.5, 2), condim 3, mu 1, armature 0.02 and the `kd += kp/120`
implicit-drive term. "tol def." is MuJoCo's default (1e-8) on CPU and Newton's default for mujoco-warp (1e-6) on
GPU. Steps = physics steps per 1/30 s control step.

| name | origin | dt | steps | integrator | iter / ls | tol | cone, impratio | contact solref / solimp | refsafe | PD gains |
|---|---|---|---|---|---|---|---|---|---|---|
| P240 (reference only) | `isaacgym_parity_240` | 1/240 | 8 | implicitfast | 100/50 | def. | pyr, 1 | (0.005,1) / default | off | gym TGS scales |
| T1 | old "fast" preset | 1/120 | 4 | implicitfast | 100/50 | def. | pyr, 1 | (1/60,1) | on | none |
| T2 | this repo | 1/120 | 4 | implicitfast | 100/50 | def. | pyr, 1 | (0.005,1) | off | none |
| T2b | this repo | 1/120 | 4 | implicitfast | 100/50 | def. | pyr, 1 | (1/120,1) | off | none |
| T2c | this repo | 1/120 | 4 | implicitfast | 100/50 | def. | pyr, 1 | (0.0125,1) | off | none |
| M6 | this repo | 1/180 | 6 | implicitfast | 100/50 | def. | pyr, 1 | (1/90,1) | on | none |
| M6o | this repo | 1/180 | 6 | implicitfast | 100/50 | def. | pyr, 1 | (1/120,1) | off | none |
| T3 | this repo | 1/240 | 8 | implicitfast | 100/50 | def. | pyr, 1 | (1/120,1) | on | none |
| T3c | this repo | 1/240 | 8 | implicitfast | 6/5 | def. | pyr, 1 | (1/120,1) | on | none |
| O1 | mjlab / Playground H1 | 1/180 | 6 | implicitfast | 100/50 | 1e-8 | pyr, 1 | (0.02,1) default | on | none |
| O2 | MuJoCo/Playground G1, unitree, MimicKit (dt 0.002) | 1/480 | 16 | implicitfast | 100/50 | 1e-8 | pyr, 1 | (0.02,1) default | on | none |
| O3 | Newton G1 example | 1/360 | 12 | implicitfast | 100/50 | 1e-8 | elliptic, 100 | (0.02,1) default | on | none |
| O4e | MJX/Brax-style aggressive | 1/180 | 6 | Euler, eulerdamp off | 4/5 | def. | pyr, 1 | (0.02,1) | on | none |
| O4i / O4i8 | same, implicitfast | 1/180 | 6 | implicitfast | 4/5 or 8/5 | def. | pyr, 1 | (0.02,1) | on | none |
| O5 | SMPL/PHC MuJoCo XML | 1/180 | 6 | implicitfast | 100/50 | 1e-8 | pyr, 1 | (0.015,1) / (0.9,0.99,0.003) | on | none |
| S2 | coarsest | 1/60 | 2 | implicitfast | 100/50 | def. | pyr, 1 | (1/30,1) | on | none |

## 2. Measured results (GPU unless marked)

| name | steps | stable (NaN / overflow) | drop pen. peak mm | impact pen. peak mm | tracking ground pen. mean / p95 mm | lying-flat rest pen. mm | foot slip mm/s | GPU SPS @1024 / @4096 | GPU-CPU div. p90 @2 s mm, frac >1 cm @4 s | GPU-GPU rerun same | niter mean / max | ckpt eval GPU s0, s1 (secondary) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P240 (ref) | 8 | ok | 18.0 | 14.5 | 0.5 / 1.5 | 0.07 | 44 | 43.3k / 54.0k | 0.4, 0.26 | 0.9, 0.22 | 2.5 / 8 | 0.984, 0.992 |
| T1 | 4 | ok | 66.0 | 54.8 | 5.3 / 17.6 | 0.58 | 114 | 76.9k / 99.9k | 0.5, 0.25 | 0.0, 0.12 | 3.2 / 9 | 0.636, 0.651 |
| T2 | 4 | ok, but chaotic contacts | 55.3 | 53.6 | 1.8 / 6.8 | 0.74 | 112 | 81.9k / 105.2k | 205, 0.49 | 225, 0.46 | 2.9 / 10 | 0.674, 0.736 (MPJPE 71-74) |
| **T2b** | 4 | ok | 39.8 | 35.8 | 2.6 / 9.4 | 0.22 | 83 | 77.6k / 102.6k | 0.1, 0.30 | 0.1, 0.19 | 3.3 / 10 | 0.806, 0.783 |
| T2c | 4 | ok | 55.3 | 44.8 | 3.9 / 13.3 | 0.41 | 100 | not run (same dt/steps as T1) | 0.1, 0.22 | 0.1, 0.12 | 3.3 / 9 | 0.713, 0.690 |
| M6 | 6 | ok | 46.9 | 37.7 | 2.2 / 7.7 | 0.35 | 67 | 54.7k / 69.6k | 0.1, 0.19 | 0.0, 0.12 | 2.7 / 8 | 0.907, 0.891 |
| **M6o** | 6 | ok | 36.4 | 31.9 | 1.6 / 5.9 | 0.22 | 58 | 56.0k / 70.3k | 0.1, 0.20 | 0.3, 0.21 | 2.8 / 9 | 0.876, 0.907 |
| T3 | 8 | ok | 34.3 | 28.3 | 1.1 / 4.0 | 0.22 | 51 | 42.0k / 53.7k | 0.2, 0.27 | 0.1, 0.14 | 2.5 / 8 | 0.961, 0.992 |
| T3c | 8 | ok, ITERATIONS flag | 35.2 | 28.3 | 1.2 / 4.1 | 0.22 | 51 | 42.5k / 54.0k | 3.6, 0.29 | 0.2, 0.12 | 2.5 / 6 | 0.961, 0.992 |
| O1 | 6 | ok | 69.5 | 57.1 | 4.6 / 15.1 | 0.71 | 87 | 53.7k / 69.2k | 0.1, 0.17 | 0.0, 0.06 | 2.6 / 8 | 0.783, 0.791 |
| O2 | 16 | ok | 55.5 | 45.7 | 2.8 / 8.9 | 0.71 | 51 | 22.4k / 27.8k | 0.3, 0.32 | 0.1, 0.15 | 1.6 / 6 | 0.938, 0.946 |
| O3 | 12 | ok | 58.9 | 48.0 | 3.1 / 9.8 | 0.76 | 55 | 15.2k / 20.0k | 5.7, 0.46 | 0.1, 0.14 | 5.2 / 18 | 0.899, 0.891 |
| O4e | 6 | **NaN in every scenario (GPU and CPU)** | - | - | - | - | - | not run | - | - | 3.1 / 4 | crashed |
| O4i | 6 | ok, ITERATIONS flag | 68.3 | 57.1 | 4.6 / 15.2 | 0.71 | 86 | 55.5k / 69.6k | **383, 0.89** | 0.1, 0.16 | 2.6 / 4 | 0.752, 0.775 |
| O4i8 | 6 | ok, ITERATIONS flag | 69.4 | 57.1 | 4.6 / 14.7 | 0.71 | 86 | not run | 0.2, 0.24 | 0.0, 0.13 | 2.6 / 8 | 0.783, 0.752 |
| O5 | 6 | ok | 55.8 | 45.9 | 3.2 / 10.8 | 0.71 | 79 | 50.4k / 66.1k | 0.1, 0.25 | 0.1, 0.13 | 2.9 / 10 | 0.783, 0.798 |
| S2 | 2 | no NaN, very soft | 119.6 | 101.5 | 14.2 / 42.8 | 1.67 | 358 | not run | not run | not run | 3.4 / 9 | 0.023, 0.016 |

Where it was run, CPU (MuJoCo C) agrees with GPU on the same setting:
- Stability: no NaN except O4e, and the same flat-rest penetration.
- Closed-loop tracking penetration and foot slip: mean penetration within 0.1 mm, p95 within 0.7 mm, slip within 1-2 mm/s (T1, T2, T3, O2, O3, O5).
- Checkpoint success, CPU s0 vs GPU s0/s1: T1 0.667, T3 0.961, O1 0.775, O2 0.938, O3 0.891, O5 0.806.

`newton_180` as shipped (no gym gains), open-loop replay of the M6o seed-0 PD stream (129 envs, early termination
off) on GPU and CPU: fraction of envs >1 cm apart is 0.11 at 2 s and 0.38 at 4 s, against 0.11 / 0.34 for CPU vs
CPU (1e-6 rad probe) and 0.09 / 0.38 for GPU vs GPU rerun. Ground penetration (mean 1.66 vs 1.67 mm, p95 7.10 vs
7.14) and foot slip (135 vs 137 mm/s) match. Without gym gains this policy falls, so the stream is contact-heavy.

On the CPU-vs-CPU chaos probe (1e-6 rad on the first PD target), divergence at 2 s stays at about 0.03 mm p90 or less, and
0.03-0.15 of envs pass 1 cm by 4 s. The one exception is T2, at 174 mm p90.

## 3. How each metric was measured

- **Stable**: `stab.py`, 64 envs, 3 s, PD targets held at the start pose, no gym gains, sampled every physics
  step. The four scenarios are: reset pose dropped from +1 m; lying flat (T-pose) dropped 0.3 m; standing reset pose;
  and an impact from +0.3 m with random root lin vel (vz -3, |vxy| <= 2 m/s) and ang vel (<= 4 rad/s). "Stable"
  means no NaN; mujoco-warp overflow flags are listed.
- **Drop / impact pen. peak**: per world, the deepest ground-contact penetration in each scenario, averaged over worlds.
- **Lying-flat rest pen.**: mean ground penetration over the last 0.5 s of the flat scenario.
- **Tracking ground pen.**: closed-loop checkpoint eval, first 120 control steps. Per env and step, the deepest
  ground-contact penetration, as mean / p95.
- **Foot slip**: same closed-loop trace. Mean horizontal speed of the ankle/toe bodies while they are within 1 cm of
  their low (5th-percentile) height.
- **GPU SPS**: `sps.py`, `backend.step` with hold-pose targets on the CUDA-graph path. Env-steps/s over 60 control
  steps after 10 warm-up steps. nvidia-smi showed no other compute process; CPU-only replays ran concurrently on the CPU.
- **GPU-CPU div.**: the PD-target stream of one GPU closed-loop eval (seed 0, 129 envs), replayed open loop on GPU and
  on CPU from identical reset states. The value is per-env max body-position error: p90 at 2 s, and the fraction of
  envs above 1 cm at 4 s. "GPU-GPU rerun" replays the same stream on GPU against the original run (mujoco-warp is not
  bitwise deterministic run to run), which is the noise floor that GPU-CPU should be compared with.
- **niter**: mujoco-warp solver iterations in the eval (last substep of each `simulate()`).
- **Ckpt eval**: `ckpt_amass_800m_acc97.pt` on 129 motions, with seeds that fix the motion windows. The setting is
  run *with* gym TGS gains added, because without them this checkpoint scores about 0. Use it only as a rough
  contact-fidelity proxy: the checkpoint was trained on PhysX. Seed noise is about ±0.02.

## 4. Recommendation

Cost is set almost entirely by physics steps per control step: about 100k SPS at 4, 70k at 6, 54k at 8 and 28k at
16 (4096 envs). Within a step count, contact quality depends mostly on the contact timeconst. MuJoCo's default
solref of 0.02 (O1-O4; O5 uses 0.015) gives 2-4x more tracking penetration than tc = 1/120. Iteration caps buy nothing: mujoco-warp
already exits early (2.5-3 mean iterations) and the 4096-env graph was not faster. Euler with these PD gains blows up.

Both are in `SOLVER_PRESETS` (`backend/_newton_mujoco.py`), separate from the `isaacgym_parity_*` ones: select with
`--env.solver-preset newton_180` or `newton_120`.

- **newton_180 (M6o), primary.** It costs 1.3x less than 8 steps, and its contacts are close to T3: drop 36 vs 34
  mm, tracking penetration 1.6 vs 1.1 mm, slip 58 vs 51 mm/s. It was stable in every scenario and agrees with CPU
  as closely as GPU agrees with itself. M6 (refsafe on, tc 1/90) is the conservative fallback at the same cost, with
  slightly softer contacts.
- **newton_120 (T2b), fastest.** It has T1's cost and clearly better contacts than T1: drop 40 vs 66 mm, tracking
  pen. 2.6 vs 5.3 mm, slip 83 vs 114 mm/s, ckpt 0.78-0.81 vs 0.64-0.65. Do not go stiffer at 1/120: T2 (tc 0.005) has
  chaotic contacts.

A short training run (e.g. 1-2 h, 4096 envs, each preset) must confirm:
1. no NaN or contact/constraint overflow under random early-training actions;
2. the learning curve and success compared with an Isaac Gym run of the same budget;
3. end-to-end SPS with the policy and env overhead included;
4. that the trained gait does not exploit slip or penetration (check foot slip in eval);
5. sim-to-sim: eval the trained policy on `mujoco_cpu`.

## 5. Open questions

- refsafe off with tc = 1.5 dt (M6o) or tc = dt (T2b) was stable in these 3 s tests only. Long training with
  random actions is the real test.
- An iteration cap of 4 (O4i) makes GPU and CPU diverge far beyond the noise floor, because the unconverged
  solvers differ. Avoid caps if the policy will be evaluated on MuJoCo C. O3 (elliptic, impratio 100) also diverges
  somewhat more (p90 5.7 mm at 2 s).
- SPS was measured with hold-pose targets, where most humanoids end up lying down with many contacts. Training
  rollouts may run faster.
- T2c, O4i8, S2 SPS: not run. O4e (Euler) was not tried with smaller dt or gains.
- Seed noise on the checkpoint eval is about ±0.02, and some rows have one or two seeds only.

## Sources

- O1: mjlab [sim.py (MujocoCfg)](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sim/sim.py),
  [velocity_env_cfg.py](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/velocity_env_cfg.py);
  Playground [h1/joystick.py](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/joystick.py).
- O2: Playground [g1/joystick.py](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/joystick.py),
  [g1/base.py](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/base.py);
  unitree_mujoco [g1/scene.xml](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/scene.xml);
  MimicKit [humanoid.xml](https://github.com/xbpeng/MimicKit/blob/main/data/assets/humanoid/humanoid.xml).
- O3: Newton [example_robot_g1.py](https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_g1.py).
- O4: MJX [humanoid.xml](https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/test_data/humanoid/humanoid.xml);
  Brax [humanoid.xml](https://github.com/google/brax/blob/main/brax/envs/assets/humanoid.xml); Playground
  [berkeley_humanoid_mjx_feetonly.xml](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/berkeley_humanoid_mjx_feetonly.xml).
- O5: PHC [smpl_humanoid.xml](https://github.com/ZhengyiLuo/PHC/blob/master/phc/data/assets/mjcf/smpl_humanoid.xml).
- Also surveyed: ProtoMotions [simulator/mujoco/config.py](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/simulator/mujoco/config.py),
  [simulator/newton/config.py](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/simulator/newton/config.py);
  dm_control [suite/humanoid.xml](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/humanoid.xml),
  [suite/humanoid_CMU.xml](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/humanoid_CMU.xml);
  Gymnasium [humanoid.xml](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/assets/humanoid.xml).
