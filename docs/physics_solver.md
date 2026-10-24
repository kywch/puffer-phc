# Physics solver settings: Isaac Gym (training) vs MuJoCo-CPU (eval)

The provided checkpoint was trained on Isaac Gym/PhysX (`puffer_phc/backend/isaacgym.py`). Eval of a gym-trained
checkpoint can also run on the MuJoCo-CPU backend (`puffer_phc/backend/mujoco_cpu.py`, Newton's
`SolverMuJoCo(use_mujoco_cpu=True)`) or its GPU twin `mujoco_gpu` (`mujoco_gpu.py`, mujoco-warp; same model and
presets, agrees with CPU to ~1 um over 2 control steps). PhysX and MuJoCo disagree enough on contact and joint
dynamics that naively porting gym's numbers halved the eval success rate. This doc records what gym actually does,
what MuJoCo-CPU is configured to do to match it for eval, why, and the evidence.

## Isaac Gym / PhysX, as configured

From `puffer_phc/backend/base.py` and `isaacgym.py`:

- Policy/control runs at 30 Hz: `control_freq_inv = 2`, `dt = control_freq_inv / 60 = 1/30` s.
- `sim_params.dt = dt / control_freq_inv = 1/60` s. `SimParams.substeps` is never set, so it keeps PhysX's
  default of 2 → the actual physics step is `1/120` s, 4 physics steps per 1/30 s control step.
- `sim_params.physx.num_position_iterations = 4`, `num_velocity_iterations = 0`: PhysX's TGS solver, 4 position
  iterations. This is far from converged (64 iterations is converged; see "Root causes" below).
- PD drives: `driveMode = DOF_MODE_POS`, `stiffness`/`damping` read from the MJCF asset as kp/kd. PhysX applies
  the drive **implicitly** (backward Euler on the spring term), which is equivalent to an explicit spring plus
  extra damping `kp * dt_physics`.
- Gym's 3-hinge joints are simulated as PhysX spherical (ball) joints; `dof_pos`/`dof_vel` are the exponential
  map (rotation vector) and child-frame relative angular velocity of that ball joint, not 3 independent hinges.

## MuJoCo-CPU eval config

`puffer_phc/backend/_newton_mujoco.py` (used by both MuJoCo backends). Physics dt/substeps, contact solref and refsafe
on/off are selected via `SOLVER_PRESETS`, chosen with `[env] solver_preset` in config.ini or `--env.solver-preset`:
`isaacgym_parity_240` (8 x 1/240 s per control step, refsafe off; use it to eval the provided gym checkpoint) or
`isaacgym_parity_480` (16 x 1/480 s, refsafe on, higher fidelity, about 2x the cost). Everything else below is a fixed
constant.

| setting | value | why |
|---|---|---|
| preset `dt` / `substeps` | 1/480 s x 8, or 1/240 s x 4, per `simulate()` (1/30 s control step) | Finer than gym's 1/120 so the contact timeconst can be set to 0.005 s under MuJoCo's `refsafe` safeguard (timeconst >= 2*dt) when refsafe is on. At 1/120 the smallest safe timeconst is 1/60 s, which is too soft (see root cause 1). |
| preset `solref` | `(0.005, 1.0)` for both presets | Close to PhysX's effectively-rigid contacts; refsafe off (240 preset) does not need 0.005 >= 2*dt to hold. |
| `solimp` | MuJoCo default `(0.9, 0.95, 0.001, 0.5, 2.0)` | Stiffer solimp was tried and did not help (see results); default is fine once solref is right. |
| `condim` / friction `mu` | 3 / 1.0 | The MJCF default `condim=1` is frictionless; gym's ground plane uses friction 1.0. |
| joint type | ball joints (`convert_3d_hinge_to_ball_joints=True`), `dof_pos`/`dof_vel` as rotvec / child-frame angular velocity | Matches gym's PhysX spherical joints; 3 independent MuJoCo hinges give a different pose for the same `dof_pos` (up to 54 mm hand error at motion frame 0, R_Hand). |
| PD drive target frame `Q_j` (`smpl_humanoid_drive_frame.txt`) | per-joint rotation applied to the target before it is sent to the ball joint | PhysX applies its drive in the child body's principal-inertia frame, not the raw joint target. Equilibrium is `rotvec = Q_j @ target`; 2-38 deg of rotation depending on the joint, 0 for spheres/boxes. |
| `kd += kp * PHYSX_DT` (`PHYSX_DT = 1/120`) | added to the explicit kd | Reproduces PhysX's implicit drive (an explicit spring would be measurably faster/stiffer: 7% too fast on a single-joint step response without this term). `PHYSX_DT` is gym's physics step, independent of the preset's own `dt`. |
| preset `gains` (`GYM_TGS_GAIN_SCALE` for both presets) | ankle/toe 0.2, hip/knee 0.8, torso/spine/chest 0.5, scales both kp and kd | Gym's 4-iteration TGS solver does not converge under load; every loaded joint acts softer than a converged PD drive, not just the ankles. See root cause 2. |

## Root causes (evidence from `tmp/mujoco/REPORT.md`, `tmp/newton/dynamics_report.md` — both gitignored, local-only)

1. **Contacts are too soft — the main effect (success 0.50 → 0.95).** At dt 1/120, MuJoCo's `refsafe` clamps the
   contact timeconst to >= 2*dt = 1/60 s. At that stiffness, loaded feet slip 3-6x more than in PhysX, and the
   slip grows over the episode. Friction-side knobs (noslip, elliptic cone, impratio, condim 6) do not help, nor
   does a contact margin (tested against rest penetration, not friction: a 5 mm margin did nothing); only contact
   stiffness (solref timeconst) and a small enough dt to allow it do. Body tracking error is the same as gym for
   the first ~0.5 s and then diverges once the feet start slipping.
2. **Isaac Gym's unconverged TGS (4 position iterations) softens every loaded joint, not only the ankles.** PhysX
   at 4 iterations sags under gravity 1.1-1.2x more than converged PhysX in the legs, and far more in the spine
   (`dynamics_report.md`). Open-loop replay of gym's own PD targets gets monotonically closer to gym as hip/knee
   and torso/spine/chest gains are softened on top of the ankle/toe factor. The ankle/toe 0.2 factor alone is
   best within noise (0.15 → 0.953, 0.2 → 0.946; 0.3 → 0.915, 0.5 → 0.767 success, closed loop) and was not
   compensating for soft contacts — it holds up once contacts are fixed too.
3. **The eval itself is not deterministic (harness bug, not fixed here).** `motion_lib.py:780` picks a random
   300-frame window of longer motions whenever `is_deterministic` is False, which is true in eval. Run-to-run
   spread is about ±0.02 success on MuJoCo (and gym vs MuJoCo comparisons start from different motion windows
   for those motions). A fix would be e.g. `start = 0 if (config.is_deterministic or config.im_eval) else ...`.

## Results (closed-loop eval, `ckpt_amass_800m_acc97.pt` on 129 AMASS motions, success / MPJPE mm)

| config | success | MPJPE |
|---|---|---|
| gym (PhysX, training config) | 0.992 | 37.4 |
| old shipped MuJoCo-CPU (4x1/120, tc 1/60, ankle/toe 0.2 only) | 0.50 | 57-60 |
| 8x1/240, refsafe off, tc 0.005, no gain scales (seeds 0/1) | 0.930 / 0.961 | 43.8 / 43.0 |
| 16x1/480, tc 0.005, no gain scales ("B") | 0.95 (0.946-0.961) | 41.0-41.3 |
| **isaacgym_parity_480 preset (16x1/480, tc 0.005, hip/knee 0.8 + torso/spine/chest 0.5, seeds 0/1)** | **0.977 / 0.961** | **38.7 / 39.1** |
| **isaacgym_parity_240 preset (8x1/240, refsafe off, tc 0.005, same gains)** | **0.984** (rerun 0.992) | **39.5** (rerun 38.0) |
| isaacgym_parity_240 on mujoco_gpu (6 runs) / isaacgym_parity_480 on mujoco_gpu (2 runs) | 0.953-0.977 / 0.969, 0.977 | 38.9-40.4 / 39.0, 38.9 |
| seed noise (same config, different seeds) | about ±0.015 | about ±0.3 |

Full sweep (contact knobs, gain variants, open-loop replay diagnostics) is in `tmp/mujoco/REPORT.md` (gitignored,
not linked further here since it is local-only).

`isaacgym_parity_240` is the preset for gym-checkpoint eval (config.ini's `newton_120` is for training): about half
the cost of `isaacgym_parity_480` (~3.5 min vs ~6 min wall time for this 129-motion eval) for a success/MPJPE within
noise of it.

## Training candidates for GPU Newton (untested on GPU)

A policy trained on Newton adapts to its own contacts, so the gym-parity settings above (especially the per-body
gain scales, which exist only to match gym's specific undertrained TGS solver) do not apply. These are candidates
only, to be judged on stability and cost, not on how closely they match gym:

- **4 x 1/120 (old shipped / `tmp/newton` "fast" preset; no Newton training backend exists yet)**: cheapest,
  stable in every closed-loop CPU run tried, ~80k SPS at 1024 envs on an RTX 3090. Contacts are soft (more
  penetration and slip than PhysX).
- **4 x 1/120, refsafe off, tc 1/120 or 0.005**: same cost as fast. All 129 closed-loop CPU episodes ran without
  warnings. `tmp/newton/FINDINGS.md` saw NaN for refsafe-off/small-timeconst in an isolated passive-drop test, so
  re-run that drop/impact scenario before trusting it for training.
- **8 x 1/240, tc 1/120 (refsafe on)**: about half the SPS of fast, clearly stiffer contacts; a stable middle
  ground.
- **2 x 1/60**: not recommended — `tmp/newton/FINDINGS.md` saw NaN in the passive drop test at this dt.
