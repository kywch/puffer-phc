"""Fast checks for the Newton-built backends: mujoco_cpu (MuJoCo C) and mujoco_gpu (mujoco-warp on CUDA).
Run with `pixi run -e default test`; the mujoco_gpu cases skip without CUDA."""

import os

import numpy as np
import pytest

newton = pytest.importorskip("newton")
import mujoco  # noqa: E402
import torch  # noqa: E402  (newton must be importable before this matters for mujoco_cpu's own import order)
import warp as wp  # noqa: E402

from puffer_phc.backend._newton_mujoco import pd_targets_to_newton  # noqa: E402
from puffer_phc.torch_utils import exp_map_to_quat, quat_rotate  # noqa: E402

DATA = "data/amass_train_take6_upright.pkl"
pytestmark = pytest.mark.skipif(not os.path.isfile(DATA), reason=f"{DATA} not present")
NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
BACKENDS = [("mujoco_cpu", "cpu"), pytest.param(("mujoco_gpu", "cuda"), marks=NO_CUDA)]


def make_env(backend, num_envs, **kwargs):
    from puffer_phc.environment import make

    name, device_type = backend
    e = make(
        "puffer_phc",
        motion_file=DATA,
        has_self_collision=True,
        num_envs=num_envs,
        device_type=device_type,
        headless=True,
        backend=name,
        **kwargs,
    )
    e.reset()
    return e


@pytest.fixture(scope="module", params=BACKENDS, ids=lambda p: p[0])
def env(request):
    return make_env(
        request.param, 4, solver_preset="isaacgym_parity_240"
    )  # the gains and dof_force tests need gym gains


def test_reset_envs_round_trip(env):
    b = env.env.backend
    env.env.reset()
    ids, other = torch.tensor([1, 3]), torch.tensor([0, 2])
    root_before, dof_pos_before, dof_vel_before = b.root_states.clone(), b.dof_pos.clone(), b.dof_vel.clone()
    target_before = torch.stack([t[7:] for t in b._target]).clone()

    g = torch.Generator().manual_seed(0)
    dof_pos = (torch.rand(2, b.num_dof, generator=g) * 2 - 1) * 0.75
    dof_vel = (torch.rand(2, b.num_dof, generator=g) * 2 - 1) * 1e-3
    root = torch.zeros(2, 13)
    root[:, :3] = torch.rand(2, 3, generator=g)
    root[:, 3:7] = exp_map_to_quat((torch.rand(2, 3, generator=g) * 2 - 1) * 0.5)
    root[:, 7:] = (torch.rand(2, 6, generator=g) * 2 - 1) * 1e-3
    root, dof_pos, dof_vel = root.to(b.device), dof_pos.to(b.device), dof_vel.to(b.device)

    b.root_states[ids], b.dof_pos[ids], b.dof_vel[ids] = root, dof_pos, dof_vel
    b.reset_envs(torch.tensor([False, True, False, True], device=b.device))
    b.refresh()

    assert torch.allclose(b.dof_pos[ids], dof_pos, atol=1e-5)
    assert torch.allclose(b.dof_vel[ids], dof_vel, atol=1e-5)
    assert torch.equal(b.root_states[ids], root)
    # Untouched envs must be bit-identical to before this reset_envs call.
    assert torch.equal(b.root_states[other], root_before[other])
    assert torch.equal(b.dof_pos[other], dof_pos_before[other])
    assert torch.equal(b.dof_vel[other], dof_vel_before[other])
    target = torch.stack([t[7:] for t in b._target])  # PD targets reset to the new pose, others untouched
    assert torch.allclose(target[ids], pd_targets_to_newton(b._drive_frame, dof_pos), atol=1e-6)
    assert torch.equal(target[other], target_before[other])

    # dof_vel's child-frame convention: (w_child - w_parent) == R_child @ dof_vel, in world ang vel/rotation.
    parent = env.env.skeleton_trees[0].parent_indices
    w, rot = b.rigid_body_ang_vel[ids], b.rigid_body_rot[ids]
    rel_world = quat_rotate(rot[:, 1:], b.dof_vel[ids].view(2, -1, 3))
    assert torch.allclose(rel_world, w[:, 1:] - w[:, parent[1:]], atol=1e-4)


def test_fk_matches_motion_lib(env):
    e = env.env
    e.reset()
    ids = torch.arange(e.num_envs)
    motion_res = e._motion_lib.get_motion_state(e.all_env_ids, e._motion_start_times)
    err = (e._rigid_body_pos[ids] - motion_res["rg_pos"]).norm(dim=-1).mean().item()
    assert err < 1e-3, f"FK vs motion-lib rg_pos mean error {err * 1000:.3f} mm"


def test_pd_target_mapping(env):
    # Recomputes quat(exp(Q_j @ target)) inline and checks it reaches b._target after step(), so this
    # exercises the plumbing from pd_targets into the stored drive target. It does not independently
    # verify that Q_j (b._drive_frame) itself is the correct per-joint drive frame.
    b = env.env.backend
    target = (torch.rand(b.num_envs, b.num_dof, device=b.device) * 2 - 1) * 0.3
    rotvec = torch.einsum("jab,njb->nja", b._drive_frame, target.view(-1, b.num_dof // 3, 3))
    expected = exp_map_to_quat(rotvec).view(b.num_envs, -1).cpu()
    b.step(target)
    actual = torch.stack([t[7:] for t in b._target])
    assert torch.allclose(actual.cpu(), expected, atol=1e-6)


def test_dof_force_sign(env, monkeypatch):
    # Static case: zero velocity, one physics substep, so the kp spring term dominates. refresh() reads the last
    # substep's qfrc_actuator; after all substeps, light distal joints (hands, neck) have nearly closed the error
    # and -kd*qd dominates instead.
    e, b = env.env, env.env.backend
    monkeypatch.setitem(b.preset, "substeps", 1)
    e.reset()
    b.dof_vel[:], b.root_states[:, 7:] = 0, 0
    b.reset_envs(torch.ones(b.num_envs, dtype=torch.bool, device=b.device))
    b.refresh()
    q = b.dof_pos.clone()
    nj = b.num_dof // 3
    target = q + 0.05
    local_target = torch.einsum("jab,njb->nja", b._drive_frame, target.view(len(q), -1, 3)).reshape(len(q), -1)
    expected_dir = (local_target - q).view(len(q), nj, 3)  # kp * (Q_j @ target - q), sign only

    for t, local_t in zip(b._target, pd_targets_to_newton(b._drive_frame, target)):
        t[7:] = local_t
    b.simulate()
    b.refresh()

    force = b.dof_force.view(len(q), nj, 3)
    assert b.dof_force.abs().max() > 0
    assert ((force * expected_dir).sum(-1) > 0).float().mean() > 0.8


def test_pd_gains_match_xml(env):
    # Independent of create_envs: re-parses the raw XML for stiffness/damping (not Newton's ModelBuilder,
    # whose gains are the code under test), to check the active preset's per-body gain scaling and the
    # kd += kp/120 term.
    b = env.env.backend
    mjm = mujoco.MjModel.from_xml_string(b._xml)
    kp = torch.tensor(mjm.jnt_stiffness[1:], dtype=torch.float32)  # one entry per raw hinge dof, root excluded
    kd = torch.tensor(mjm.dof_damping[6:], dtype=torch.float32)
    gains = b.preset["gains"]
    s = torch.tensor([gains.get(b.body_names[1 + d // 3], 1.0) for d in range(b.num_dof)])
    ke, kd_model = (
        wp.to_torch(a).cpu().view(-1, 75)[:, 6:] for a in (b.model.joint_target_ke, b.model.joint_target_kd)
    )
    assert torch.allclose(ke, s * kp, atol=1e-4)
    assert torch.allclose(kd_model, s * (kd + kp / 120.0), atol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda p: p[0])
@pytest.mark.parametrize("preset", ["isaacgym_parity_240", "isaacgym_parity_480", "newton_180", "newton_120"])
def test_solver_presets(backend, preset):
    # Each preset must build and step finitely, with refsafe set as the preset declares.
    from puffer_phc.backend._newton_mujoco import SOLVER_PRESETS

    b = make_env(backend, 2, solver_preset=preset).env.backend
    opt = b.solver.mjw_model.opt if backend[0] == "mujoco_gpu" else b.solvers[0].mj_model.opt
    off = bool(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    assert off == (not SOLVER_PRESETS[preset]["refsafe"])
    for _ in range(3):
        b.step(torch.zeros(b.num_envs, b.num_dof, device=b.device))
    assert torch.isfinite(b.dof_pos).all()


def test_finite_obs_after_steps(env):
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = rng.uniform(-1, 1, size=(env.num_envs, *env.single_action_space.shape)).astype(np.float32)
        obs, *_ = env.step(actions)
    assert torch.isfinite(torch.as_tensor(obs)).all()


@NO_CUDA
def test_gpu_matches_cpu():
    # Same reset state and PD targets on both backends: 2 control steps (16 physics substeps) must agree. The head
    # starts at 3 rad with its target at -3 rad: MuJoCo C servos toward the nearest representative, across pi.
    cpu, gpu = (make_env(backend, 2).env.backend for backend in (("mujoco_cpu", "cpu"), ("mujoco_gpu", "cuda")))
    head = cpu.body_names.index("Head") - 1
    cpu.dof_pos[:, 3 * head : 3 * head + 3] = torch.tensor([3.0, 0, 0])
    for name in ("root_states", "dof_pos", "dof_vel"):
        getattr(gpu, name)[:] = getattr(cpu, name).to(gpu.device)
    for b in (cpu, gpu):
        b.reset_envs(torch.ones(2, dtype=torch.bool, device=b.device))
        b.refresh()
    target = cpu.dof_pos + 0.05
    target[:, 3 * head : 3 * head + 3] = torch.linalg.solve(cpu._drive_frame[head], torch.tensor([-3.0, 0, 0]))
    for _ in range(2):
        cpu.step(target)
        gpu.step(target.to(gpu.device))
    assert (cpu.rigid_body_pos - gpu.rigid_body_pos.cpu()).norm(dim=-1).max() < 1e-3
    # Measured: body pos 1e-6 m, torque max diff 2e-3 Nm (float32 vs MuJoCo C float64) of ~38 Nm peak
    assert torch.allclose(cpu.dof_force, gpu.dof_force.cpu(), rtol=0, atol=0.05)


def test_episode_stats_include_terminal_step():
    e = make_env(BACKENDS[0], 1, termination_distance=0.05)
    zeros = np.zeros((1, *e.single_action_space.shape), np.float32)
    rewards = []
    while True:
        rewards.append(e.step(zeros)[1].item())
        if e.env.extras["terminate"].item() or len(rewards) > 200:
            break
    ret, length = (e._ep_stats[:2] / e._ep_stats[3]).tolist()
    assert length == len(rewards)
    assert ret == pytest.approx(sum(rewards), rel=1e-4)
    assert e.episode_lengths.item() == 0


def test_amp_obs(monkeypatch):
    np.random.seed(0), torch.manual_seed(0)
    e = make_env(BACKENDS[0], 4, use_amp_obs=True, termination_distance=0.05)
    g, ids = e.env, torch.arange(4)
    # right after reset(): the sim's current obs vs the reference at the start time
    ref = g._get_amp_obs(ids, g._motion_start_times)
    groups = {
        "root_h": (0, 1),
        "root_rot": (1, 7),
        "root_vel": (7, 10),
        "root_ang_vel": (10, 13),
        "dof_pos": (13, 127),
        "dof_vel": (127, 184),
        "key_pos": (184, 196),
    }
    diff = {k: (e.amp_obs[:, a:b] - ref[:, a:b]).abs().max().item() for k, (a, b) in groups.items()}
    assert all(d < (2e-3 if k == "key_pos" else 1e-5) for k, d in diff.items()), (
        diff
    )  # measured: <= 1.1e-6, key_pos 6.4e-4 (FK)
    # demo frame k is the reference at t - k*dt (root height read straight from the motion lib; t clear of the clamp)
    with monkeypatch.context() as mp:
        mp.setattr(torch, "randint", lambda *a, **k: ids)
        mp.setattr(torch, "rand", lambda n, **k: torch.full((n,), 0.5))
        demo = g.fetch_amp_obs_demo(4).view(4, 10, -1)
    t = 0.5 * g._motion_lib._motion_lengths[:4]
    for k in range(10):
        z = g._motion_lib.get_motion_state(ids, t - k * g.dt)["rg_pos"][:, 0, 2]
        assert torch.allclose(demo[:, k, 0], z, atol=1e-5), k
    # every step rolls the history by one frame; a terminated env's history is then refilled from the reference
    for _ in range(300):
        prev = g._amp_obs_buf.clone()
        e.step(np.random.uniform(-1, 1, (4, *e.single_action_space.shape)).astype(np.float32))
        out, buf, done = e.amp_obs.view(4, 10, -1), g._amp_obs_buf, e.terminals | e.truncations
        assert torch.equal(out[:, 1:], prev[:, :-1])  # also the terminal (pre-reset) rows of done envs
        assert torch.equal(buf[~done], out[~done])  # the refill leaves the other envs alone
        if done.any() and not done.all():
            break
    assert done.any() and not done.all()
    assert torch.allclose(buf[done, 1:], g._get_amp_obs_hist(ids, g._motion_start_times, 1)[done], atol=1e-5)
    d0, d1 = g.fetch_amp_obs_demo(8), g.fetch_amp_obs_demo(8)
    assert d0.shape == (8, 1960) and d0.isfinite().all() and not torch.equal(d0, d1)
