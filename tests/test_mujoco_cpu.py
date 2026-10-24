"""Fast CPU checks for the MuJoCo CPU backend (puffer_phc/backend/mujoco_cpu.py). Run with `pixi run -e newton test`."""

import os

import numpy as np
import pytest

newton = pytest.importorskip("newton")
import mujoco  # noqa: E402
import torch  # noqa: E402  (newton must be importable before this matters for mujoco_cpu's own import order)
import warp as wp  # noqa: E402

from puffer_phc.torch_utils import exp_map_to_quat, quat_rotate  # noqa: E402

DATA = "data/amass_train_take6_upright.pkl"
pytestmark = pytest.mark.skipif(not os.path.isfile(DATA), reason=f"{DATA} not present")


@pytest.fixture(scope="module")
def env():
    from puffer_phc.environment import make

    e = make(
        "puffer_phc", motion_file=DATA, has_self_collision=True, num_envs=4,
        device_type="cpu", headless=True, backend="mujoco_cpu",
    )
    e.reset()
    return e


def test_reset_envs_round_trip(env):
    b = env.env.backend
    env.env.reset()
    ids, other = torch.tensor([1, 3]), torch.tensor([0, 2])
    root_before, dof_pos_before, dof_vel_before = b.root_states.clone(), b.dof_pos.clone(), b.dof_vel.clone()

    g = torch.Generator().manual_seed(0)
    dof_pos = (torch.rand(2, b.num_dof, generator=g) * 2 - 1) * 0.75
    dof_vel = (torch.rand(2, b.num_dof, generator=g) * 2 - 1) * 1e-3
    root = torch.zeros(2, 13)
    root[:, :3] = torch.rand(2, 3, generator=g)
    root[:, 3:7] = exp_map_to_quat((torch.rand(2, 3, generator=g) * 2 - 1) * 0.5)
    root[:, 7:] = (torch.rand(2, 6, generator=g) * 2 - 1) * 1e-3

    b.root_states[ids], b.dof_pos[ids], b.dof_vel[ids] = root, dof_pos, dof_vel
    b.reset_envs(ids)
    b.refresh()

    assert torch.allclose(b.dof_pos[ids], dof_pos, atol=1e-5)
    assert torch.allclose(b.dof_vel[ids], dof_vel, atol=1e-5)
    assert torch.equal(b.root_states[ids], root)
    # Untouched envs must be bit-identical to before this reset_envs call.
    assert torch.equal(b.root_states[other], root_before[other])
    assert torch.equal(b.dof_pos[other], dof_pos_before[other])
    assert torch.equal(b.dof_vel[other], dof_vel_before[other])

    # dof_vel's child-frame convention: (w_child - w_parent) == R_child @ dof_vel, in world ang vel/rotation.
    parent = env.env.skeleton_trees[0].parent_indices
    w, rot = b.rigid_body_ang_vel[ids], b.rigid_body_rot[ids]
    rel_world = quat_rotate(rot[:, 1:], b.dof_vel[ids].view(2, -1, 3))
    assert torch.allclose(rel_world, w[:, 1:] - w[:, parent[1:]], atol=1e-4)


def test_fk_matches_motion_lib(env):
    e = env.env
    e.reset()
    ids = torch.arange(e.num_envs)
    motion_res = e._motion_lib.get_motion_state(
        e._reset_ref_motion_ids, e._reset_ref_motion_times, offset=e._global_offset[ids]
    )
    err = (e._rigid_body_pos[ids] - motion_res["rg_pos"]).norm(dim=-1).mean().item()
    assert err < 1e-3, f"FK vs motion-lib rg_pos mean error {err * 1000:.3f} mm"


def test_pd_target_mapping(env):
    # Recomputes quat(exp(Q_j @ target)) inline and checks it reaches b._target after step(), so this
    # exercises the plumbing from pd_targets into the stored drive target. It does not independently
    # verify that Q_j (b._drive_frame) itself is the correct per-joint drive frame.
    b = env.env.backend
    target = (torch.rand(b.num_envs, b.num_dof) * 2 - 1) * 0.3
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
    ids = torch.arange(b.num_envs)
    b.dof_vel[:], b.root_states[:, 7:] = 0, 0
    b.reset_envs(ids)
    b.refresh()
    q = b.dof_pos.clone()
    nj = b.num_dof // 3
    target = q + 0.05
    local_target = torch.einsum("jab,njb->nja", b._drive_frame, target.view(len(q), -1, 3)).reshape(len(q), -1)
    expected_dir = (local_target - q).view(len(q), nj, 3)  # kp * (Q_j @ target - q), sign only

    for t, local_t in zip(b._target, b._targets(target)):
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
    assert torch.allclose(wp.to_torch(b.model.joint_target_ke)[6:], s * kp, atol=1e-4)
    assert torch.allclose(wp.to_torch(b.model.joint_target_kd)[6:], s * (kd + kp / 120.0), atol=1e-4)


@pytest.mark.parametrize("preset", ["isaacgym_parity_240", "isaacgym_parity_480"])
def test_solver_presets(preset):
    # Each preset must build and step finitely, with refsafe set as the preset declares.
    from puffer_phc.backend.mujoco_cpu import SOLVER_PRESETS
    from puffer_phc.environment import make

    e = make(
        "puffer_phc", motion_file=DATA, has_self_collision=True, num_envs=2,
        device_type="cpu", headless=True, backend="mujoco_cpu", solver_preset=preset,
    )
    e.reset()
    b = e.env.backend
    off = bool(b.solvers[0].mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    assert off == (not SOLVER_PRESETS[preset]["refsafe"])
    for _ in range(3):
        b.step(torch.zeros(b.num_envs, b.num_dof))
    assert torch.isfinite(b.dof_pos).all()


def test_finite_obs_after_steps(env):
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = rng.uniform(-1, 1, size=(env.num_envs, *env.single_action_space.shape)).astype(np.float32)
        obs, *_ = env.step(actions)
    assert torch.isfinite(torch.as_tensor(obs)).all()
