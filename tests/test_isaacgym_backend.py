"""Fast GPU checks for the Isaac Gym backend. Run with `pixi run test`."""

import os

import numpy as np
import pytest

isaacgym = pytest.importorskip("isaacgym.gymapi")  # must be imported before torch
import torch  # noqa: E402

DATA = "data/amass_train_take6_upright.pkl"
pytestmark = pytest.mark.skipif(
    not (os.path.isfile(DATA) and torch.cuda.is_available()), reason=f"{DATA} or CUDA not available"
)


@pytest.fixture(scope="module")
def env():
    from puffer_phc.environment import make

    e = make("puffer_phc", motion_file=DATA, has_self_collision=True, num_envs=4, headless=True, backend="isaacgym")
    e.reset()
    return e


def test_seeded_steps_stay_finite(env):
    env.reset()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = rng.uniform(-1, 1, size=(env.num_envs, *env.single_action_space.shape)).astype(np.float32)
        obs, *_ = env.step(actions)
    assert torch.isfinite(torch.as_tensor(obs)).all()


def test_partial_reset_leaves_other_envs_unchanged(env):
    e = env.env
    e.reset()
    ids, other = torch.tensor([1, 3], device=e.device), torch.tensor([0, 2], device=e.device)
    root_before = e._humanoid_root_states[other].clone()
    dof_pos_before, dof_vel_before = e._dof_pos[other].clone(), e._dof_vel[other].clone()
    rb_pos_before = e._rigid_body_pos[other].clone()

    e.reset(ids)

    assert torch.equal(e._humanoid_root_states[other], root_before)
    assert torch.equal(e._dof_pos[other], dof_pos_before)
    assert torch.equal(e._dof_vel[other], dof_vel_before)
    assert torch.equal(e._rigid_body_pos[other], rb_pos_before)


def test_state_tensors_alias_backend(env):
    e, b = env.env, env.env.backend
    assert e._humanoid_root_states.data_ptr() == b.root_states.data_ptr()
    assert e._dof_pos.data_ptr() == b.dof_pos.data_ptr()
    assert e._dof_vel.data_ptr() == b.dof_vel.data_ptr()
    # rigid_body_pos is a narrowed view into the gym rigid-body-state buffer, not a copy of it.
    assert b.rigid_body_pos.untyped_storage().data_ptr() == b.all_rigid_body_state.untyped_storage().data_ptr()
    env.step(np.zeros((env.num_envs, *env.single_action_space.shape), dtype=np.float32))
    assert e._rigid_body_pos is b.rigid_body_pos  # refreshed in place, not rebound by the backend


def test_sim_and_control_dt(env):
    b = env.env.backend
    assert b.sim_params.dt == pytest.approx(1 / 60)
    assert b.dt == pytest.approx(2 * b.sim_params.dt)
