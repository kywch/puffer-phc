"""CPU checks for the AMP discriminator parts of clean_pufferl.train, run with a stub vecenv."""

import copy
from types import SimpleNamespace

import numpy as np
import pufferlib.cleanrl
import pytest
import torch
from gymnasium import spaces

from puffer_phc import clean_pufferl
from puffer_phc.policy import PHCPolicy, Recurrent

OBS, AMP, N_ENV, BATCH, MB = 24, 10, 4, 16, 8


class StubVecenv:
    num_agents, use_amp_obs = N_ENV, True
    single_observation_space = spaces.Box(-1, 1, (OBS,), np.float32)
    single_action_space = spaces.Box(-1, 1, (5,), np.float32)
    amp_observation_space = spaces.Box(-np.inf, np.inf, (AMP,), np.float32)

    def async_reset(self, seed):
        pass

    def fetch_amp_obs_demo(self, n):
        return torch.randn(n, AMP) + 1


def amp_train_data(wrapped, minibatch=MB, penalty=5.0, coef=0.5):
    torch.manual_seed(0)
    cfg = SimpleNamespace(
        seed=0, torch_deterministic=False, batch_size=BATCH, bptt_horizon=4, minibatch_size=minibatch,
        cpu_offload=False, device="cpu", compile=False, optimizer="adam", learning_rate=1e-4, env="t",
        amp_reward_coef=coef, disc_coef=1.0, disc_grad_penalty=penalty, gamma=0.99, gae_lambda=0.95, bf16=False,
        clip_coef=0.2, vf_clip_coef=0.2, clip_vloss=True, norm_adv=True, ent_coef=0.0, vf_coef=0.5, bound_coef=0.0,
        l2_reg_coef=0.0, max_grad_norm=1.0, update_epochs=1, target_kl=None, anneal_lr=False,
        total_timesteps=10**9, checkpoint_interval=10**9,
    )  # fmt: skip
    env = StubVecenv()
    policy = PHCPolicy(env, hidden_size=32)
    if wrapped:
        policy = pufferlib.cleanrl.RecurrentPolicy(Recurrent(env, policy, 32, 32))
    else:
        policy = pufferlib.cleanrl.Policy(policy)
    data = clean_pufferl.create(cfg, env, policy)
    ex = data.experience
    ex.env_ids[:] = torch.arange(BATCH) % N_ENV  # step-major rows, so the env sort is a real permutation
    ex.obs.normal_(), ex.rewards.normal_(), ex.amp_obs.normal_(), ex.values.normal_()
    ex.actions.uniform_(-1, 1)
    ex.ptr = BATCH
    return data, data.policy.policy.policy if wrapped else data.policy.policy


class Stop(Exception):
    pass


def test_style_reward_follows_own_transition(monkeypatch):
    data, _ = amp_train_data(False)  # 2 minibatches
    ex = data.experience
    monkeypatch.setattr(data.policy.policy, "discriminate", lambda x: x[:, :1])  # logit = first amp feature
    seen = {}

    def grab(dones, values, rewards, *a):
        seen["r"] = rewards
        raise Stop

    monkeypatch.setattr(clean_pufferl, "compute_gae", grab)
    r0, a0 = ex.rewards.clone(), ex.amp_obs.clone()
    with pytest.raises(Stop):
        clean_pufferl.train(data)
    idxs = torch.argsort(ex.env_ids, stable=True)
    expect = r0[idxs] + 0.5 * torch.nn.functional.softplus(a0[idxs, 0])  # -log(1 - sigmoid(x)) = softplus(x)
    np.testing.assert_allclose(seen["r"], expect.numpy(), atol=1e-5)


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "recurrent"])
def test_disc_update_and_grad_penalty(wrapped):
    data, net = amp_train_data(wrapped, minibatch=BATCH)
    init, demos, fetch = copy.deepcopy(net), [], data.vecenv.fetch_amp_obs_demo
    data.vecenv.fetch_amp_obs_demo = lambda n: demos.append(fetch(n)) or demos[-1]
    clean_pufferl.train(data)  # BCE + gradient penalty backward, also through policy.Recurrent
    grads = [p.grad for p in net._disc_mlp.parameters()]
    assert all(g is not None and g.isfinite().all() and g.abs().sum() > 0 for g in grads)
    base = data.losses.disc_loss
    data0, net0 = amp_train_data(wrapped, minibatch=BATCH, penalty=0.0)
    clean_pufferl.train(data0)
    assert not all(torch.equal(g, p.grad) for g, p in zip(grads, net0._disc_mlp.parameters()))  # GP reaches backward
    x = init.amp_obs_norm(demos[0]).requires_grad_()  # the penalty: 5 * E[|grad_x D(norm(demo))|^2], at the initial weights
    (gx,) = torch.autograd.grad(init._disc_logits(init._disc_mlp(x)).sum(), x)
    assert base - data0.losses.disc_loss == pytest.approx(5 * gx.pow(2).sum(-1).mean().item(), rel=1e-3)
