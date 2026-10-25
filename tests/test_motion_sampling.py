"""CPU checks for puffer_phc/sampling.py (port of tmp/sampling/sim_r3.py tests) and its HumanoidPHC hooks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puffer_phc.humanoid_phc import HumanoidPHC
from puffer_phc.sampling import ALPHA, MotionSampler, StartSampler, cap_slots, slot_counts

N = 4096


@pytest.mark.parametrize("M", [1, 2, 7, 20, 21, 100, 1000, 2048, 2049, 4097, 11000])
def test_slot_counts(M):
    rng, C = np.random.default_rng(M), cap_slots(M, N)
    for h in (np.zeros(M), np.ones(M), rng.random(M) * (rng.random(M) < 0.3), np.r_[1e3, np.zeros(M - 1)]):
        c = slot_counts(h, np.full(M, -1), rng, N)
        assert c.sum() == N and c.min() >= 0 and c.max() <= C
        if C == -(-N // M):  # small M: exactly uniform for any h
            assert c.max() - c.min() <= 1
        if M <= N // 2:  # coverage floor
            assert c.min() >= int(ALPHA * N) // M
    # every motion loads within ceil(2M/N) loads under adversarial h
    h, last = np.r_[np.full(M // 2, 5.0), np.zeros(M - M // 2)], np.full(M, -1)
    for load in range(-(-M // int(ALPHA * N))):
        last[slot_counts(h, last, rng, N) > 0] = load
    assert (last >= 0).all()


def test_slot_counts_no_low_id_bias():
    M, rng = 11000, np.random.default_rng(0)
    tot = sum(slot_counts(np.ones(M), np.full(M, -1), rng, N) for _ in range(6))
    assert abs(tot[:2048].mean() - tot[2048:].mean()) < 0.1 * tot.mean()


def test_slot_counts_decay():
    M, rng = 100, np.random.default_rng(1)
    c = [slot_counts(np.r_[0.5**k, np.zeros(M - 1)], np.full(M, -1), rng, N)[0] for k in range(40)]
    assert all(b <= a + 1 for a, b in zip(c, c[1:])) and c[0] == cap_slots(M, N) and c[-1] <= -(-N // M) + 1


def test_motion_sampler():
    m = MotionSampler(200, "pmcp", "cpu")
    ids = m.sample(1000)
    assert ids.shape == (1000,) and ids.device.type == "cpu" and m.load_count == 1 and (m.last == 0).all()
    m.record_eval([5])
    m.record_eval([7])
    assert m.h[5] == 0.5 and m.h[7] == 1 and m.h.sum() == 1.5
    assert (m.sample(1000) == 7).sum() > (m.sample(1000) == 3).sum() + 5 and m.load_count == 3
    torch.manual_seed(0)
    u = MotionSampler(200, "uniform", "cpu")
    u.record_eval([7])
    torch.manual_seed(0)
    assert torch.equal(u.sample(1000), MotionSampler(200, "uniform", "cpu").sample(1000))  # uniform ignores failures


def fake_lib(n, M=3):
    """Slot i holds motion i % M, with 30 + 10 * (i % 7) frames."""
    i = torch.arange(n)
    return SimpleNamespace(
        _curr_motion_ids=i % M, _motion_num_frames=30 + 10 * (i % 7), _motion_dt=torch.full((n,), 1 / 30)
    )


def test_score_peaks_at_target():
    s = StartSampler(1, "sgs", 1, "cpu", target=0.3).score(torch.linspace(0.01, 0.99, 99, dtype=torch.float64))
    assert s.max() <= 1 and abs(0.01 * (s.argmax().item() + 1) - 0.3) < 0.011


@pytest.mark.parametrize("mode, fu, fw", [("sgs", 0.3, 0.5), ("uniform", 0.8, 0.0)])  # uniform, weighted branches
def test_start_times(mode, fu, fw):
    n = 20000
    s, lib = StartSampler(3, mode, n, "cpu"), fake_lib(n)
    s.succ[:] = torch.rand_like(s.succ) * 50
    s.att[:] = s.succ + 50
    slots = torch.arange(n)
    frame, F, b = (s.start_times(lib, slots) * 30).round().long(), lib._motion_num_frames, s.bin
    assert ((frame >= 0) & (frame < F - 1)).all()  # never on the last frame
    assert ((frame >= b * F // 20) & (frame < (b + 1) * F // 20)).all()  # inside the stored bin
    assert (frame[s.probe] == 0).all()
    assert abs(s.probe.float().mean() - 0.2) < 0.02
    if mode == "uniform":  # frames are uniform over [0, F - 1)
        assert abs((frame - (F - 2) / 2)[~s.probe].float().mean()) < 0.6
    # branch split: 100 frames, only bin 3 near the target; its share of non-probe starts mixes both branches
    lib._motion_num_frames[:] = 100
    s.att[:], s.succ[:] = 1000.0, 0.0
    s.succ[:, 3] = 500.0
    s.start_times(lib, slots)
    expect = (fu * 5 / 99 + fw * 1.05 / (1.05 + 19 * 0.05)) / (fu + fw)
    assert abs((s.bin[~s.probe] == 3).float().mean() - expect) < 0.02


def test_start_times_mask_keeps_running_envs():
    s, lib, slots = StartSampler(3, "sgs", 1000, "cpu"), fake_lib(1000), torch.arange(1000)
    s.start_times(lib, slots)
    bin0, probe0 = s.bin.clone(), s.probe.clone()
    mask = slots % 2 == 0
    s.start_times(lib, slots, mask)
    assert (s.bin[~mask] == bin0[~mask]).all() and (s.probe[~mask] == probe0[~mask]).all()
    assert (s.bin[mask] != bin0[mask]).any()


def test_record():
    s = StartSampler(2, "sgs", 4, "cpu")
    s.bin[:], s.probe[:] = torch.tensor([3, 3, 5, 7]), torch.tensor([True, False, True, True])
    ids, steps = torch.tensor([1, 1, 1, 0]), torch.tensor([10, 10, 200, 10])
    done, term = torch.tensor([True, True, True, False]), torch.tensor([False, True, True, False])
    s.record(ids, steps, done, term)
    assert s.att[1, 3] == 2 and s.succ[1, 3] == 1  # the probe and a duplicate cell both count
    assert s.att[1, 5] == 1 and s.succ[1, 5] == 1  # terminated, but past HORIZON
    assert s.att[0].sum() == 0  # not done
    assert s.counts.tolist() == [2, 1]  # probe: ended, not terminated


def test_record_decay_keeps_ratio():
    s = StartSampler(1, "sgs", 1, "cpu")
    s.att[0, 0], s.succ[0, 0] = 600.0, 150.0
    s.record(torch.tensor([0]), torch.tensor([1]), torch.tensor([False]), torch.tensor([False]))
    assert s.att[0, 0] == 512 and s.succ[0, 0] == 128
    s.start_times(fake_lib(1, M=1), torch.tensor([0]))  # weights from decayed counts stay valid


def test_lifecycle_attribution_and_no_eval_recording():
    s, lib, slots = StartSampler(5, "sgs", 6, "cpu"), fake_lib(6, M=3), torch.arange(6)
    s.start_times(lib, slots)  # load, then reset: labels only, nothing recorded
    lib._curr_motion_ids = torch.tensor([4, 4, 3, 3, 4, 3])  # resample
    s.start_times(lib, slots)
    assert s.att.sum() == 0
    env = SimpleNamespace(flag_test=True, start_sampler=s, _motion_lib=lib)
    bins = s.bin.clone()
    assert (HumanoidPHC._sample_time(env, slots) == 0).all() and (s.bin == bins).all()  # eval leaves it alone
    s.record(lib._curr_motion_ids[slots], torch.ones(6, dtype=torch.long), torch.ones(6, dtype=torch.bool),
             torch.zeros(6, dtype=torch.bool))  # fmt: skip
    expect = torch.zeros(5, 20).index_put_((lib._curr_motion_ids, bins), torch.ones(6), accumulate=True)
    assert torch.equal(s.att, expect)


def test_heading_at_query_time_matches_baked():
    from scipy.spatial.transform import Rotation

    from puffer_phc.motion_lib import MotionLibBase
    from puffer_phc.torch_utils import quat_mul

    n, F, J = 3, 10, 24
    g = torch.Generator().manual_seed(0)
    store = {k: torch.randn(n * F, J, 3, generator=g) for k in ("gts", "gvs", "gavs", "dvs")}
    store.update(
        {k: torch.nn.functional.normalize(torch.randn(n * F, J, 4, generator=g), dim=-1) for k in ("grs", "lrs")}
    )
    rot = Rotation.from_euler("z", [[0.3], [-2.0], [3.0]])
    heading = torch.tensor(rot.as_quat(), dtype=torch.float32)
    env = torch.arange(n).repeat_interleave(F)
    baked = {
        k: store[k] @ torch.tensor(rot.as_matrix(), dtype=torch.float32)[env].transpose(1, 2)
        for k in ("gts", "gvs", "gavs")
    }
    baked["grs"] = quat_mul(heading[env, None].expand(-1, J, -1), store["grs"])

    def lib(heading, store):
        m = object.__new__(MotionLibBase)
        vars(m).update(store, length_starts=env.unique() * F, _motion_num_frames=torch.full((n,), F), _heading=heading,
                       _motion_dt=torch.full((n,), 1 / 30), _motion_lengths=torch.full((n,), (F - 1) / 30),
                       _motion_bodies=torch.zeros(n, 17), _motion_limb_weights=torch.zeros(n, 10))  # fmt: skip
        return m

    query, old = lib(heading, store), lib(torch.tensor([[0.0, 0, 0, 1]]).repeat(n, 1), {**store, **baked})
    ids, t, offset = torch.arange(n), torch.tensor([0.05, 0.11, 0.2]), torch.randn(n, 3, generator=g)
    a, b = query.get_motion_state(ids, t, offset), old.get_motion_state(ids, t, offset)
    assert all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)
    assert torch.allclose(
        query.get_root_pos_smpl(ids, t)["root_pos"], old.get_root_pos_smpl(ids, t)["root_pos"], atol=1e-5
    )
