"""CPU checks for scripts/train.py's streamed eval metrics and its length-sorted eval batches."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

from puffer_phc.humanoid_phc import HumanoidPHC

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train  # noqa: E402


def reference(pred, gt, frames, keep):
    """compute_metrics_lite on the per-motion slices the old EvalStats passed it. pred, gt: (M, T, J, 3)."""
    ref = compute_metrics_lite([pred[k, : frames[k]] for k in keep], [gt[k, : frames[k]] for k in keep], use_tqdm=False)
    return {k: float(np.mean(v)) for k, v in ref.items()}


def trajectories(rng, M, T, J=24):
    gt = np.cumsum(rng.normal(0, 0.01, (M, T, J, 3)), 1).astype(np.float32)
    return (gt + np.cumsum(rng.normal(0, 0.003, gt.shape), 1)).astype(np.float32), gt


def test_eval_metrics_match_compute_metrics_lite():
    rng, N, T = np.random.default_rng(0), 64, 100
    pred, gt = trajectories(rng, N, T)
    L = rng.integers(4, 140, N)  # motions with L - 1 > T are cut at T frames
    valid = torch.arange(T)[:, None] < torch.as_tensor(L) - 1
    m = train.EvalMetrics(N, "cpu")
    for c in range(0, T, 16):
        p, g = (torch.from_numpy(x[:, c : c + 16]).transpose(0, 1) for x in (pred, gt))
        m.update(p, g, valid[c : c + 16])
    sums, counts = m.sums.numpy(), m.counts.numpy()
    for keep in (np.ones(N, bool), rng.random(N) < 0.7):
        new = train.reduce_metrics(sums[keep], counts[keep])
        for k, v in reference(pred, gt, L - 1, np.flatnonzero(keep)).items():
            np.testing.assert_allclose(new[k], v, rtol=1e-5)


def test_sorted_eval_reports_in_dataset_order():
    rng, M, N = np.random.default_rng(1), 10, 4
    steps = rng.integers(5, 60, M)
    term_at = rng.integers(0, 80, M)  # terminations at or after a motion's last step do not count
    pred, gt = trajectories(rng, M, 80, J=4)

    def run(fps):  # eval batches go shortest (frames / fps) first
        lib = SimpleNamespace(_frames=steps, _fps=fps, _num_unique_motions=M,
                              _motion_data_keys=np.array([f"m{i}" for i in range(M)]))  # fmt: skip
        lib.load_motions = lambda ids: setattr(lib, "_curr_motion_ids", ids)
        lib.get_motion_num_steps = lambda: torch.as_tensor(steps)[lib._curr_motion_ids]
        env = object.__new__(HumanoidPHC)
        vars(env).update(num_envs=N, device="cpu", _motion_lib=lib, _step_cap=1200, reset=lambda: None)
        env.toggle_eval_mode = lambda: env.forward_motion_samples(first=True) or M
        stats, done = train.EvalStats(SimpleNamespace(env=env)), False
        while not done:
            ids, t = env.current_motion_ids, stats.curr_steps
            env.extras = dict(terminate=torch.as_tensor(term_at)[ids] == t, body_pos=torch.from_numpy(pred[ids, t]),
                              body_pos_gt=torch.from_numpy(gt[ids, t]))  # fmt: skip
            done = stats.post_step_eval()[0]
        return stats

    stats, unsorted = run(np.full(M, 30.0)), run(steps.astype(float))
    assert stats.results == unsorted.results
    assert np.array_equal(stats.batches[0][0].numpy(), np.argsort(steps, kind="stable")[:N])  # shortest first
    success = term_at >= steps
    assert np.array_equal(stats.results_by_motion["success"], success)
    assert np.array_equal(stats.results_by_motion["played_steps"], np.minimum(term_at, steps))
    assert np.array_equal(stats.results_by_motion["motion_length"], steps)
    assert list(stats.failed_keys) == [f"m{i}" for i in np.flatnonzero(~success)]

    frames = np.where(success, steps - 1, np.minimum(steps - 1, term_at + 1))  # failed motions stop at termination
    ref_all = reference(pred, gt, frames, range(M))
    ref_succ = reference(pred, gt, frames, np.flatnonzero(success))
    for key, ref in (("mpjpe_all", ref_all["mpjpe_g"]), ("mpjpe_succ", ref_succ["mpjpe_g"]),
                     ("mpjpe_pa", ref_succ["mpjpe_pa"]), ("accel_dist", ref_succ["accel_dist"])):  # fmt: skip
        np.testing.assert_allclose(stats.results[f"eval/{key}"], ref, rtol=1e-5)
