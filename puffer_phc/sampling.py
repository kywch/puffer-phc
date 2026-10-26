"""Motion (clip-level) and start-time (within-clip) sampling. See tmp/sampling/unified_r3.md."""

import math

import numpy as np
import torch

# Per-motion slot cap = max(uniform share, min(ABS_CAP * N, KAPPA * N / M)): 47 slots at M=11313, 204 (ABS_CAP) at M=944.
# GAMMA: eval-failure history decay, h = GAMMA * h + failed; lower focuses on current failures.
KAPPA, GAMMA, ABS_CAP = 128, 0.3, 0.05


def cap_slots(M, N):
    return int(max(-(-N // M), min(np.floor(ABS_CAP * N), np.ceil(KAPPA * N / M))))


def alloc_capped(w, budget, room, prio):
    """Integer counts ∝ w (all w>0), out <= room, sum == budget. Largest remainder, ties by prio."""
    out = np.zeros(len(w), int)
    while budget > 0:
        act = room - out > 0
        x = np.minimum(budget * w * act / (w * act).sum(), room - out)
        fl = np.floor(x).astype(int)
        out += fl
        budget -= fl.sum()
        if fl.sum() == 0:  # remainder: one slot each, largest fraction, random ties
            r = np.where(act, x - fl, -1.0)
            idx = np.lexsort((prio, -r))[:budget]
            out[idx] += 1
            budget -= len(idx)
    return out


def slot_counts(h, last, rng, N, alpha):
    """h: decayed eval failures (>=0). last: last load index (-1 = never). alpha: coverage fraction. Returns int counts, sum N."""
    M = len(h)
    C = cap_slots(M, N)
    if C == -(-N // M):  # cap == uniform share (M <= 20 at N=4096): exact uniform
        cnt = np.full(M, N // M)
        cnt[rng.permutation(M)[: N % M]] += 1
        return cnt
    rank = np.empty(M, int)
    rank[np.lexsort((rng.random(M), last))] = np.arange(M)  # LRU first, random ties
    F = int(alpha * N)
    cov = np.full(M, F // M)
    cov[rank < F % M] += 1  # 1. coverage, round robin
    w = h + 1.0 / M  # 2. uniform pseudomass = one failed motion in total
    return cov + alloc_capped(w, N - F, C - cov, rng.random(M))


class MotionSampler:
    """Which dataset motion each env slot holds: alpha LRU coverage, the rest eval failures (capped). uniform: h = 0."""

    def __init__(self, num_motions, mode, device, alpha):
        self.pmcp, self.device, self.alpha = mode == "pmcp", device, alpha
        self.h, self.failed = np.zeros(num_motions), np.zeros(num_motions, bool)
        self.last, self.load_count, self.fail_mass = np.full(num_motions, -1), 0, 0.0
        self.rng = np.random.default_rng(int(torch.randint(2**31, ())))

    def record_eval(self, failed_ids):
        self.failed[:] = False
        self.failed[failed_ids] = True
        self.h = GAMMA * self.h + self.failed

    def sample(self, n):
        cnt = slot_counts(self.h * self.pmcp, self.last, self.rng, n, self.alpha)
        self.last[cnt > 0] = self.load_count
        self.load_count += 1
        self.fail_mass = cnt[self.failed].sum() / n
        return torch.from_numpy(self.rng.permutation(np.repeat(np.arange(len(cnt)), cnt))).to(self.device)


FLOOR, POWER, HORIZON, CAP = 0.05, 1.5, 150, 512
FRACTIONS = {"sgs": (0.2, 0.3, 0.5), "uniform": (0.2, 0.8, 0.0)}  # probe, uniform, weighted


class StartSampler:
    """Episode start frame, by success-rate bins over each motion's frames (SGS). Probes start at frame 0."""

    def __init__(self, num_motions, mode, num_envs, device, bins=20, target=0.5):
        self.B, self.target, (self.f_probe, _, self.f_weighted) = bins, target, FRACTIONS[mode]
        self.att = torch.zeros(num_motions, bins, device=device)
        self.succ = torch.zeros_like(self.att)
        self.bin = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.probe = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # probe episodes: ended, ended without termination (clip end or step cap)
        self.counts = torch.zeros(2, device=device)

    def score(self, p):
        """SGS score: 1 at p == target, lower on either side (log space avoids underflow)."""
        t = self.target
        log = p.log() - math.log(t) + (1 - t) * (torch.log1p(-p) - math.log1p(-t)) / t
        return log.clamp(max=0).exp()

    def start_times(self, lib, slots, mask=None):
        ids, F = lib._curr_motion_ids[slots], lib._motion_num_frames[slots]
        edges = torch.arange(self.B + 1, device=F.device) * F[:, None] // self.B
        lo, hi = edges[:, :-1], edges[:, 1:].minimum(F[:, None] - 1)  # the last frame is never a start
        ov = (hi - lo).clamp_min(0)
        p = (self.succ[ids] + 1) / (self.att[ids] + 2)
        u = torch.rand(len(slots), device=F.device)
        probe, weighted = u < self.f_probe, u >= 1 - self.f_weighted
        w = torch.where(weighted[:, None], (ov > 0) * (FLOOR + self.score(p) ** POWER), ov.float())
        w = torch.where(probe[:, None], ((lo == 0) & (ov > 0)).float(), w)
        # Exponential race: a sync-free multinomial (torch.multinomial syncs on torch 2.3); a 0 draw would give NaN
        b = (w / torch.empty_like(w).exponential_().clamp_min_(1e-20)).argmax(1, keepdim=True)
        lo, hi = lo.gather(1, b).squeeze(1), hi.gather(1, b).squeeze(1)
        frame = torch.where(probe, 0, lo + (torch.rand_like(u) * (hi - lo)).long())
        if mask is not None:
            b, probe = torch.where(mask, b.squeeze(1), self.bin[slots]), torch.where(mask, probe, self.probe[slots])
        self.bin[slots], self.probe[slots] = b.view(-1), probe
        return frame * lib._motion_dt[slots]

    def record(self, ids, steps, done, terminated):
        """ids: dataset motion id per env. All N envs, no host sync."""
        cell = ids * self.B + self.bin
        self.att.view(-1).index_add_(0, cell, done.float())
        survived = done & (~terminated | (steps >= HORIZON))
        self.succ.view(-1).index_add_(0, cell, survived.float())
        s = CAP / self.att.clamp_min(CAP)
        self.att.mul_(s)
        self.succ.mul_(s)
        ended = done & self.probe
        self.counts += torch.stack([ended, ended & ~terminated]).float().sum(1)
