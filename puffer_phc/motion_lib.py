# BSD 3-Clause Clear License
#
# Copyright (c) 2023 Carnegie Mellon University
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gc
import glob
import time
import random
import hashlib
import os
import os.path as osp
from enum import Enum
from pathlib import Path

import joblib

import numpy as np
from scipy.spatial.transform import Rotation as sRot

import torch
import torch.multiprocessing as mp

from smpl_sim.smpllib.smpl_parser import SMPL_Parser

from puffer_phc import torch_utils, ROOT_DIR
from puffer_phc.poselib_skeleton import SkeletonMotion, SkeletonState


USE_CACHE = False
# print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)


class MotionlibMode(Enum):
    file = 1
    directory = 2


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


# Bump to invalidate every cached height curve after a processing change.
_MOTION_CACHE_VERSION = 1


def _motion_cache_dir(motion_file, fix_height, smpl_type, gender, betas, max_length):
    # Keyed on everything the height curve depends on, including the SMPL model file.
    smpl_file = ROOT_DIR / "smpl" / f"SMPL_{('neutral', 'male', 'female')[gender].upper()}.pkl"
    smpl_stat = (osp.getmtime(smpl_file), osp.getsize(smpl_file)) if osp.isfile(smpl_file) else None
    raw = repr(
        (
            _MOTION_CACHE_VERSION,
            motion_file,
            osp.getmtime(motion_file),
            osp.getsize(motion_file),
            fix_height.value,
            smpl_type,
            gender,
            tuple(betas),
            max_length,
            str(smpl_file),
            smpl_stat,
        )
    )
    key = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return Path(f"{motion_file}.motion_cache") / key


def _motion_cache_path(cache_dir, motion_key):
    item = hashlib.sha1(motion_key.encode()).hexdigest()[:16]
    return cache_dir / f"{item}.pt"


# def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
#     # Assume each joint is 3dof
#     diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
#     diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
#     dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

#     return dof_vel[1:, :].flatten()


# def compute_motion_dof_vels(motion):
#     num_frames = motion.tensor.shape[0]
#     dt = 1.0 / motion.fps
#     dof_vels = []

#     for f in range(num_frames - 1):
#         local_rot0 = motion.local_rotation[f]
#         local_rot1 = motion.local_rotation[f + 1]
#         frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
#         dof_vels.append(frame_dof_vel)

#     dof_vels.append(dof_vels[-1])
#     dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

#     return dof_vels


# Vectorized across frames.
@torch.jit.script
def compute_motion_dof_vels_jit(local_rotation, fps):
    # type: (Tensor, int) -> Tensor
    dt = 1.0 / fps
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rotation[:-1]), local_rotation[1:])
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = (diff_axis * diff_angle.unsqueeze(-1) / dt)[:, 1:, :]
    return torch.cat([dof_vel, dof_vel[-1:]], dim=0)


def _batched_fk(motion_data_list, skeleton_tree):
    # FK is per-frame independent (batch over full motions); velocity/dof_vels aren't, so those wait for the crop.
    n = len(motion_data_list)
    trans_list = [d["root_trans_offset"] for d in motion_data_list]
    pose_quat_list = [to_torch(d["pose_quat_global"]) for d in motion_data_list]
    fps_list = [d.get("fps", 30) for d in motion_data_list]
    counts = [t.shape[0] for t in trans_list]

    big_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, torch.cat(pose_quat_list, dim=0), torch.cat(trans_list, dim=0), is_local=False
    )
    big_gts, big_grs, big_lrs = big_state.global_translation, big_state.global_rotation, big_state.local_rotation

    gts_l, grs_l, lrs_l, offset = [], [], [], 0
    for f in range(n):
        sl = slice(offset, offset + counts[f])
        gts_l.append(big_gts[sl])
        grs_l.append(big_grs[sl])
        lrs_l.append(big_lrs[sl])
        offset += counts[f]
    return gts_l, grs_l, lrs_l, fps_list, counts


class MotionLibBase:
    def __init__(self, motion_lib_cfg):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1 / getattr(self.m_cfg, "step_dt", 1 / 30)  # CHECK ME: hardcoded
        print("SIM FPS (from MotionLibBase):", self._sim_fps)
        self._device = self.m_cfg.device

        self.mesh_parsers = None

        self.load_data(self.m_cfg.motion_file, min_length=self.m_cfg.min_length, im_eval=self.m_cfg.im_eval)
        self.setup_constants(fix_height=self.m_cfg.fix_height, num_thread=self.m_cfg.num_thread)

    def load_data(self, motion_file, min_length=-1, im_eval=False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
            assert len(self._motion_data_load) > 0

        data_list = self._motion_data_load

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {
                    k: v for k, v in list(self._motion_data_load.items()) if len(v["pose_quat_global"]) >= min_length
                }
            elif im_eval:
                data_list = {
                    item[0]: item[1]
                    for item in sorted(
                        self._motion_data_load.items(),
                        key=lambda entry: len(entry[1]["pose_quat_global"]),
                        reverse=True,
                    )
                }
                # data_list = self._motion_data
            else:
                data_list = self._motion_data_load

            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)

        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(
                self._motion_data_load[0]
            )  # set self._motion_data_load to a sample of the data

    def setup_constants(self, fix_height=FixHeightMode.full_fix, num_thread=1):
        self.fix_height = fix_height
        self.num_thread = max(num_thread, 1)

        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = (
            torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
        )  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches

    @staticmethod
    def compute_height_curves(ids, motion_data_list, shape_params, mesh_parsers, config, queue, pid):
        raise NotImplementedError

    @staticmethod
    def height_curve_batch(pose_aa_list, trans_list, gender, betas, mesh_parsers, fix_height_mode):
        raise NotImplementedError

    def load_motions(
        self,
        skeleton_trees,
        gender_betas,
        limb_weights,
        random_sample=True,
        start_idx=0,
        max_len=-1,
        sample_idxes=None,
    ):
        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del (
                self.gts,
                self.grs,
                self.lrs,
                self.grvs,
                self.gravs,
                self.gavs,
                self.gvs,
                self.dvs,
            )
            del (
                self._motion_lengths,
                self._motion_fps,
                self._motion_dt,
                self._motion_num_frames,
                self._motion_bodies,
            )

        torch.cuda.empty_cache()
        gc.collect()

        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        # If sample_idxes is provided, use it
        if sample_idxes is None or len(sample_idxes) != num_motion_to_load:
            if not self.m_cfg.is_deterministic and random_sample:
                sample_idxes = torch.multinomial(
                    self._sampling_prob, num_samples=num_motion_to_load, replacement=True
                ).to(self._device)
            else:
                sample_idxes = torch.remainder(
                    torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions
                ).to(self._device)

        # import ipdb; ipdb.set_trace()
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(
            self._curr_motion_ids, num_classes=self._num_unique_motions
        ).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = (
            self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()
        )

        start_time = time.time()

        print("\n****************************** Current motion keys ******************************")
        print("Sampling motion:", sample_idxes[:30])
        if len(self.curr_motion_keys) < 100:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:30], ".....")
        print("*********************************************************************************\n")

        sample_idxes_np = sample_idxes.cpu().numpy()
        # object dtype: directory mode holds path strings, and a loaded dict must fit in the same array.
        motion_data_list = self._motion_data_list[sample_idxes_np].astype(object)
        for i, entry in enumerate(motion_data_list):
            if not isinstance(entry, dict) and osp.isfile(entry):
                motion_data_list[i] = joblib.load(entry)[entry.split("/")[-1].split(".")[0]]

        # One humanoid for every env: shared motions can share work; only the heading differs (applied below).
        assert all(st is skeleton_trees[0] for st in skeleton_trees), "expected one skeleton for all envs"
        assert (gender_betas == gender_betas[0]).all(), "expected one shared body shape for all envs"

        max_length = self.m_cfg.max_length
        is_deterministic = self.m_cfg.is_deterministic
        apply_heading = not (is_deterministic or self.m_cfg.im_eval)

        starts = np.zeros(num_motion_to_load, dtype=np.int64)
        ends = np.zeros(num_motion_to_load, dtype=np.int64)
        headings = np.zeros(num_motion_to_load, dtype=np.float64)
        for f in range(num_motion_to_load):
            seq_len = motion_data_list[f]["root_trans_offset"].shape[0]
            if max_length == -1 or seq_len < max_length:
                starts[f], ends[f] = 0, seq_len
            else:
                starts[f] = 0 if is_deterministic else random.randint(0, seq_len - max_length)
                ends[f] = starts[f] + max_length
            if apply_heading:
                headings[f] = np.pi * (2 * np.random.random() - 1.0)

        # De-dupe by motion id only; crops are handled per env below (random crops rarely repeat).
        group_of = np.empty(num_motion_to_load, dtype=np.int64)
        first_seen = {}
        rep_idxes = []
        for f in range(num_motion_to_load):
            key = int(sample_idxes_np[f])
            g = first_seen.get(key)
            if g is None:
                g = len(rep_idxes)
                first_seen[key] = g
                rep_idxes.append(f)
            group_of[f] = g
        num_groups = len(rep_idxes)

        mp.set_sharing_strategy("file_descriptor")

        rep_motion_data = motion_data_list[rep_idxes]
        rep_shape_params = gender_betas[rep_idxes]

        # Cache only the height curve (the expensive step); FK/velocity/dof_vels are cheap, so recomputed every call.
        cache_dir = None
        if self.mode == MotionlibMode.file and osp.isfile(self.m_cfg.motion_file):
            cache_dir = _motion_cache_dir(
                self.m_cfg.motion_file,
                self.fix_height,
                self.m_cfg.smpl_type,
                int(gender_betas[0, 0].item()),
                gender_betas[0, 1:].tolist(),
                self.m_cfg.max_length,
            )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:  # e.g. read-only dataset dir: run without a cache
                cache_dir = None
        rep_keys = self._motion_data_keys[sample_idxes_np[rep_idxes]]
        cache_paths = [_motion_cache_path(cache_dir, rep_keys[g]) if cache_dir else None for g in range(num_groups)]
        miss = [g for g in range(num_groups) if cache_paths[g] is None or not cache_paths[g].exists()]

        res_acc = {}  # using dictionary ensures order of the results.
        if miss:
            num_jobs = min(self.num_thread, len(miss))
            ids = np.array(miss)
            chunk = int(np.ceil(len(miss) / num_jobs))
            jobs = [
                (
                    ids[i : i + chunk],
                    rep_motion_data[miss][i : i + chunk],
                    rep_shape_params[miss][i : i + chunk],
                    self.mesh_parsers,
                    self.m_cfg,
                )
                for i in range(0, len(miss), chunk)
            ]
            queue = None
            if len(jobs) > 1:
                manager = mp.Manager()
                queue = manager.Queue()
                for i in range(1, len(jobs)):
                    worker = mp.Process(target=self.compute_height_curves, args=(*jobs[i], queue, i))
                    worker.start()
            res_acc.update(self.compute_height_curves(*jobs[0], None, 0))
            for _ in range(len(jobs) - 1):
                res_acc.update(queue.get())
            for g in miss:
                if cache_paths[g] is not None:
                    tmp_path = cache_paths[g].with_suffix(f".{os.getpid()}.tmp")
                    try:
                        torch.save(res_acc[g], tmp_path)
                        os.replace(tmp_path, cache_paths[g])  # atomic publish; avoids partial files on interrupt
                    except (OSError, RuntimeError):  # read-only cache dir: keep the computed result, skip caching
                        pass

        group_height_curve = [res_acc[g] if g in res_acc else torch.load(cache_paths[g]) for g in range(num_groups)]
        has_beta = np.array([("beta" in rep_motion_data[g]) for g in range(num_groups)])

        gts0_l, grs_l, lrs_l, fps_list, raw_num_frames = _batched_fk(rep_motion_data, skeleton_trees[0])
        group_fps = np.array(fps_list, dtype=np.float64)
        raw_num_frames = np.array(raw_num_frames, dtype=np.int64)

        group_bodies = rep_shape_params.clone().float()
        group_bodies[~torch.from_numpy(has_beta)] = 0.0
        self._motion_bodies = group_bodies[group_of].to(self._device)

        # Per-env, not per-group: env slots sharing a motion can still have different crop windows.
        env_num_frames = ends - starts
        env_dt = (1.0 / group_fps)[group_of]
        self._motion_fps = torch.tensor(group_fps[group_of], device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(env_dt, device=self._device, dtype=torch.float32)
        self._motion_lengths = torch.tensor((env_num_frames - 1) * env_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(env_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = num_motion_to_load

        group_length_starts = np.concatenate(([0], np.cumsum(raw_num_frames)[:-1]))
        env_length_starts = np.concatenate(([0], np.cumsum(env_num_frames)[:-1]))
        total_frames = int(env_num_frames.sum())

        owner_env = np.repeat(np.arange(num_motion_to_load), env_num_frames)
        frame_within_env = np.arange(total_frames) - env_length_starts[owner_env]
        # Offset by this env's own crop start into its group's full (uncropped) frame range.
        gather_idx = torch.from_numpy(group_length_starts[group_of[owner_env]] + starts[owner_env] + frame_within_env)

        # Per-env height correction: slice + min() of the cached curve instead of rerunning smplx.
        diff_fix = np.array(
            [
                group_height_curve[group_of[f]][starts[f] : starts[f] + min(30, ends[f] - starts[f])].min().item()
                for f in range(num_motion_to_load)
            ]
        )

        # Velocity/dof_vels need the actual crop window, so re-dedupe by (motion, crop) here.
        crop_of = np.empty(num_motion_to_load, dtype=np.int64)
        crop_first_seen, crop_rep = {}, []
        for f in range(num_motion_to_load):
            key = (int(sample_idxes_np[f]), int(starts[f]), int(ends[f]))
            c = crop_first_seen.get(key)
            if c is None:
                c = len(crop_rep)
                crop_first_seen[key] = c
                crop_rep.append(f)
            crop_of[f] = c
        gvs_l, gavs_l, dvs_l = [], [], []
        for f in crop_rep:
            u, s, e = group_of[f], starts[f], ends[f]
            fps = int(group_fps[u])  # compute_motion_dof_vels_jit is a TorchScript fn typed (Tensor, int)
            gvs_l.append(SkeletonMotion._compute_velocity(gts0_l[u][s:e], 1.0 / fps))
            gavs_l.append(SkeletonMotion._compute_angular_velocity(grs_l[u][s:e], 1.0 / fps))
            dvs_l.append(compute_motion_dof_vels_jit(lrs_l[u][s:e], fps))
        crop_num_frames = (ends - starts)[crop_rep]
        crop_length_starts = np.concatenate(([0], np.cumsum(crop_num_frames)[:-1]))
        crop_gather_idx = torch.from_numpy(crop_length_starts[crop_of[owner_env]] + frame_within_env)

        group_gts0 = torch.cat(gts0_l, dim=0)
        group_grs = torch.cat(grs_l, dim=0)
        group_lrs = torch.cat(lrs_l, dim=0)
        group_gvs = torch.cat(gvs_l, dim=0)
        group_gavs = torch.cat(gavs_l, dim=0)
        group_dvs = torch.cat(dvs_l, dim=0)

        # Per-env heading (identity when disabled); float64 regardless of poselib's own dtypes, cast to float32 after.
        heading_rot = sRot.from_euler(
            "xyz", np.stack([np.zeros(num_motion_to_load), np.zeros(num_motion_to_load), headings], axis=-1)
        )
        rot_mat_T = torch.tensor(heading_rot.as_matrix()).transpose(-1, -2)[owner_env]
        rot_quat = torch.tensor(heading_rot.as_quat())[owner_env]
        rot_quat = rot_quat.unsqueeze(1).expand(-1, self.num_joints, -1)

        gathered_gts = group_gts0[gather_idx].double()
        gathered_gts[..., 2] -= torch.from_numpy(diff_fix)[torch.from_numpy(owner_env)].unsqueeze(-1)

        self.gts = torch.matmul(gathered_gts, rot_mat_T).float().to(self._device)
        self.gvs = torch.matmul(group_gvs[crop_gather_idx].double(), rot_mat_T).float().to(self._device)
        self.gavs = torch.matmul(group_gavs[crop_gather_idx].double(), rot_mat_T).float().to(self._device)
        self.grs = torch_utils.quat_mul(rot_quat, group_grs[gather_idx].double()).float().to(self._device)
        self.lrs = group_lrs[gather_idx].float().to(self._device)
        # Root has no parent, so its "local" rotation is its global rotation; needs the heading too.
        self.lrs[:, 0] = self.grs[:, 0]
        self.dvs = group_dvs[crop_gather_idx].float().to(self._device)
        self.grvs = self.gvs[:, 0]
        self.gravs = self.gavs[:, 0]

        self.length_starts = torch.from_numpy(env_length_starts).to(self._device, dtype=torch.long)
        self.motion_ids = torch.arange(num_motion_to_load, dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(
            f"Loaded {num_motions:d} motions ({num_groups} unique) with a total length of "
            f"{total_len:.3f}s and {self.gts.shape[0]} frames."
        )
        print(f"Time to load motions: {int(time.time() - start_time)}s")

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    # def update_sampling_weight(self):
    #     ## sampling weight based on success rate.
    #     # sampling_temp = 0.2
    #     sampling_temp = 0.1
    #     curr_termination_prob = 0.5

    #     curr_succ_rate = 1 - self._termination_history[self._curr_motion_ids] / self._sampling_history[self._curr_motion_ids]
    #     self._success_rate[self._curr_motion_ids] = curr_succ_rate
    #     sample_prob = torch.exp(-self._success_rate / sampling_temp)

    #     self._sampling_prob = sample_prob / sample_prob.sum()
    #     self._termination_history[self._curr_motion_ids] = 0
    #     self._sampling_history[self._curr_motion_ids] = 0

    #     topk_sampled = self._sampling_prob.topk(50)
    #     print("Current most sampled", self._motion_data_keys[topk_sampled.indices.cpu().numpy()])

    def update_hard_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._sampling_prob[:] = 0
            self._sampling_prob[indexes] = 1 / len(indexes)
            print(
                "############################################################ Auto PMCP ############################################################"
            )
            print(f"Training on only {len(failed_keys)} seqs")
            print(failed_keys)
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = (
                torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
            )  # For use in sampling batches

    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)

            print(
                "############################################################ Auto PMCP ############################################################"
            )
            print(f"Training mostly on {len(self._sampling_prob.nonzero())} seqs ")
            print(self._motion_data_keys[self._sampling_prob.nonzero()].flatten())
            print(
                "###############################################################################################################################"
            )
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = (
                torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
            )  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history) and termination_history.sum() > 0:
            self._sampling_prob[:] = termination_history / termination_history.sum()
            self._termination_history = termination_history
            return True
        else:
            return False

    # def update_sampling_history(self, env_ids):
    #     self._sampling_history[self._curr_motion_ids[env_ids]] += 1
    #     # print("sampling history: ", self._sampling_history[self._curr_motion_ids])

    # def update_termination_history(self, termination):
    #     self._termination_history[self._curr_motion_ids] += termination
    #     # print("termination history: ", self._termination_history[self._curr_motion_ids])

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        # n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        # n = len(motion_ids)
        # num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [
            local_rot0,
            local_rot1,
            body_vel0,
            body_vel1,
            body_ang_vel0,
            body_ang_vel1,
            rg_pos0,
            rg_pos1,
            dof_vel0,
            dof_vel1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1

        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        }

    def get_root_pos_smpl(self, motion_ids, motion_times):
        # n = len(motion_ids)
        # num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        vals = [rg_pos0, rg_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].clone()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)  # clip blend to be within 0 and 1

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)


class MotionLibSMPL(MotionLibBase):
    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg=motion_lib_cfg)

        data_dir = str(ROOT_DIR / "smpl")
        if osp.exists(data_dir):
            if motion_lib_cfg.smpl_type == "smpl":
                # NOTE: SMPL model files must be present in the data_dir.
                # Download from https://smpl.is.tue.mpg.de/
                smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
                smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
                smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
            else:
                raise NotImplementedError(f"SMPL type {motion_lib_cfg.smpl_type} not implemented")

            # Height-fix SMPL forward runs on self._device; moved here once instead of per call.
            self.mesh_parsers = {
                0: smpl_parser_n.to(self._device),
                1: smpl_parser_m.to(self._device),
                2: smpl_parser_f.to(self._device),
            }

        else:
            print("SMPL models not found, set mesh_parsers to None")
            self.mesh_parsers = None

    # Frames per smplx forward() call; bounds peak memory regardless of how many motions are loaded.
    _HEIGHT_FIX_CHUNK_FRAMES = 4096

    @staticmethod
    def height_curve_batch(pose_aa_list, trans_list, gender, betas, mesh_parsers, fix_height_mode):
        # Per-frame min-vertex-height for every motion; a crop's height fix is curve[start:start+30].min().
        mesh_parser = mesh_parsers[gender]
        device = mesh_parser.v_template.device  # smplx forward runs wherever the parser lives
        betas = betas.to(device)
        counts = [t.shape[0] for t in trans_list]
        big_pose = torch.cat(pose_aa_list, dim=0)
        big_trans = torch.cat(trans_list, dim=0)

        pick = None
        if fix_height_mode == FixHeightMode.ankle_fix:
            assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
            pick = (
                (
                    (assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int()
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int()
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int()
                    + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()
                    == 4
                )
                .nonzero()
                .squeeze()
            )

        chunk = MotionLibSMPL._HEIGHT_FIX_CHUNK_FRAMES
        heights = []
        with torch.no_grad():
            for i in range(0, big_pose.shape[0], chunk):
                pose_c = big_pose[i : i + chunk].to(device)
                trans_c = big_trans[i : i + chunk].to(device)
                vertices, joints = mesh_parser.get_joints_verts(
                    pose_c, betas[None].expand(pose_c.shape[0], -1), trans_c
                )
                offset = joints[:, 0] - trans_c  # add back the SMPL root offset baked into trans_c
                verts = vertices[:, pick] if pick is not None else vertices
                heights.append((verts - offset[:, None])[..., -1].min(dim=-1).values.cpu())
        heights = torch.cat(heights)

        out, start = [], 0
        for c in counts:
            out.append(heights[start : start + c].clone())  # drop the shared-chunk storage before this is cached
            start += c
        return out

    @staticmethod
    def compute_height_curves(ids, motion_data_list, shape_params, mesh_parsers, config, queue, pid):
        # Only the height curve; FK/velocity/dof_vels are recomputed in load_motions either way.
        n = len(motion_data_list)
        assert len(ids) == n

        if mesh_parsers is None or config.fix_height == FixHeightMode.no_fix:
            curves = [torch.zeros(motion_data_list[f]["root_trans_offset"].shape[0]) for f in range(n)]
        else:
            gender = int(shape_params[0, 0].item())
            betas = shape_params[0, 1:]
            # No crop can look past seq_len - max_length + 30 (the latest start's calibration window).
            seq_lens = [motion_data_list[f]["root_trans_offset"].shape[0] for f in range(n)]
            needed = [
                sl if config.max_length == -1 or sl < config.max_length else sl - config.max_length + 30
                for sl in seq_lens
            ]
            trans_list = [motion_data_list[f]["root_trans_offset"][: needed[f]] for f in range(n)]
            pose_list = [to_torch(motion_data_list[f]["pose_aa"])[: needed[f]] for f in range(n)]
            curves = MotionLibSMPL.height_curve_batch(
                pose_list, trans_list, gender, betas, mesh_parsers, config.fix_height
            )

        res = {ids[f]: curves[f] for f in range(n)}
        if queue is not None:
            queue.put(res)
        else:
            return res
