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

import time
import hashlib
import os
import os.path as osp
from enum import Enum
from pathlib import Path

import joblib

import numpy as np

import torch
from tqdm import tqdm

from smpl_sim.smpllib.smpl_parser import SMPL_Parser

from puffer_phc import torch_utils, ROOT_DIR
from puffer_phc.poselib_skeleton import SkeletonMotion, SkeletonState


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


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


# Any change to the processing code invalidates every cache key.
_SOURCE = b"".join(
    (Path(__file__).parent / f).read_bytes() for f in ("motion_lib.py", "poselib_skeleton.py", "torch_utils.py")
)


def _motion_cache_dir(motion_file, fix_height, smpl_type, gender, betas):
    # Keyed on everything the height curve depends on, including the SMPL model file.
    smpl_file = ROOT_DIR / "smpl" / f"SMPL_{('neutral', 'male', 'female')[gender].upper()}.pkl"
    smpl_stat = (osp.getmtime(smpl_file), osp.getsize(smpl_file)) if osp.isfile(smpl_file) else None
    raw = repr(
        (
            hashlib.sha1(_SOURCE).hexdigest(),
            motion_file,
            osp.getmtime(motion_file),
            osp.getsize(motion_file),
            fix_height.value,
            smpl_type,
            gender,
            tuple(betas),
            str(smpl_file),
            smpl_stat,
        )
    )
    key = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return Path(f"{motion_file}.motion_cache") / key


def _motion_cache_path(cache_dir, motion_key):
    item = hashlib.sha1(motion_key.encode()).hexdigest()[:16]
    return cache_dir / f"{item}.pt"


# Vectorized across frames.
@torch.jit.script
def compute_motion_dof_vels_jit(local_rotation, fps):
    # type: (Tensor, int) -> Tensor
    dt = 1.0 / fps
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rotation[:-1]), local_rotation[1:])
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = (diff_axis * diff_angle.unsqueeze(-1) / dt)[:, 1:, :]
    return torch.cat([dof_vel, dof_vel[-1:]], dim=0)


_FK_CHUNK_FRAMES = 100_000


def _batched_fk(motion_data_list, skeleton_tree):
    # FK is per-frame independent, so batch over all motions; velocity/dof_vels are per motion.
    n = len(motion_data_list)
    trans_list = [d["root_trans_offset"] for d in motion_data_list]
    pose_quat_list = [to_torch(d["pose_quat_global"]) for d in motion_data_list]
    fps_list = [d.get("fps", 30) for d in motion_data_list]
    counts = [t.shape[0] for t in trans_list]

    rot, trans, outs = torch.cat(pose_quat_list), torch.cat(trans_list), []
    for i in tqdm(range(0, len(rot), _FK_CHUNK_FRAMES), desc="FK", unit="chunk"):  # chunked only for progress
        st = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree, rot[i : i + _FK_CHUNK_FRAMES], trans[i : i + _FK_CHUNK_FRAMES], is_local=False
        )
        outs.append((st.global_translation, st.global_rotation, st.local_rotation))
    big_gts, big_grs, big_lrs = (torch.cat(x) for x in zip(*outs))

    gts_l, grs_l, lrs_l, offset = [], [], [], 0
    for f in range(n):
        sl = slice(offset, offset + counts[f])
        gts_l.append(big_gts[sl])
        grs_l.append(big_grs[sl])
        lrs_l.append(big_lrs[sl])
        offset += counts[f]
    return gts_l, grs_l, lrs_l, fps_list, counts


def _yaw_rotate(h, v):
    """Rotates vectors v about z by the yaw quaternions h (xyzw with x = y = 0), in the xy plane."""
    c, s = 1 - 2 * h[..., 2:3] ** 2, 2 * h[..., 2:3] * h[..., 3:]
    return torch.cat([c * v[..., :1] - s * v[..., 1:2], s * v[..., :1] + c * v[..., 1:2], v[..., 2:]], -1)


class MotionLibBase:
    def __init__(self, motion_lib_cfg, skeleton_tree, gender_betas, limb_weights):
        """Processes every motion once (FK, velocities, height fix) into host tensors, one copy per motion."""
        self.m_cfg = cfg = motion_lib_cfg
        self._sim_fps = 1 / getattr(cfg, "step_dt", 1 / 30)  # CHECK ME: hardcoded
        self._device = cfg.device
        self._motion_limb_weights = limb_weights.float().to(self._device)  # per env
        start_time = time.time()

        gender, betas = int(gender_betas[0, 0].item()), gender_betas[0, 1:].tolist()
        cache_dir = _motion_cache_dir(cfg.motion_file, cfg.fix_height, cfg.smpl_type, gender, betas)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:  # e.g. read-only dataset dir: run without a cache
            cache_dir = None
        # The processed library also depends on min_length and the skeleton (as parsed from the MJCF).
        tree = (
            skeleton_tree.node_names,
            skeleton_tree.parent_indices.tolist(),
            skeleton_tree.local_translation.tolist(),
        )
        raw = repr((cfg.min_length, tree)).encode()
        lib_path = cache_dir and cache_dir.with_name(f"{cache_dir.name}-{hashlib.sha1(raw).hexdigest()[:16]}.lib.pt")
        state = None
        if lib_path:
            try:  # MAP_PRIVATE: writes never reach the file; a concurrent unlink after open is harmless
                state = torch.load(lib_path, mmap=True, weights_only=True)
                print(f"Motion cache hit: {lib_path}", flush=True)
            except (FileNotFoundError, RuntimeError):  # missing, unlinked mid-load, or corrupt: rebuild
                pass
        if state is None:
            print(f"Motion cache miss: processing {cfg.motion_file}", flush=True)
            state = self._process(cfg, skeleton_tree, gender_betas, cache_dir, start_time)
            if lib_path:
                tmp_path = lib_path.with_suffix(f".{os.getpid()}.tmp")
                try:
                    torch.save(state, tmp_path)
                    os.replace(tmp_path, lib_path)
                    for p in lib_path.parent.glob("*.lib.pt"):  # stale keys of this motion file
                        if p != lib_path:
                            p.unlink()
                except (OSError, RuntimeError):
                    pass
        self._host, self._bodies, self._motion_data_keys = state["host"], state["bodies"], np.array(state["keys"])
        self._frames, self._fps = state["frames"].numpy(), state["fps"].numpy()
        self._starts = np.concatenate(([0], np.cumsum(self._frames)[:-1]))
        self._num_unique_motions = M = len(self._frames)
        print(f"Processed {M} motions, {self._frames.sum()} frames, in {time.time() - start_time:.1f}s", flush=True)

    def _process(self, cfg, skeleton_tree, gender_betas, cache_dir, start_time):
        data = joblib.load(cfg.motion_file)
        data = {k: v for k, v in data.items() if len(v["pose_quat_global"]) >= cfg.min_length}
        motions, M = list(data.values()), len(data)
        print(f"Loaded {M} motions in {time.time() - start_time:.1f}s", flush=True)

        # Per-motion height curves are cached separately so a processed-library miss reuses them.
        cache_paths = [_motion_cache_path(cache_dir, k) if cache_dir else None for k in data]
        miss = [i for i in range(M) if cache_paths[i] is None or not cache_paths[i].exists()]
        curves = {}
        if miss:
            curves = self.compute_height_curves(miss, [motions[i] for i in miss], gender_betas, self.mesh_parsers, cfg)
            for i in miss:
                if cache_paths[i] is not None:
                    tmp_path = cache_paths[i].with_suffix(f".{os.getpid()}.tmp")
                    try:
                        torch.save(curves[i], tmp_path)
                        os.replace(tmp_path, cache_paths[i])  # atomic publish; avoids partial files on interrupt
                    except (OSError, RuntimeError):  # read-only cache dir: keep the computed result, skip caching
                        pass

        gts_l, grs_l, lrs_l, fps_list, counts = _batched_fk(motions, skeleton_tree)
        gvs_l, gavs_l, dvs_l = [], [], []
        for i, (gts, grs, lrs, fps) in enumerate(tqdm(zip(gts_l, grs_l, lrs_l, fps_list), total=M, desc="Velocities")):
            fps = int(fps)  # compute_motion_dof_vels_jit is a TorchScript fn typed (Tensor, int)
            gts[..., 2] -= (curves[i] if i in curves else torch.load(cache_paths[i])).min()  # one offset per clip
            gvs_l.append(SkeletonMotion._compute_velocity(gts, 1.0 / fps))
            gavs_l.append(SkeletonMotion._compute_angular_velocity(grs, 1.0 / fps))
            dvs_l.append(compute_motion_dof_vels_jit(lrs, fps))
        store = dict(gts=gts_l, grs=grs_l, lrs=lrs_l, gvs=gvs_l, gavs=gavs_l, dvs=dvs_l)
        has_beta = torch.tensor(["beta" in m for m in motions])
        return dict(
            host={k: torch.cat(v).float() for k, v in store.items()},
            frames=torch.tensor(counts),
            fps=torch.tensor(fps_list, dtype=torch.float64),
            keys=list(data),
            bodies=gender_betas[:1].float().repeat(M, 1) * has_beta[:, None],
        )

    def load_motions(self, ids):
        """ids: dataset motion id per env slot (a device tensor). Copies the unique motions' frames to the device."""
        start_time = time.time()
        self._curr_motion_ids = ids
        ids_np = ids.cpu().numpy()
        uniq, inv = np.unique(ids_np, return_inverse=True)
        frames = self._frames[uniq]
        idx = torch.from_numpy(np.concatenate([np.arange(s, s + n) for s, n in zip(self._starts[uniq], frames)]))
        for k, v in self._host.items():
            setattr(self, k, v[idx].to(self._device))

        dev, n = self._device, len(ids_np)
        self.length_starts = torch.from_numpy(np.concatenate(([0], np.cumsum(frames)[:-1]))[inv]).to(dev)
        self._motion_num_frames = torch.from_numpy(self._frames[ids_np]).to(dev)
        self._motion_fps = torch.tensor(self._fps[ids_np], device=dev, dtype=torch.float32)
        self._motion_dt = 1.0 / self._motion_fps
        self._motion_lengths = (self._motion_num_frames - 1) * self._motion_dt
        self._motion_bodies = self._bodies[ids_np].to(dev)
        # Per-env heading about z, applied at query time (xyzw quaternion); identity when deterministic or in eval
        yaw = (torch.rand(n, device=dev) * 2 - 1) * np.pi * (not (self.m_cfg.is_deterministic or self.m_cfg.im_eval))
        zero = torch.zeros_like(yaw)
        self._heading = torch.stack([zero, zero, (yaw / 2).sin(), (yaw / 2).cos()], -1)
        print(f"Loaded {n} motions ({len(uniq)} unique, {len(idx)} frames) in {time.time() - start_time:.1f}s")

    def get_motion_num_steps(self):
        return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
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
        heading = self._heading[motion_ids, None]

        rg_pos = _yaw_rotate(heading, (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1)
        if offset is not None:
            rg_pos = rg_pos + offset[..., None, :]  # ZL: apply offset

        body_vel = _yaw_rotate(heading, (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1)
        body_ang_vel = _yaw_rotate(heading, (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1)
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1

        # dof_pos drops the root, so the local rotations need no heading
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:]).reshape(local_rot.shape[0], -1)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.quat_mul(heading.expand_as(rb_rot0), torch_utils.slerp(rb_rot0, rb_rot1, blend_exp))

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
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
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
        return {"root_pos": _yaw_rotate(self._heading[motion_ids], rg_pos[..., 0, :])}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)  # clip blend to be within 0 and 1

        return frame_idx0, frame_idx1, blend


class MotionLibSMPL(MotionLibBase):
    def __init__(self, motion_lib_cfg, skeleton_tree, gender_betas, limb_weights):
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
                0: smpl_parser_n.to(motion_lib_cfg.device),
                1: smpl_parser_m.to(motion_lib_cfg.device),
                2: smpl_parser_f.to(motion_lib_cfg.device),
            }

        else:
            print("SMPL models not found, set mesh_parsers to None")
            self.mesh_parsers = None
        super().__init__(motion_lib_cfg, skeleton_tree, gender_betas, limb_weights)

    # Frames per smplx forward() call; bounds peak memory regardless of how many motions are loaded.
    _HEIGHT_FIX_CHUNK_FRAMES = 4096

    @staticmethod
    def height_curve_batch(pose_aa_list, trans_list, gender, betas, mesh_parsers, fix_height_mode):
        # Per-frame min-vertex-height for every motion.
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
            for i in tqdm(range(0, big_pose.shape[0], chunk), desc="Height curves", unit="chunk"):
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
    def compute_height_curves(ids, motion_data_list, shape_params, mesh_parsers, config):
        n = len(motion_data_list)
        assert len(ids) == n

        if mesh_parsers is None or config.fix_height == FixHeightMode.no_fix:
            curves = [torch.zeros(1) for f in range(n)]
        else:
            gender = int(shape_params[0, 0].item())
            betas = shape_params[0, 1:]
            # A clip's height offset is the min over its first 30 frames
            trans_list = [to_torch(motion_data_list[f]["root_trans_offset"])[:30] for f in range(n)]
            pose_list = [to_torch(motion_data_list[f]["pose_aa"])[:30] for f in range(n)]
            curves = MotionLibSMPL.height_curve_batch(
                pose_list, trans_list, gender, betas, mesh_parsers, config.fix_height
            )

        return {ids[f]: curves[f] for f in range(n)}
