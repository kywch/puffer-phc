# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy import ndimage as filters

from puffer_phc.torch_utils import (
    quat_identity_like,
    quat_mul_norm,
    quat_inverse,
    quat_angle_axis,
    transform_mul,
    transform_rotation,
    transform_translation,
    transform_from_rotation_translation,
)


class SkeletonTree:
    """A rigid skeleton: node names, parent indices (-1 is the root) and the local translation of each node."""

    def __init__(self, node_names, parent_indices, local_translation):
        """
        :param node_names: a list of names for each tree node
        :type node_names: List[str]
        :param parent_indices: an int32-typed tensor that represents the edge to its parent.\
        -1 represents the root node
        :type parent_indices: Tensor
        :param local_translation: a 3d vector that gives local translation information
        :type local_translation: Tensor
        """
        ln, lp, ll = len(node_names), len(parent_indices), len(local_translation)
        assert len(set((ln, lp, ll))) == 1
        self._node_names = node_names
        self._parent_indices = parent_indices.long()
        self._local_translation = local_translation
        self._node_indices = {self.node_names[i]: i for i in range(len(self))}

    def __len__(self):
        """number of nodes in the skeleton tree"""
        return len(self.node_names)

    @property
    def node_names(self):
        return self._node_names

    @property
    def parent_indices(self):
        return self._parent_indices

    @property
    def local_translation(self):
        return self._local_translation

    @property
    def num_joints(self):
        """number of nodes in the skeleton tree"""
        return len(self)

    @classmethod
    def from_mjcf(cls, path: str) -> "SkeletonTree":
        """
        Parses a mujoco xml scene description file and returns a Skeleton Tree.
        We use the model attribute at the root as the name of the tree.

        :param path:
        :type path: string
        :return: The skeleton tree constructed from the mjcf file
        :rtype: SkeletonTree
        """
        tree = ET.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        node_names = []
        parent_indices = []
        local_translation = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            curr_index = node_index
            node_index += 1
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index

        _add_xml_node(xml_body_root, -1, 0)

        return cls(
            node_names,
            torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            torch.from_numpy(np.array(local_translation, dtype=np.float32)),
        )


class SkeletonState:
    """Per-joint rotations (local or global) plus the root translation, packed in one tensor."""

    def __init__(self, tensor_backend, skeleton_tree, is_local):
        self._skeleton_tree = skeleton_tree
        self._is_local = is_local
        self.tensor = tensor_backend.clone()

    @property
    def rotation(self):
        if not hasattr(self, "_rotation"):
            self._rotation = self.tensor[..., : self.num_joints * 4].reshape(
                *(self.tensor.shape[:-1] + (self.num_joints, 4))
            )
        return self._rotation

    @property
    def _local_rotation(self):
        if self._is_local:
            return self.rotation
        else:
            return None

    @property
    def _global_rotation(self):
        if not self._is_local:
            return self.rotation
        else:
            return None

    @property
    def is_local(self):
        """is the rotation represented in local frame?

        :rtype: bool
        """
        return self._is_local

    @property
    def num_joints(self):
        """number of joints in the skeleton tree

        :rtype: int
        """
        return self.skeleton_tree.num_joints

    @property
    def skeleton_tree(self):
        """skeleton tree

        :rtype: SkeletonTree
        """
        return self._skeleton_tree

    @property
    def root_translation(self):
        """root translation

        :rtype: Tensor
        """
        if not hasattr(self, "_root_translation"):
            self._root_translation = self.tensor[..., self.num_joints * 4 : self.num_joints * 4 + 3]
        return self._root_translation

    @property
    def global_transformation(self):
        """global transformation of each joint (transform from joint frame to global frame)"""
        # Forward Kinematics
        if not hasattr(self, "_global_transformation"):
            local_transformation = self.local_transformation
            global_transformation = []
            parent_indices = self.skeleton_tree.parent_indices.numpy()
            # global_transformation = local_transformation.identity_like()
            for node_index in range(len(self.skeleton_tree)):
                parent_index = parent_indices[node_index]
                if parent_index == -1:
                    global_transformation.append(local_transformation[..., node_index, :])
                else:
                    global_transformation.append(
                        transform_mul(
                            global_transformation[parent_index],
                            local_transformation[..., node_index, :],
                        )
                    )
            self._global_transformation = torch.stack(global_transformation, axis=-2)
        return self._global_transformation

    @property
    def global_rotation(self):
        """global rotation of each joint (rotation matrix to rotate from joint's F.O.R to global
        F.O.R)"""
        if self._global_rotation is None:
            if not hasattr(self, "_comp_global_rotation"):
                self._comp_global_rotation = transform_rotation(self.global_transformation)
            return self._comp_global_rotation
        else:
            return self._global_rotation

    @property
    def global_translation(self):
        """global translation of each joint"""
        if not hasattr(self, "_global_translation"):
            self._global_translation = transform_translation(self.global_transformation)
        return self._global_translation

    @property
    def local_rotation(self):
        """the rotation from child frame to parent frame given in the order of child nodes appeared
        in `.skeleton_tree.node_names`"""
        if self._local_rotation is None:
            if not hasattr(self, "_comp_local_rotation"):
                local_rotation = quat_identity_like(self.global_rotation)
                for node_index in range(len(self.skeleton_tree)):
                    parent_index = self.skeleton_tree.parent_indices[node_index]
                    if parent_index == -1:
                        local_rotation[..., node_index, :] = self.global_rotation[..., node_index, :]
                    else:
                        local_rotation[..., node_index, :] = quat_mul_norm(
                            quat_inverse(self.global_rotation[..., parent_index, :]),
                            self.global_rotation[..., node_index, :],
                        )
                self._comp_local_rotation = local_rotation
            return self._comp_local_rotation
        else:
            return self._local_rotation

    @property
    def local_transformation(self):
        """local translation + local rotation. It describes the transformation from child frame to
        parent frame given in the order of child nodes appeared in `.skeleton_tree.node_names`"""
        if not hasattr(self, "_local_transformation"):
            self._local_transformation = transform_from_rotation_translation(
                r=self.local_rotation, t=self.local_translation
            )
        return self._local_transformation

    @property
    def local_translation(self):
        """local translation of the skeleton state. It is identical to the local translation in
        `.skeleton_tree.local_translation` except the root translation. The root translation is
        identical to `.root_translation`"""
        if not hasattr(self, "_local_translation"):
            broadcast_shape = (
                tuple(self.tensor.shape[:-1])
                + (len(self.skeleton_tree),)
                + tuple(self.skeleton_tree.local_translation.shape[-1:])
            )
            local_translation = self.skeleton_tree.local_translation.broadcast_to(*broadcast_shape).clone()
            local_translation[..., 0, :] = self.root_translation
            self._local_translation = local_translation
        return self._local_translation

    @staticmethod
    def _to_state_vector(rot, rt):
        state_shape = rot.shape[:-2]
        vr = rot.reshape(*(state_shape + (-1,)))
        vt = rt.broadcast_to(*state_shape + rt.shape[-1:]).reshape(*(state_shape + (-1,)))
        v = torch.cat([vr, vt], axis=-1)
        return v

    @classmethod
    def from_rotation_and_root_translation(cls, skeleton_tree, r, t, is_local=True):
        """
        Construct a skeleton state from rotation and root translation

        :param skeleton_tree: the skeleton tree
        :type skeleton_tree: SkeletonTree
        :param r: rotation (either global or local)
        :type r: Tensor
        :param t: root translation
        :type t: Tensor
        :param is_local: to indicate that whether the rotation is local or global
        :type is_local: bool, optional, default=True
        """
        assert r.dim() > 0, "the rotation needs to have at least 1 dimension (dim = {})".format(r.dim)
        state_vec = SkeletonState._to_state_vector(r, t)

        return cls(
            state_vec,
            skeleton_tree=skeleton_tree,
            is_local=is_local,
        )


class SkeletonMotion:
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)

        return velocity

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        diff_quat_data = quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = quat_mul_norm(r[..., 1:, :, :], quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(
                filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),
            )
        return angular_velocity
