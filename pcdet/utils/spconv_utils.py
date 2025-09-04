from typing import Set

import spconv

if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


def split_sp_tensor(x, coords=None):
    spatial_shape = x.spatial_shape
    voxel_indices = x.indices
    voxel_features = x.features

    spatial_indices = []
    num_voxels = []
    batch_size = x.batch_size
    batch_index = voxel_indices[:, 0]
    batch_feature = []

    for bs_idx in range(batch_size):
        batch_inds = batch_index == bs_idx
        if coords is None:
            if voxel_indices.shape[1] == 3:
                spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            else:
                spatial_indices.append(voxel_indices[batch_inds][:, [3, 2, 1]])
        else:
            spatial_indices.append(coords[batch_inds])
        batch_feature.append(voxel_features[batch_inds])
        num_voxels.append(batch_inds.sum())

    return spatial_shape, batch_index, batch_feature, spatial_indices, num_voxels


def get_knn_padding_feats_coords(x, coords=None):
    _, _, batch_feature, batch_indices, num_voxels = split_sp_tensor(x, coords)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32).to(x.features.device)
    # we will be padding each bs
    max_nums = torch.max(num_voxels, dim=0)[0]
    for i in range(len(batch_feature)):
        batch_feature[i] = F.pad(batch_feature[i], [0, 0, 0, max_nums - batch_feature[i].shape[0]])
        batch_indices[i] = F.pad(batch_indices[i], [0, 0, 0, max_nums - batch_indices[i].shape[0]], value=1e4)

    padding_feats = torch.stack(batch_feature)
    padding_coords = torch.stack(batch_indices)

    return padding_feats, padding_coords


def get_mirror_padding_feats_coords(x, coords=None):
    _, _, batch_feature, batch_indices, num_voxels = split_sp_tensor(x, coords)
    num_voxels = torch.stack(num_voxels).to(dtype=torch.int32, device=x.features.device)

    max_nums = torch.max(num_voxels, dim=0)[0]

    for i in range(len(batch_feature)):
        mirror_feats = torch.flip(batch_feature[i], [0])
        mirror_coords = torch.flip(batch_indices[i], [0])

        while batch_feature[i].size(0) < max_nums:
            batch_feature[i] = torch.cat(
                [batch_feature[i], mirror_feats[1: max_nums - batch_feature[i].size(0) + 1]], dim=0
            )
            batch_indices[i] = torch.cat(
                [batch_indices[i], mirror_coords[1: max_nums - batch_indices[i].size(0) + 1]], dim=0
            )
            mirror_feats = torch.flip(mirror_feats, [0])
            mirror_coords = torch.flip(mirror_coords, [0])

    padding_feats = torch.stack(batch_feature)
    padding_coords = torch.stack(batch_indices)

    return padding_feats, padding_coords


def get_zeros_padding_feats_coords(x, coords=None):
    spatial_shape, _, batch_feature, batch_indices, num_voxels = split_sp_tensor(x, coords)
    num_voxels = torch.stack(num_voxels).to(dtype=torch.int32, device=x.features.device)

    max_nums = torch.max(num_voxels, dim=0)[0]
    mask = torch.arange(max_nums, device=max_nums.device).unsqueeze(0) >= num_voxels.unsqueeze(1)

    padding_feats = pad_sequence(batch_feature, batch_first=True, padding_value=0)
    padding_coords = pad_sequence(batch_indices, batch_first=True, padding_value=-1)

    return padding_feats, padding_coords, num_voxels, mask


def tensor2spconv(feats, coords, spatial_shape, voxel_num, mask):
    """Forward Function of MultiScaleDeformAttention.
    Args:
        feats (torch.Tensor): pytorch tensor with shape
            (bs, max_voxels, c)
        coords (torch.Tensor): spconv coords with shape
            (bs, max_voxels, 2[x, y] or 3 [bs, x, y] or 4 [bs, x, y, z])
        spatial_shape (list): The original size of the spconv tensor with shape
            (bs, )
        voxel_num (torch.Tensor): The number of non-empty voxels per bs, per level with shape
            (bs, num_lvl)
        mask (torch.Tensor): The mask for each feature is used to
            take out the tensor after each padding with shape
            (bs, max_voxels)
    Returns:
        spconv.SparseConvTensor
            list: len() = num_level.
    """
    bs, _, dim = coords.shape
    bs, levels = voxel_num.shape

    # TODO: Support dim=3, dim=4
    assert dim == 2 or dim == 3, "Only 2D coordinates are supported."

    # 预处理：合并所有批次的特征和坐标，按层级划分
    all_feats = [None] * levels
    all_coords = [None] * levels

    # 扁平化所有 batch 数据
    valid_mask = ~mask  # (bs, max_voxels)
    valid_feats = [feats[b][valid_mask[b]] for b in range(bs)]
    valid_coords = [coords[b][valid_mask[b]] for b in range(bs)]

    # 拆分每个 batch 的特征和坐标（按层级）
    split_feats = [feat.split(voxel_num[b].tolist()) for b, feat in enumerate(valid_feats)]
    split_coords = [coord.split(voxel_num[b].tolist()) for b, coord in enumerate(valid_coords)]

    # 按层级合并所有 batch 数据
    for lvl in range(levels):
        lvl_feats = [split_feats[b][lvl] for b in range(bs)]

        if dim == 2:
            lvl_coords = [
                torch.cat([
                    torch.full((split_coords[b][lvl].shape[0], 1), b, device=coords.device, dtype=coords.dtype),
                    split_coords[b][lvl][:, [1, 0]]  # (x, y) -> (y, x)
                ], dim=1) for b in range(bs)
            ]
        elif dim == 3:
            lvl_coords = [
                torch.cat([
                    torch.full((split_coords[b][lvl].shape[0], 1), b, device=coords.device, dtype=coords.dtype),
                    split_coords[b][lvl][:, [2, 1, 0]]  # (x, y, z) -> (z, y, x)
                ], dim=1) for b in range(bs)
            ]
        else:
            raise NotImplementedError

        all_feats[lvl] = torch.cat(lvl_feats, dim=0)
        all_coords[lvl] = torch.cat(lvl_coords, dim=0)

    # 创建 SparseConvTensor（单层循环）
    spconv_tensors = [
        spconv.SparseConvTensor(
            features=all_feats[lvl],
            indices=all_coords[lvl],
            spatial_shape=spatial_shape[lvl],
            batch_size=bs
        ) for lvl in range(levels)
    ]

    return spconv_tensors