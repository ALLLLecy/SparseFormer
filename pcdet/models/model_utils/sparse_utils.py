# import copy
# from functools import partial
# import math
# from typing import Optional, Union, List

# from timm.models.layers import DropPath
# import torch
# import torch.nn as nn
# from torch import Tensor
# from pcdet.utils.spconv_utils import spconv, replace_feature, tensor2spconv, get_zeros_padding_feats_coords, \
#     split_sp_tensor
# from pcdet.ops.deformable_attn.ms_deform_attn_func import MSDeformAttnFunction
# import torch.nn.functional as F
# import torch_scatter
# from mamba_ssm.models.mixer_seq_simple import create_block
# from pytorch3d.ops import knn_points
# import torch.utils.checkpoint as cp
# from ...utils.serialization import FlattenWindowsSerialization, HilbertSerialization, ZOrderSerialization


# class ResidualSparseBasicBlock2D(spconv.SparseModule):
#     expansion = 1

#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, norm_fn=None, indice_key=None):
#         super(ResidualSparseBasicBlock2D, self).__init__()

#         assert norm_fn is not None
#         bias = norm_fn is not None
#         self.conv1 = spconv.SubMConv2d(
#             inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias,
#             indice_key=indice_key
#         )
#         self.bn1 = norm_fn(planes)
#         self.act = nn.GELU()

#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = replace_feature(out, self.bn1(out.features))
#         out = replace_feature(out, out.features + identity.features)
#         out = replace_feature(out, self.act(out.features))

#         return out


# class SparseBasicBlock2D(spconv.SparseModule):
#     expansion = 1

#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, norm_fn=None, indice_key=None):
#         super(SparseBasicBlock2D, self).__init__()

#         assert norm_fn is not None
#         bias = norm_fn is not None
#         self.conv1 = spconv.SubMConv2d(
#             inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias,
#             indice_key=indice_key
#         )
#         self.bn1 = norm_fn(planes)
#         self.relu = nn.ReLU()
#         self.conv2 = spconv.SubMConv2d(
#             planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias,
#             indice_key=indice_key
#         )

#         self.stride = stride

#     def forward(self, x):
#         out = self.conv1(x)
#         out = replace_feature(out, self.bn1(out.features))
#         out = replace_feature(out, self.relu(out.features))

#         out = self.conv2(out)

#         return out


# class ConvEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """

#     def __init__(self, input_channel, num_pos_feats=288):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
#             nn.BatchNorm1d(num_pos_feats),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

#     def forward(self, xyz):
#         xyz = xyz.transpose(1, 2).contiguous()
#         position_embedding = self.stem(xyz.float())
#         return position_embedding.transpose(1, 2).contiguous()


# class LinearEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """

#     def __init__(self, input_channel, num_pos_feats=288):
#         super().__init__()
#         self.windows_embedding_head = nn.Sequential(
#             nn.Linear(input_channel, num_pos_feats),
#             nn.LayerNorm(num_pos_feats),
#             nn.ReLU(inplace=True),
#             nn.Linear(num_pos_feats, num_pos_feats),
#         )

#     def forward(self, xyz):
#         position_embedding = self.windows_embedding_head(xyz.float())
#         return position_embedding


# class MSSubConvEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """

#     def __init__(self, in_channel, num_levels=3, num_pos_feats=288):
#         super().__init__()
#         self.in_channel = in_channel
#         self.embed_channels = num_pos_feats
#         norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=True)

#         self.stem = nn.ModuleList()
#         for i in range(num_levels):
#             self.stem.append(
#                 SparseBasicBlock2D(
#                     inplanes=in_channel, planes=num_pos_feats,
#                     kernel_size=3, stride=1,
#                     norm_fn=norm_layer,
#                     indice_key=f'pos_embed_{i}'
#                 )
#             )

#     def forward(self, x):
#         y = []
#         for P, stem in zip(x, self.stem):
#             E = spconv.SparseConvTensor(
#                 features=P.indices[:, 1:].float(),
#                 indices=P.indices,
#                 spatial_shape=P.spatial_shape,
#                 batch_size=P.batch_size
#             )
#             E = stem(E)
#             P = replace_feature(P, P.features + E.features)
#             y.append(P)

#         return y


# class SPDecoder(nn.Module):
#     def __init__(
#             self,
#             model_cfg,
#             input_channel,
#             cross_only=False,
#             activation="relu",
#     ):
#         super().__init__()
#         # cross cfg
#         self.window_shape = model_cfg.WINDOW_SHAPE
#         self.depth = model_cfg.DEPTH
#         self.num_levels = 1

#         dim_feedforward = model_cfg.FFN_DIM
#         dropout = model_cfg.DROPOUT

#         self.cross_only = cross_only
#         if not self.cross_only:
#             self.self_attn = nn.MultiheadAttention(
#                 input_channel,
#                 model_cfg.NUM_HEADS,
#                 dropout=dropout,
#                 batch_first=True
#             )
#         self.cross_attn = SDCA(
#             model_cfg=model_cfg.SDCA,
#             input_channel=input_channel,
#             depth=self.depth,
#             window_shape=self.window_shape
#         )

#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(input_channel, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, input_channel)

#         self.norm1 = nn.LayerNorm(input_channel)
#         self.norm2 = nn.LayerNorm(input_channel)
#         self.norm3 = nn.LayerNorm(input_channel)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         def _get_activation_fn(activation):
#             """Return an activation function given a string"""
#             if activation == "relu":
#                 return F.relu
#             if activation == "gelu":
#                 return F.gelu
#             if activation == "glu":
#                 return F.glu
#             raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

#         self.activation = _get_activation_fn(activation)
#         self.self_posembed = ConvEmbeddingLearned(
#             input_channel=len(self.window_shape),
#             num_pos_feats=input_channel
#         )
#         self.cross_posembed = MSSubConvEmbeddingLearned(
#             in_channel=len(self.window_shape),
#             num_levels=self.num_levels,
#             num_pos_feats=input_channel
#         )

#         # self.fusion = nn.MultiheadAttention(input_channel, model_cfg.NUM_HEADS, batch_first=True)

#     def forward(
#             self,
#             query,
#             key,
#             query_coords,
#             key_padding_mask=None,
#             **kwargs
#     ):
#         if self.self_posembed is not None:
#             query_pos_embed = self.self_posembed(query_coords)
#             query = query + query_pos_embed

#         if self.cross_posembed is not None:
#             key = self.cross_posembed([key])[0]

#         if not self.cross_only:
#             q = k = v = query
#             query2 = self.self_attn(q, k, value=v)[0]
#             query = query + self.dropout1(query2)
#             query = self.norm1(query)

#         query2 = self.cross_attn(
#             query=query,
#             key=key,
#             value=key,
#             query_pos=None,
#             query_coords=query_coords
#         )

#         # for i in range(self.num_levels):
#         #     query2.append(
#         #         self.cross_attn(
#         #             query=query,
#         #             key=key,
#         #             value=key,
#         #             query_pos=None,
#         #             query_coords=query_coords
#         #         )
#         #     )
#         # query2 = torch.stcak(query2, dim=1)
#         # query2 = self.fusion(query2)[0]
#         # query2 = query2.max(dim=1)

#         query = query + self.dropout2(query2)
#         query = self.norm2(query)

#         query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
#         query = query + self.dropout3(query2)
#         query = self.norm3(query)

#         return query.permute(0, 2, 1).contiguous()


# class SDCA(nn.Module):
#     def __init__(
#             self,
#             model_cfg,
#             input_channel,
#             depth,
#             window_shape
#     ):
#         super().__init__()

#         # input layers
#         self.window_shape = window_shape
#         self.orders = model_cfg.ORDERS
#         self.shifts_rate = model_cfg.get("SHIFTS_RATE", 0.0)

#         # DCA
#         self.input_channel = input_channel
#         self.embed_dim = model_cfg.EMBED_DIM
#         self.num_heads = model_cfg.NUM_HEADS
#         self.num_points = model_cfg.NUM_POINTS
#         self.dropout = model_cfg.DROPOUT

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.input_layer = SerializationLayer(
#             window_shape=window_shape,
#             orders=self.orders,
#             depth=depth,
#             shifts_rate=self.shifts_rate
#         )

#         self.blocks = nn.ModuleList()
#         for i in range(len(self.orders)):
#             self.blocks.append(
#                 DeformableAttention(
#                     input_channel=self.input_channel,
#                     embed_dim=self.embed_dim,
#                     num_heads=self.num_heads,
#                     num_points=self.num_points,
#                     dropout=self.dropout
#                 )
#             )

#     def convert_query_coords(self, query_coords):
#         bs, token, _ = query_coords.shape
#         bs_indices = torch.arange(bs, device=query_coords.device).view(-1, 1).repeat(1, token)
#         x = query_coords[..., 0]
#         y = query_coords[..., 1]
#         return torch.stack((bs_indices, y, x), dim=-1).view(-1, 3)

#     def forward(
#             self,
#             query,
#             key,
#             value,
#             query_pos,
#             query_coords,
#             key_padding_mask=None,
#             **kwargs
#     ):
#         bs, token, C = query.shape

#         L = max([
#             (value.indices[:, 0] == b).sum().item() for b in range(value.batch_size)
#         ])
#         points = torch.linspace(0, 1, L, dtype=torch.float32, device=self.device)
#         mappings = self.input_layer(value.indices, value.batch_size, L, value.spatial_shape)
#         mask = mappings["mask"].view(-1, L)  # [bs, L]

#         # 多个 block 循环
#         for i, block in enumerate(self.blocks):
#             inds = mappings[self.orders[i]]

#             if key is not None and value is not None and torch.equal(query if isinstance(query, Tensor) else query.features, key.features) and torch.equal(key.features, value.features):
#                 refer_points = points[None, :].expand(query.shape[0], -1)[None].unsqueeze(-1)  # [bs, L, 1]
#                 q = query.features[inds][mappings["flat2win"]]
#                 q = q.view(-1, L, q.shape[-1])

#                 k = v = q
#                 q = block(
#                     query=q,
#                     key=k,
#                     value=v,
#                     identity=None,
#                     query_pos=query_pos,
#                     key_padding_mask=mask,
#                     reference_points=refer_points,
#                     spatial_shape=[L],
#                     level_start_index=[0]
#                 )
#                 query.features[inds] = q.view(-1, q.shape[-1])[mappings["win2flat"]]

#             else:
#                 coords = value.indices[inds][mappings["flat2win"]].view(-1, L, value.indices.shape[-1])  # [bs, L, 3]
#                 coords[mask] = -1  # 对无效点置为 -1
#                 coords = coords[..., [2, 1]]

#                 dists, knn_inds, _ = knn_points(query_coords.float(), coords.float(), K=1)
#                 assert (dists == 0).all(), "Each query coord must exactly match a key coord!"
#                 refer_points = torch.gather(points.expand(bs, L), dim=1, index=knn_inds.squeeze(-1))  # [bs, token]
#                 refer_points = refer_points[:, :, None, None] # [bs, token, 1, 1]

#                 k = key.features[inds][mappings["flat2win"]]
#                 k = k.view(-1, L, k.shape[-1])
#                 v = value.features[inds][mappings["flat2win"]]
#                 v = v.view(-1, L, v.shape[-1])

#                 query = block(
#                     query=query,
#                     key=k,
#                     value=v,
#                     identity=None,
#                     query_pos=query_pos,
#                     key_padding_mask=mask,
#                     reference_points=refer_points,
#                     spatial_shapes=torch.as_tensor([L], dtype=torch.long, device=query.device).unsqueeze(-1),
#                     level_start_index=torch.as_tensor([0], dtype=torch.long, device=query.device)
#                 )

#         return query


# class SerializationLayer(nn.Module):
#     def __init__(
#             self,
#             window_shape,
#             orders=["z", "z-trans"],
#             depth=8,
#             shifts_rate=0.0
#     ):
#         super().__init__()
#         self.window_shape = window_shape
#         self.depth = depth
#         self.orders = orders
#         self.shifts_rate = 0.0

#         # serialization
#         if "z" in orders or "z-trans" in orders:
#             self.z_order_serialization = ZOrderSerialization(window_shape=self.window_shape, depth=self.depth)
#         if "x" in orders or "y" in orders:
#             self.windows_serialization = FlattenWindowsSerialization(window_shape=self.window_shape, win_version='v3')
#         if "xx" in orders or "yy" in orders or "xy" in orders or "yx" in orders:
#             self.windows_serialization = FlattenWindowsSerialization(window_shape=self.window_shape, win_version='v3e')
#         if "hilbert" in orders or "hilbert-trans" in orders:
#             self.hilbert_serialization = HilbertSerialization(window_shape=self.window_shape, depth=self.depth)

#     def forward(
#             self,
#             coords,
#             batch_size,
#             max_voxels,
#             sparse_shape
#     ):
#         shifts = [True if self.shifts_rate > torch.rand(1) else False for _ in range(len(self.orders))]
#         orders = self.orders

#         coords = coords.long()
#         _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
#         batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))

#         # 每个 batch 补齐到 max_voxels
#         num_per_batch_p = torch.full((batch_size,), max_voxels, dtype=torch.long, device=coords.device)

#         batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))

#         total_padded = batch_start_indices_p[-1]
#         flat2win = torch.empty(total_padded, dtype=torch.long, device=coords.device)
#         win2flat = torch.arange(batch_start_indices[-1], device=coords.device)

#         # mask（False: real points, True: padding points）
#         mask = torch.ones(total_padded, dtype=torch.bool, device=coords.device)

#         for i in range(batch_size):
#             start = batch_start_indices[i]
#             end = batch_start_indices[i + 1]
#             padded_start = batch_start_indices_p[i]
#             padded_end = batch_start_indices_p[i + 1]

#             num_real = end - start
#             num_padded = padded_end - padded_start

#             # 把真实点填入 flat2win
#             flat2win[padded_start: padded_start + num_real] = win2flat[start:end]

#             # 设置真实点 mask 为 True
#             mask[padded_start: padded_start + num_real] = False

#             if num_real < num_padded:
#                 num_missing = num_padded - num_real

#                 # 构造镜像 padding 索引（例如: 0,1,2,3 → 2,1,0,1,2,3,...）
#                 reflect_base = torch.cat([
#                     torch.arange(num_real - 2, -1, -1, device=coords.device),
#                     torch.arange(1, num_real, device=coords.device)
#                 ])
#                 # 重复并裁剪到 num_missing 长度
#                 repeat_count = (num_missing + reflect_base.numel() - 1) // reflect_base.numel()
#                 mirror = reflect_base.repeat(repeat_count)[:num_missing]

#                 # 取真实点映射过来的值做映射
#                 mirror_extended = win2flat[start:end][mirror]

#                 # 填入 flat2win 的 padding 区域
#                 flat2win[padded_start + num_real: padded_end] = mirror_extended

#             # 修改索引
#             win2flat[start:end] += (padded_start - start)
#             # flat2win[padded_start:padded_end] -= (padded_start - start)

#         # 输出 mappings
#         mappings = {
#             "flat2win": flat2win,
#             "win2flat": win2flat,
#             "mask": mask
#         }

#         for order, shift in zip(orders, shifts):
#             mappings.update(self.encoder(coords, sparse_shape, shift, order))

#         return mappings

#     def encoder(self, coords, sparse_shape, shifts, order):
#         """
#         :param coords: [bs, y, x]
#         :return:
#         """
#         meta = {}
#         n, ndim = coords.shape
#         #                   z order                hilbert                 flatten windows
#         assert order in {"z", "z-trans", "hilbert", "hilbert-trans", "x", "y", "xy", "yx", "xx", "yy"}
#         if order == "z":
#             coords2curve = self.z_order_serialization(
#                 coords, sparse_shape, shifts
#             )
#         elif order == "z-trans":
#             coords2curve = self.z_order_serialization(
#                 coords[..., [1, 0, 2]] if ndim == 3 else coords[..., [1, 0]],
#                 sparse_shape, shifts
#             )
#         elif order == "hilbert":
#             coords2curve = self.hilbert_serialization(
#                 coords, sparse_shape, shifts
#             )
#         elif order == "hilbert-trans":
#             coords2curve = self.hilbert_serialization(
#                 coords[..., [1, 0, 2]] if ndim == 3 else coords[..., [1, 0]],
#                 sparse_shape, shifts
#             )
#         elif order == "xy" or order == "x":
#             coords2curve = self.windows_serialization(
#                 coords, sparse_shape, shifts, mapping_name=order
#             )
#         elif order == "yx" or order == "y":
#             coords2curve = self.windows_serialization(
#                 coords, sparse_shape, shifts, mapping_name=order
#             )
#         elif order == "xx":
#             coords2curve = self.windows_serialization(
#                 coords, sparse_shape, shifts, mapping_name=order
#             )
#         elif order == "yy":
#             coords2curve = self.windows_serialization(
#                 coords, sparse_shape, shifts, mapping_name=order
#             )
#         else:
#             raise NotImplementedError

#         meta[order] = coords2curve

#         return meta


# class DeformableAttention(nn.Module):
#     def __init__(
#             self,
#             input_channel=256,
#             embed_dim=256,
#             num_heads=8,
#             num_points=32,
#             dropout=0.1,
#             im2col_step=64,
#     ):
#         super().__init__()
#         self.in_channel = input_channel

#         # attn
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_points = num_points
#         self.im2col_step = im2col_step

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # offset layer
#         self.sampling_offsets = nn.Linear(self.embed_dim, self.num_heads * self.num_points)
#         self.sampling_weights = nn.Linear(self.embed_dim, self.num_heads * self.num_points)

#         # proj layer
#         self.value_proj = nn.Linear(self.in_channel, self.embed_dim)
#         self.output_proj = nn.Linear(self.embed_dim, self.in_channel)

#         self.dropout = nn.Dropout(dropout)
#         self.init_weights()

#     def init_weights(self):
#         """Default initialization for Parameters of Module."""

#         def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
#             # If the module has a weight and the weight is not None, initialize the weight to a constant
#             if hasattr(module, 'weight') and module.weight is not None:
#                 nn.init.constant_(module.weight, val)
#             # If the module has a bias and the bias is not None, initialize the bias to a constant
#             if hasattr(module, 'bias') and module.bias is not None:
#                 nn.init.constant_(module.bias, bias)

#         def xavier_init(module: nn.Module,
#                         gain: float = 1,
#                         bias: float = 0,
#                         distribution: str = 'normal') -> None:
#             assert distribution in ['uniform', 'normal']
#             if hasattr(module, 'weight') and module.weight is not None:
#                 if distribution == 'uniform':
#                     nn.init.xavier_uniform_(module.weight, gain=gain)
#                 else:
#                     nn.init.xavier_normal_(module.weight, gain=gain)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 nn.init.constant_(module.bias, bias)

#         thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
#         grid_init_base = torch.stack([thetas.cos()], -1)  # (num_heads, 1)
#         constant_init(self.sampling_offsets, val=0.)

#         grid_init = (grid_init_base / grid_init_base.abs().max(-1, keepdim=True)[0]).view(
#             self.num_heads, 1, 1, 1).repeat(1, 1, self.num_points, 1)
#         for head_index in range(self.num_heads):
#             scale_factor = (head_index % (self.num_heads // 2) + 1)  # 计算缩放因子
#             for i in range(self.num_points):
#                 grid_init[head_index, :, i, :] *= scale_factor * (i + 1)
#         self.sampling_offsets.bias.data = grid_init.view(-1)

#         constant_init(self.sampling_weights, val=0., bias=0.)
#         xavier_init(self.value_proj, distribution='uniform', bias=0.)
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)

#     def forward(
#             self,
#             query: torch.Tensor,
#             key: Optional[torch.Tensor] = None,
#             value: Optional[torch.Tensor] = None,
#             identity: Optional[torch.Tensor] = None,
#             query_pos: Optional[torch.Tensor] = None,
#             key_padding_mask: Optional[torch.Tensor] = None,
#             reference_points: Optional[torch.Tensor] = None,
#             spatial_shapes: Optional[torch.Tensor] = None,
#             level_start_index: Optional[torch.Tensor] = None,
#             **kwargs,
#     ):
#         """Forward Function of MultiScaleDeformAttention.

#          Args:
#             query (torch.Tensor): Query of Transformer with shape
#                  (bs, num_query, embed_dims).
#             value (torch.Tensor | None): The value tensor with shape
#                  `(bs, num_key, embed_dims)`.
#             identity (torch.Tensor): The tensor used for addition, with the
#                  same shape as `query`. Default None. If None,
#                  `query` will be used.
#             query_pos (torch.Tensor): The pospatial_shapessitional encoding for `query`.
#                  Default: None.
#             key_padding_mask (torch.Tensor): ByteTensor for `query`, with
#                  shape [bs, num_key].
#             reference_points (torch.Tensor):  The normalized reference
#                  points with shape (bs, num_query, num_levels, 1),
#                  all elements is range in [0, 1], top-left (0,0),
#                  bottom-right (1, 1), including padding area.
#                  or (N, Length_{query}, num_levels, 4), add
#                  additional two dimensions is (w, h) to
#                  form reference boxes.
#             spatial_shapes (torch.Tensor): Spatial shape of features in
#                  each different levels. With shape (num_levels, 2),
#                  last dimension represents (h, w).
#             level_start_index (torch.Tensor): It is used to indicate
#                  the starting position of each scale feature
#                  for cuda and code calculation
#          Returns:
#              torch.Tensor: forwarded results with shape
#              [bs, num_query, embed_dims].
#          """

#         # self attn
#         if key is not None and value is not None and torch.equal(query, key) and torch.equal(key, value):
#             temp_key = temp_value = query

#             query = self._forward(
#                 query=query,
#                 key=temp_key,
#                 value=temp_value,
#                 query_pos=query_pos,
#                 key_padding_mask=key_padding_mask,
#                 pos_padding_mask=None,
#                 reference_points=reference_points,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#             )
#             return query

#         # cross attn
#         else:
#             query = self._forward(
#                 query=query,
#                 key=key,
#                 value=value,
#                 query_pos=query_pos,
#                 key_padding_mask=key_padding_mask,
#                 pos_padding_mask=None,
#                 reference_points=reference_points,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index
#             )
#             return query

#     def _forward(self,
#                  query: torch.Tensor,
#                  key: Optional[torch.Tensor] = None,
#                  value: Optional[torch.Tensor] = None,
#                  identity: Optional[torch.Tensor] = None,
#                  query_pos: Optional[torch.Tensor] = None,
#                  key_padding_mask: Optional[torch.Tensor] = None,
#                  pos_padding_mask: Optional[torch.Tensor] = None,
#                  reference_points: Optional[torch.Tensor] = None,
#                  spatial_shapes: Optional[torch.Tensor] = None,
#                  level_start_index: Optional[torch.Tensor] = None,
#                  **kwargs
#                  ):
#         """Forward Function of MultiScaleDeformAttention.

#          Args:
#              query (torch.Tensor): Query of Transformer with shape
#                  (bs, num_query, embed_dims).
#              value (torch.Tensor | None): The value tensor with shape
#                  `(bs, num_key, embed_dims)`.
#              identity (torch.Tensor): The tensor used for addition, with the
#                  same shape as `query`. Default None. If None,
#                  `query` will be used.
#              query_pos (torch.Tensor): The pospatial_shapessitional encoding for `query`.
#                  Default: None.
#              key_padding_mask (torch.Tensor): ByteTensor for `query`, with
#                  shape [bs, num_key].
#              reference_points (torch.Tensor):  The normalized reference
#                  points with shape (bs, num_query, num_levels, 1),
#                  all elements is range in [0, 1], top-left (0,0),
#                  bottom-right (1, 1), including padding area.
#                  or (N, Length_{query}, num_levels, 4), add
#                  additional two dimensions is (w, h) to
#                  form reference boxes.
#              spatial_shapes (torch.Tensor): Spatial shape of features in
#                  different levels. With shape (num_levels),
#                  last dimension represents (l).
#          Returns:
#              torch.Tensor: forwarded results with shape
#              [num_query, bs, embed_dims].
#          """
#         if value is None:
#             value = query

#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             if pos_padding_mask is not None:
#                 query = query + query_pos.masked_fill(pos_padding_mask[..., None], 0.0)
#             else:
#                 query = query + query_pos

#         bs, num_query, _ = query.shape
#         bs, num_value, _ = value.shape
#         assert spatial_shapes.sum() == num_value

#         # value: [bs, num_value, embed_dims]
#         value = self.value_proj(value)
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], 0.0)

#         value = value.view(bs, num_value, self.num_heads, -1)

#         # (bs, num_query, num_heads, sum(num_points), 1)
#         sampling_offsets = self.sampling_offsets(query)
#         # (bs, num_query, num_heads, num_levels=1, num_points_l, 1)
#         sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, 1, self.num_points, 1)

#         # (bs, num_query, num_heads, num_levels=1, num_points_l)
#         sampling_weights = self.sampling_weights(query)
#         sampling_weights = sampling_weights.view(bs, num_query, self.num_heads, 1, self.num_points)
#         sampling_weights = F.softmax(sampling_weights, dim=-1)

#         sampling_locations = (reference_points[:, :, None, :, None, :]
#                               + sampling_offsets
#                               / spatial_shapes[None, None, None, :, None, :])

#         if torch.cuda.is_available() and value.is_cuda:
#             output = MSDeformAttnFunction.apply(
#                 value, spatial_shapes, level_start_index, sampling_locations,
#                 sampling_weights, self.im2col_step
#             )
#         else:
#             output = deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, sampling_weights
#             )

#         output = self.output_proj(output)

#         return self.dropout(output) + identity


# def deformable_attn_pytorch(
#         value: torch.Tensor, value_spatial_shapes: torch.Tensor,
#         sampling_locations: torch.Tensor,
#         sampling_weights: torch.Tensor) -> torch.Tensor:
#     """CPU version of multi-scale deformable attention.

#     Args:
#         value (torch.Tensor): The value has shape
#             (bs, num_keys, embed_dims)
#         value_spatial_shapes (torch.Tensor): Spatial shape of
#             each feature map, has shape (1,),
#             last dimension 1 represent (L)
#         sampling_locations (torch.Tensor): The location of sampling points,
#             has shape
#             (bs ,num_queries, num_points, 1),
#             the last dimension 1 represent (x).
#         sampling_weights (torch.Tensor): The weight of sampling points used
#             when calculate the attention, has shape
#             (bs ,num_queries, num_points),

#     Returns:
#         torch.Tensor: has shape (bs, num_queries, embed_dims)
#     """

#     bs, _, num_heads, embed_dims = value.shape
#     _, num_queries, num_heads, num_levels, num_points, _ = \
#         sampling_locations.shape
#     value_list = value.split([L_ for L_ in value_spatial_shapes], dim=1)
#     # sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for level, L_ in enumerate(value_spatial_shapes):
#         # bs, L_, num_heads, embed_dims ->
#         # bs, L_, num_heads*embed_dims ->
#         # bs, num_heads*embed_dims, L_ ->
#         # bs*num_heads, embed_dims, L_
#         value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
#             bs * num_heads, embed_dims, L_
#         )
#         # bs, num_queries, num_heads, num_points, 1 ->
#         # bs, num_heads, num_queries, num_points, 1 ->
#         # bs*num_heads, num_queries, num_points, 1
#         sampling_grid_l_ = sampling_locations[:, :, :, level].transpose(1, 2).flatten(0, 1)
#         sampling_grid_l_ = sampling_grid_l_ * L_ - 1

#         # bs*num_heads, embed_dims, num_queries, num_points
#         sampling_value_l_ = grid_sample(value_l_, sampling_grid_l_, padding_mode='zeros')
#         sampling_value_list.append(sampling_value_l_)
#     # (bs, num_queries, num_heads, num_levels, num_points) ->
#     # (bs, num_heads, num_queries, num_levels, num_points) ->
#     # (bs*num_heads, 1, num_queries, num_levels*num_points)
#     attention_weights = sampling_weights.transpose(1, 2).reshape(
#         bs * num_heads, 1, num_queries, num_levels * num_points)
#     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
#               attention_weights).sum(-1).view(bs, num_heads * embed_dims,
#                                               num_queries)
#     return output.transpose(1, 2).contiguous()


# def grid_sample(value: torch.Tensor, sampling_grid: torch.Tensor, padding_mode: str = 'zeros') -> torch.Tensor:
#     """
#     PyTorch 版本的 1D 线性插值函数，模拟 grid_sample 接口。

#     Args:
#         value (torch.Tensor): 输入数据，形状为 (batch_size * num_heads, embed_dims, length)
#         sampling_grid (torch.Tensor): 采样点位置，形状为 (batch_size * num_heads, num_queries, num_points, 1)
#         padding_mode (str): 边界处理方式，默认为 'zeros'，可选 'border'

#     Returns:
#         torch.Tensor: 插值结果，形状为 (batch_size * num_heads, embed_dims, num_queries, num_points)
#     """
#     batch_heads, embed_dims, length = value.shape
#     _, _, num_points, _ = sampling_grid.shape

#     # 计算 x_low 和 x_high
#     # here may cause a bug:
#     # x = [1.0, 2.0, 3.0, 4.0]
#     # x = torch.floor(x)
#     # x = [1.0, 2.0, 3.0, 3.0]
#     x = sampling_grid.squeeze(-1)  # 去掉最后一维
#     x_low = torch.floor(x).long()
#     x_high = x_low + 1

#     lx = x - x_low.float()
#     hx = 1.0 - lx

#     # 处理边界
#     if padding_mode == 'border':
#         x_low = torch.clamp(x_low, 0, length - 1)
#         x_high = torch.clamp(x_high, 0, length - 1)
#     else:  # zeros padding
#         mask_low = (x_low >= 0) & (x_low < length)
#         mask_high = (x_high >= 0) & (x_high < length)
#         x_low = torch.clamp(x_low, 0, length - 1)
#         x_high = torch.clamp(x_high, 0, length - 1)

#     # 索引数据
#     x_low_expanded = x_low.reshape(batch_heads, -1).unsqueeze(-2).expand(-1, embed_dims, -1)
#     x_high_expanded = x_high.reshape(batch_heads, -1).unsqueeze(-2).expand(-1, embed_dims, -1)

#     v1 = torch.gather(value, -1, index=x_low_expanded).reshape(batch_heads, embed_dims, -1, num_points)
#     v2 = torch.gather(value, -1, index=x_high_expanded).reshape(batch_heads, embed_dims, -1, num_points)

#     # 处理 padding_mode='zeros' 的无效点
#     if padding_mode == 'zeros':
#         v1 = v1 * mask_low.unsqueeze(1).float()
#         v2 = v2 * mask_high.unsqueeze(1).float()

#     # 线性插值
#     interpolated = hx.unsq
import copy
from functools import partial
import math
import random
from typing import Optional, Union, List

from timm.models.layers import DropPath
import torch
import torch.nn as nn
from torch import Tensor
from pcdet.utils.spconv_utils import spconv, replace_feature, tensor2spconv, get_zeros_padding_feats_coords, \
    split_sp_tensor
from pcdet.ops.deformable_attn.ms_deform_attn_func import MSDeformAttnFunction
import torch.nn.functional as F
import torch_scatter
from mamba_ssm.models.mixer_seq_simple import create_block
from pytorch3d.ops import knn_points
import torch.utils.checkpoint as cp
from ...utils.serialization import FlattenWindowsSerialization, HilbertSerialization, ZOrderSerialization

class ResidualSparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, dim=2, kernel_size=3, stride=1, norm_fn=None, indice_key=None):
        super().__init__()
        assert norm_fn is not None
        assert dim in [2, 3], f"Unsupported dim={dim}, only 2 or 3 allowed."

        self.dim = dim
        self.stride = stride
        bias = norm_fn is not None

        if dim == 2:
            self.conv1 = spconv.SubMConv2d(
                inplanes, planes, kernel_size=kernel_size, stride=stride,
                padding=kernel_size // 2, bias=bias, indice_key=indice_key
            )
        elif dim == 3:
            self.conv1 = spconv.SubMConv3d(
                inplanes, planes, kernel_size=kernel_size, stride=stride,
                padding=kernel_size // 2, bias=bias, indice_key=indice_key
            )

        self.bn1 = norm_fn(planes)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act(out.features))
        return out



class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, dim=2, kernel_size=3, stride=1, norm_fn=None, indice_key=None):
        """
        dim: 2 或 3，决定用 2D 还是 3D sparse conv
        """
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None

        if dim == 2:
            Conv = spconv.SubMConv2d
        elif dim == 3:
            Conv = spconv.SubMConv3d
        else:
            raise ValueError(f"Unsupported dim={dim}, only 2 or 3 allowed.")

        self.conv1 = Conv(
            inplanes, planes, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv(
            planes, planes, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=bias, indice_key=indice_key
        )

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        return out


class ConvEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用 Kaiming 初始化适用于 ReLU 激活
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.stem(xyz.float())
        return position_embedding.transpose(1, 2).contiguous()


class LinearEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.windows_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self.init_weights()
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        position_embedding = self.windows_embedding_head(xyz.float())
        return position_embedding


class MSSubConvEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, in_channel, dim=2, num_levels=3, num_pos_feats=288):
        super().__init__()
        self.in_channel = in_channel
        self.embed_channels = num_pos_feats
        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=True)

        self.stem = nn.ModuleList()
        for i in range(num_levels):
            self.stem.append(
                SparseBasicBlock(
                    inplanes=in_channel, planes=num_pos_feats,
                    dim=dim,
                    kernel_size=3, stride=1,
                    norm_fn=norm_layer,
                    indice_key=f'pos_embed_{i}'
                )
            )
        self.init_weights()


    def get_pos_embed(self, spatial_shape, coors, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)

        return location

    def forward(self, x):
        y = []
        for P, stem in zip(x, self.stem):
            localtion = self.get_pos_embed(spatial_shape=P.spatial_shape, coors=P.indices[:, 1:])
            E = spconv.SparseConvTensor(
                features=localtion,
                indices=P.indices,
                spatial_shape=P.spatial_shape,
                batch_size=P.batch_size
            )
            E = stem(E)
            P = replace_feature(P, P.features + E.features)
            y.append(P)

        return y

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SPEncoder(nn.Module):
    def __init__(
            self,
            model_cfg,
            input_channel
    ):
        super().__init__()
        self.num_layers = model_cfg.NUM_LAYERS
        self.orders = model_cfg.ORDERS
        self.dim = model_cfg.get("DIM", 2)
        self.depth = model_cfg.DEPTH
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.shuffle = model_cfg.get("SHUFFLE", False)
        assert self.num_layers == len(self.orders)

        self.blocks = nn.ModuleList()
        self.inputs = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(
                SMSA(
                    model_cfg.SMSA,
                    input_channel=input_channel,
                    layer_idx=i,
                    orders=self.orders[i],
                    dim=self.dim,
                    num_layer=self.num_layers
                )
            )
        
            self.inputs.append(
                SerializationLayer(
                    window_shape=self.window_shape,
                    orders=self.orders[i],
                    depth=self.depth
                )
            )

        self.embeddings = nn.ModuleList()
        for i in range(self.num_layers):
            self.embeddings.append(
                MSSubConvEmbeddingLearned(
                    in_channel=self.dim,
                    dim=self.dim,
                    num_pos_feats=input_channel
                )
            )

    def forward(self, x):
        shuffle_order = list(range(self.num_layers))
        if self.shuffle:
            random.shuffle(shuffle_order)
    
        for i in range(self.num_layers):
            input = self.inputs[shuffle_order[i]]
            x = self.embeddings[i](x)
            x = self.blocks[i](x, input, orders=input.orders)
        return x


class SMSA(nn.Module):
    def __init__(
            self,
            model_cfg,
            input_channel,
            orders,
            dim=2,
            layer_idx=0,
            num_layer=3
    ):
        super().__init__()

        # attn
        self.input_channel = input_channel

        # other cfg
        self.num_levels = model_cfg.NUM_LEVELS
        self.depth = model_cfg.DEPTH
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.spatial_enhance = model_cfg.SPATIAL_ENHANCE
        self.layer_idx = layer_idx
        self.orders = orders
        self.dim = dim

        # diff cfg
        self.diff_coef = model_cfg.DIFF_COEF
        self.DIFF_KERNEL = model_cfg.DIFF_KERNEL

        # mamba cfg
        norm_epsilon = model_cfg.NORM_EPSILON
        rms_norm = model_cfg.RMS_NORM
        residual_in_fp32 = model_cfg.RESIDUAL_IN_FP32
        fused_add_norm = model_cfg.FUSED_ADD_NORM

        factory_kwargs = {'device': "cuda", 'dtype': torch.float32}
        forward_blocks = []
        backward_blocks = []

        idx = layer_idx * len(orders) * 2
        for i in range(len(orders)):
            forward_blocks.append(
                create_block(
                    d_model=input_channel,
                    ssm_cfg=None,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i + idx,
                    **factory_kwargs,
                )
            )

        idx = idx + len(orders)
        for i in range(len(orders)):
            backward_blocks.append(
                create_block(
                    d_model=input_channel,
                    ssm_cfg=None,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i + idx,
                    **factory_kwargs,
                )
            )
        self.num_layers = num_layer
        
        self.forward_blocks = nn.ModuleList(forward_blocks)
        self.backward_blocks = nn.ModuleList(backward_blocks)

        # layer norm
        self.forward_norms = nn.ModuleList(
            [nn.LayerNorm(self.input_channel) for _ in range(len(orders))]
        )
        self.backward_norms = nn.ModuleList(
            [nn.LayerNorm(self.input_channel) for _ in range(len(orders))]
        )
        self.output_norms = nn.ModuleList(
            [nn.LayerNorm(self.input_channel) for _ in range(len(orders))]
        )

        self.drop_path = DropPath(model_cfg.DROP_PATH)

        if self.spatial_enhance:
            self.locals = nn.ModuleList()
            # for _ in range(len(orders)):
            #     lc = nn.ModuleList()
            for l in range(self.num_levels):
                self.locals.append(
                    ResidualSparseBasicBlock(
                        self.input_channel, self.input_channel,
                        dim=self.dim,
                        norm_fn=partial(nn.LayerNorm),
                        indice_key=f'layer_{layer_idx}_out_{l}'
                    )
                )
                # self.locals.append(lc)
        
        if self.diff_coef != 0.0 and self.layer_idx < self.num_layers - 1:
            self.diffs = nn.ModuleList()
            for l in range(self.num_levels):
                self.diffs.append(
                    SparseBasicBlock(
                        inplanes=self.input_channel, planes=self.input_channel,
                        dim=self.dim,
                        kernel_size=3, stride=1,
                        norm_fn=partial(nn.LayerNorm),
                        indice_key=f"layer_{layer_idx}_diff_{l}"
                    )
                )

        # self.input_layer = SerializationLayer(
        #     window_shape=self.window_shape,
        #     orders=self.orders,
        #     depth=self.depth
        # )

    def feature_diffusion(self, mx):
        out = []

        for i in range(len(mx)):
            x = mx[i]
            kernel_size = self.diff_kernel[i]
            _, _, batch_feature, spatial_indices, num_voxels = split_sp_tensor(x)
            selected_indices_list = []
            for bs in range(x.batch_size):
                num_voxel = num_voxels[bs]
                K = int(num_voxel * self.diff_coef)  # 0.1
                _, indices = torch.topk(batch_feature[bs].mean(dim=1), K)
                selected_indices = spatial_indices[bs][indices][:, [1, 0]]
                padding_bs = torch.ones(selected_indices.shape[0], 1, device=selected_indices.device) * bs
                selected_indices = torch.cat([padding_bs.int(), selected_indices], dim=-1)
                selected_indices_list.append(selected_indices)

            selected_indices = torch.cat(selected_indices_list, dim=0)
            one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            one_mask = F.max_pool2d(one_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            zero_indices = (one_mask[:, 0] > 0).nonzero().int()
            zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))

            cat_indices = torch.cat([x.indices, zero_indices], dim=0)
            cat_features = torch.cat([x.features, zero_features], dim=0)
            indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
            features_unique = torch_scatter.scatter_add(cat_features, _inv, dim=0, dim_size=indices_unique.shape[0])

            # key, value
            out.append(
                self.diffs[i](
                    spconv.SparseConvTensor(
                        features=features_unique,
                        indices=indices_unique,
                        spatial_shape=x.spatial_shape,
                        batch_size=x.batch_size
                    )
                )
            )

        return out

    def forward(self, x, input_layer, orders):
        assert len(x) == self.num_levels
        batch_size = x[0].batch_size

        def scale_indices(tensor, scale):
            return torch.cat([tensor.indices[:, :1], tensor.indices[:, 1:] * scale], dim=1)

        def restore_sparse_tensor(feat, coords, level):
            return spconv.SparseConvTensor(
                features=feat,
                indices=coords,
                spatial_shape=spatial_shape[level],
                batch_size=batch_size
            )

        # === Step 1: Flatten multi-level features and coords ===
        scales = [1, 2, 4]
        spatial_shape = [lvl.spatial_shape for lvl in x]
        features = [lvl.features for lvl in x]
        coords = [scale_indices(lvl, s) for lvl, s in zip(x, scales)]

        F = torch.cat(features, dim=0)
        C = torch.cat(coords, dim=0)

        device = F.device
        lens = torch.tensor([len(f) for f in features], device=device)
        starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(lens, dim=0)[:-1]])
        ends = torch.cumsum(lens, dim=0)

        # === Step 2: Build mapping for free group attention ===
        L = torch.bincount(C[:, 0]).max().item()
        mappings = input_layer(C, batch_size, L, spatial_shape[0])

        # === Step 3: Iterative SSM blocks ===
        for i, (fb, bb, fn, bn, on) in enumerate(zip(
                self.forward_blocks, self.backward_blocks,
                self.forward_norms, self.backward_norms,
                self.output_norms)):

            inds = mappings[orders[i]]

            # Reshape coords to (B, L, C) for attention
            coords_win = C[inds][mappings["flat2win"]].view(-1, L, C.shape[1])
            expected_batch = torch.arange(coords_win.shape[0], device=device).view(-1, 1).expand(-1, L)
            assert torch.all(coords_win[:, :, 0] == expected_batch), "coords batch mismatch"

            # === Forward Attention ===
            fb_feats = F[inds][mappings["flat2win"]].view(-1, L, F.shape[-1])
            fb_out = fb(fb_feats, None)[0].view(-1, F.shape[-1])[mappings["win2flat"]]
            feat_m1 = fn(torch.zeros_like(F).index_copy_(0, inds, fb_out))

            # === Backward Attention ===
            bb_feats = fb_feats.flip(1)
            bb_out = bb(bb_feats, None)[0].flip(1).view(-1, F.shape[-1])[mappings["win2flat"]]
            feat_m2 = bn(torch.zeros_like(F).index_copy_(0, inds, bb_out))

            # === Update F ===
            F = on(F + self.drop_path(feat_m1) + self.drop_path(feat_m2))

            # # === Optional Spatial Enhancement ===
            # if self.spatial_enhance:
            #     new_x = []
            #     for lvl in range(self.num_levels):
            #         start, end = starts[lvl], ends[lvl]
            #         scale = scales[lvl]

            #         lvl_feat = F[start:end]
            #         lvl_coords = torch.cat(
            #             [C[start:end][:, :1], C[start:end][:, 1:] // scale], dim=1
            #         )

            #         sparse_tensor = restore_sparse_tensor(lvl_feat, lvl_coords, lvl)
            #         enhanced = lc[lvl](sparse_tensor)
            #         new_x.append(enhanced)

            #     # Rebuild flattened features and coords
            #     x = new_x
            #     features = [lvl.features for lvl in x]
            #     coords = [scale_indices(lvl, s) for lvl, s in zip(x, scales)]

            #     F = torch.cat(features, dim=0)
            #     TC = torch.cat(coords, dim=0)

            #     assert torch.all(C == TC), "coords mismatch after enhancement"
            #     C = TC

        # === Optional Spatial Enhancement ===
        if self.spatial_enhance:
            new_x = []
            for lvl in range(self.num_levels):
                start, end = starts[lvl], ends[lvl]
                scale = scales[lvl]

                lvl_feat = F[start:end]
                lvl_coords = torch.cat(
                    [C[start:end][:, :1], C[start:end][:, 1:] // scale], dim=1
                )

                sparse_tensor = restore_sparse_tensor(lvl_feat, lvl_coords, lvl)
                enhanced = self.locals[lvl](sparse_tensor)
                new_x.append(enhanced)

            x = new_x
        else:
            # 不做 spatial enhancement，也要把 F 写回 sparse tensor
            new_x = []
            for lvl in range(self.num_levels):
                start, end = starts[lvl], ends[lvl]
                scale = scales[lvl]

                lvl_feat = F[start:end]
                lvl_coords = torch.cat(
                    [C[start:end][:, :1], C[start:end][:, 1:] // scale], dim=1
                )

                sparse_tensor = restore_sparse_tensor(lvl_feat, lvl_coords, lvl)
                new_x.append(sparse_tensor)

            x = new_x
                     

        # === Step 4: Optional diffusion layer ===
        if self.layer_idx < self.num_layers - 1 and self.diff_coef != 0.0:
            x = self.feature_diffusion(x)

        return x


class SPDecoder(nn.Module):
    def __init__(
            self,
            model_cfg,
            input_channel,
            cross_only=False,
            activation="relu",
    ):
        super().__init__()
        # cross cfg
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.depth = model_cfg.DEPTH
        self.num_levels = model_cfg.NUM_LEVELS
        self.num_points = model_cfg.NUM_POINTS
        self.num_layers = model_cfg.NUM_LAYERS
        self.orders = model_cfg.ORDERS

        assert self.num_levels == len(self.num_points) == len(self.depth) == len(self.window_shape)
        assert self.num_layers == len(self.orders)

        dim_feedforward = model_cfg.FFN_DIM
        dropout = model_cfg.DROPOUT

        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(
                input_channel,
                model_cfg.NUM_HEADS,
                dropout=dropout,
                batch_first=True
            )

        self.cross_attn = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for i in range(self.num_layers):
            blocks = nn.ModuleList()
            for j in range(self.num_levels):
                blocks.append(
                    SDCA(
                        model_cfg=model_cfg.SDCA,
                        input_channel=input_channel,
                        depth=self.depth[j],
                        window_shape=self.window_shape[j],
                        num_points=self.num_points[j],
                        orders=self.orders[i],
                        idx=j
                    )
                )
            self.cross_attn.append(blocks)
            self.fusions.append(LevelFusion(input_channel, num_levels=self.num_levels, num_heads=model_cfg.FUSION_HEADS))

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(input_channel, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_channel)

        self.norm1 = nn.LayerNorm(input_channel)
        self.norm2 = nn.LayerNorm(input_channel)
        self.norm3 = nn.LayerNorm(input_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)
        self.self_posembed = ConvEmbeddingLearned(
            input_channel=2,
            num_pos_feats=input_channel
        )
        self.cross_posembed = MSSubConvEmbeddingLearned(
            in_channel=2,
            num_levels=self.num_levels,
            num_pos_feats=input_channel
        )

        self.init_weights()

    def init_weights(self):
        # 只初始化 decoder 自己的 Linear
        nn.init.trunc_normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.trunc_normal_(self.linear2.weight, std=0.02)
        nn.init.constant_(self.linear2.bias, 0)

        # LayerNorm
        for norm in [self.norm1, self.norm2, self.norm3]:
            nn.init.constant_(norm.bias, 0)
            nn.init.constant_(norm.weight, 1.0)

        # MultiheadAttention
        if not self.cross_only:
            nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
            if self.self_attn.in_proj_bias is not None:
                nn.init.constant_(self.self_attn.in_proj_bias, 0)
            nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
            if self.self_attn.out_proj.bias is not None:
                nn.init.constant_(self.self_attn.out_proj.bias, 0)

    def forward(
            self,
            query,
            key,
            query_coords,
            key_padding_mask=None,
            **kwargs
    ):
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_coords)
            query = query + query_pos_embed

        if self.cross_posembed is not None:
            key = self.cross_posembed(key)

        if not self.cross_only:
            q = k = v = query
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = query
        for l in range(self.num_layers):
            out = []
            blocks = self.cross_attn[l]
            fusion = self.fusions[l]
            for i in range(self.num_levels):
                out.append(
                    blocks[i](
                        query=query2,
                        key=key[i],
                        value=key[i],
                        query_pos=None,
                        query_coords=query_coords
                    )
                )
            out = torch.stack(out, dim=1)
            out = fusion(out)
            query2 = out

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query.permute(0, 2, 1).contiguous()



class SDCA(nn.Module):
    def __init__(
            self,
            model_cfg,
            input_channel,
            depth,
            window_shape,
            num_points,
            orders,
            idx,
    ):
        super().__init__()

        self.idx = idx
        # input layers
        self.window_shape = window_shape
        self.orders = orders
        self.shifts_rate = model_cfg.get("SHIFTS_RATE", 0.0)

        # DCA
        self.input_channel = input_channel
        self.embed_dim = model_cfg.EMBED_DIM
        self.num_heads = model_cfg.NUM_HEADS
        self.num_points = num_points
        self.dropout = model_cfg.DROPOUT

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_layer = SerializationLayer(
            window_shape=window_shape,
            orders=self.orders,
            depth=depth,
            shifts_rate=self.shifts_rate
        )

        self.blocks = nn.ModuleList()
        for i in range(len(self.orders)):
            self.blocks.append(
                DeformableAttention(
                    input_channel=self.input_channel,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_points=self.num_points,
                    dropout=self.dropout
                )
            )

    def convert_query_coords(self, query_coords):
        bs, token, _ = query_coords.shape
        bs = torch.arange(bs, device=query_coords.device)[:, None].repeat(1, token)
        x = query_coords[..., 0]
        y = query_coords[..., 1]
        return torch.stack((bs, y, x), dim=-1).view(-1, 3)


    def input_mapping_layer(
            self,
            Cv,
            Cq,
            batch_size,
            spatial_shape
    ):
        mappings = {}
        Cq = self.convert_query_coords(Cq)
        mappings["Cq"] = Cq

        dists, knn_idx, _ = knn_points(Cq[None].float(), Cv[None].float(), K=1)
        dists = dists.squeeze()  # [N]

        match_threshold = 1e-6
        match_any = (dists < match_threshold)
        non_match_mask = ~match_any
        if non_match_mask.any():
            Cq = Cq[non_match_mask]
            Cp = torch.cat([Cv, Cq], dim=0)
            Lp = max([
                (Cp[:, 0] == b).sum().item() for b in range(batch_size)
            ])
            mask_value = torch.cat(
                [torch.ones (Cv.size(0), dtype=torch.bool, device=Cv.device),
                torch.zeros(Cq.size(0), dtype=torch.bool, device=Cp.device)], dim=0
            )

            mappings.update(self.input_layer(Cp, batch_size, Lp, spatial_shape))
            mappings["mask_value"] = mask_value
            mappings["non_match_mask"] = non_match_mask
            mappings["Cp"] = Cp

        return mappings

    def get_reference_points(
            self,
            sorted_value_coords,
            query_coords,
            is_value,
            sort_idx_all: torch.Tensor,
            query_ref: torch.Tensor,
            all_coords: torch.Tensor,
            non_match_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = query_ref.device

        N = sorted_value_coords.size(0)
        all_coords_sorted = all_coords[sort_idx_all]
        is_value_sorted = is_value[sort_idx_all]

        value_positions = torch.where(is_value_sorted)[0]
        query_positions = torch.where(~is_value_sorted)[0]

        value_seq_full = torch.empty(all_coords.size(0), dtype=torch.float, device=device)
        value_seq_full[value_positions] = torch.arange(value_positions.size(0), dtype=torch.float, device=device)

        insert_pos = torch.searchsorted(value_positions, query_positions, right=False)
        left_idx = torch.clamp(insert_pos - 1, 0, value_positions.size(0) - 1)
        right_idx = torch.clamp(insert_pos, 0, value_positions.size(0) - 1)

        left_pos = value_positions[left_idx]
        right_pos = value_positions[right_idx]

        left_seq = value_seq_full[left_pos]
        right_seq = value_seq_full[right_pos]

        denom = (right_pos.float() - left_pos.float()).clamp(min=1e-6)
        weight = (query_positions.float() - left_pos.float()) / denom
        interp_ref = left_seq + weight * (right_seq - left_seq)

        # === 新增边界限制，避免跨 batch 插值 ===
        sorted_bs = sorted_value_coords[:, 0].long()
        unique_bs, counts = torch.unique(sorted_bs, return_counts=True)
        bs_start = torch.zeros_like(counts)
        bs_start[1:] = torch.cumsum(counts[:-1], dim=0)
        bs_end = bs_start + counts - 1

        bs_range_map = dict(zip(unique_bs.tolist(), zip(bs_start.tolist(), bs_end.tolist())))
        query_batches = all_coords_sorted[query_positions][:, 0].long()
        batch_starts = torch.tensor([bs_range_map[int(b.item())][0] for b in query_batches], device=device)
        batch_ends = torch.tensor([bs_range_map[int(b.item())][1] for b in query_batches], device=device)

        interp_ref = torch.min(torch.max(interp_ref, batch_starts.float()), batch_ends.float())
        # === 边界限制结束 ===

        orig_query_indices = sort_idx_all[query_positions] - N
        interp_ref_unsorted = torch.empty_like(interp_ref)
        interp_ref_unsorted[orig_query_indices] = interp_ref

        query_ref[non_match_mask] = interp_ref_unsorted

        # Step 5: 转换为 batch 内部的 ref 索引
        sorted_bs = sorted_value_coords[:, 0].long()
        unique_bs, counts = torch.unique(sorted_bs, return_counts=True)
        bs_start = torch.zeros_like(counts)
        bs_start[1:] = torch.cumsum(counts[:-1], dim=0)
        bs_offset_map = dict(zip(unique_bs.tolist(), bs_start.tolist()))
        offset_per_query = torch.tensor(
            [bs_offset_map[int(b.item())] for b in query_coords[:, 0]],
            device=query_ref.device
        )
        query_ref_local = query_ref - offset_per_query

        return query_ref_local


    def forward(
            self,
            query,
            key,
            value,
            query_pos,
            query_coords,
            key_padding_mask=None,
            **kwargs
    ):
        bs, M, _ = query_coords.shape
        # 0. get original mappings
        query_coords = query_coords // (2 ** self.idx)
        L = max([
            (value.indices[:, 0] == b).sum().item() for b in range(value.batch_size)
        ])
        mappings = self.input_layer(value.indices, value.batch_size, L, value.spatial_shape)
        mask = mappings["mask"].view(-1, L)  # [bs, L]

        # 1. get padding mappings
        if query_coords is not None:
            padding_mappings = self.input_mapping_layer(
                value.indices,
                query_coords,
                value.batch_size,
                value.spatial_shape
            )

        # 多个 block 循环
        for i, block in enumerate(self.blocks):
            inds = mappings[self.orders[i]]

            if query_coords is None:
                points = torch.linspace(0, 1, L, dtype=torch.float32, device=self.device)
                refer_points = points[None, :].expand(query.shape[0], -1)[None].unsqueeze(-1)  # [bs, L, 1]
                q = query.features[inds][mappings["flat2win"]]
                q = q.view(-1, L, q.shape[-1])

                k = v = q
                q = block(
                    query=q,
                    key=k,
                    value=v,
                    identity=None,
                    query_pos=query_pos,
                    key_padding_mask=mask,
                    reference_points=refer_points,
                    spatial_shape=[L],
                    level_start_index=[0]
                )
                query.features[inds] = q.view(-1, q.shape[-1])[mappings["win2flat"]]

            else:
                Cv = value.indices[inds]
                Cq = padding_mappings["Cq"]

                dists, knn_idx, _ = knn_points(Cq[None].float(), Cv[None].float(), K=1)     # dists: [1, M, 1]
                dists = dists.squeeze()                                         # [M]
                knn_idx = knn_idx.squeeze()                                     # [M]

                match_threshold = 1e-6
                match_any = (dists < match_threshold)                           # [M]

                refer_points = torch.zeros(
                    Cq.size(0), dtype=torch.float32, device=Cv.device
                )
                refer_points[match_any] = knn_idx[match_any].float()
                non_match_mask = ~match_any
                if non_match_mask.any():
                    pad_inds = padding_mappings[self.orders[i]]
                    mask_value = padding_mappings["mask_value"]
                    Cp = padding_mappings["Cp"]
                    non_match_mask = padding_mappings["non_match_mask"]
                    refer_points = self.get_reference_points(
                        sorted_value_coords=Cv,
                        query_coords=Cq,
                        is_value=mask_value,
                        sort_idx_all=pad_inds,
                        query_ref=refer_points,
                        all_coords=Cp,
                        non_match_mask=non_match_mask,
                    )
                refer_points = refer_points.view(bs, -1)[:, :, None, None]
                refer_points = refer_points / (L - 1)
                k = key.features[inds][mappings["flat2win"]]
                k = k.view(-1, L, k.shape[-1])
                v = value.features[inds][mappings["flat2win"]]
                v = v.view(-1, L, v.shape[-1])

                query = block(
                    query=query,
                    key=k,
                    value=v,
                    identity=None,
                    query_pos=query_pos,
                    key_padding_mask=mask,
                    reference_points=refer_points,
                    spatial_shapes=torch.as_tensor([L], dtype=torch.long, device=query.device).unsqueeze(-1),
                    level_start_index=torch.as_tensor([0], dtype=torch.long, device=query.device)
                )

        return query


class LevelFusion(nn.Module):
    def __init__(
        self,
        input_channel,
        num_levels,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.attn = nn.MultiheadAttention(input_channel, num_heads, batch_first=True, dropout=dropout)

        self.init_weights()
        # Learnable positional embedding for each level
        self.level_pos_embedding = nn.Parameter(torch.randn(num_levels, input_channel))
        nn.init.trunc_normal_(self.level_pos_embedding, std=0.02)

        # Residual & Norm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_channel)

    def init_weights(self):
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.constant_(self.attn.out_proj.bias, 0.0)
        

    def forward(self, x):
        # x: [bs, num_levels, num_queries, C]
        bs, num_levels, num_queries, C = x.shape
        assert num_levels == self.num_levels

        # permute to [bs * num_queries, num_levels, C]
        x = x.permute(0, 2, 1, 3).reshape(bs * num_queries, num_levels, C)

        # Add position embedding to levels
        pos = self.level_pos_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)  # [bs*num_queries, num_levels, C]
        x = x + pos

        # Attention along level dimension
        attn_out, _ = self.attn(x, x, x)  # [bs*num_queries, num_levels, C]

        # Residual connection + norm
        x = self.norm(x + self.dropout(attn_out))

        # Mean-pooling over levels to aggregate multi-scale features
        out = x.mean(dim=1)  # [bs*num_queries, C]

        # reshape back: [bs, num_queries, C]
        return out.view(bs, num_queries, C)


class SerializationLayer(nn.Module):
    def __init__(
            self,
            window_shape,
            orders=["z", "z-trans"],
            depth=8,
            shifts_rate=0.0
    ):
        super().__init__()
        self.window_shape = window_shape
        self.depth = depth
        self.orders = orders
        self.shifts_rate = 0.0

        # serialization
        if "z" in orders or "z-trans" in orders:
            self.serialization = ZOrderSerialization(window_shape=self.window_shape, depth=self.depth)
        if "x" in orders or "y" in orders:
            self.serialization = FlattenWindowsSerialization(window_shape=self.window_shape, win_version='v3')
        if "xx" in orders or "yy" in orders or "xy" in orders or "yx" in orders:
            self.serialization = FlattenWindowsSerialization(window_shape=self.window_shape, win_version='v3e')
        if "hilbert" in orders or "hilbert-trans" in orders:
            self.serialization = HilbertSerialization(window_shape=self.window_shape, depth=self.depth)

    def forward(
            self,
            coords,
            batch_size,
            max_voxels,
            sparse_shape
    ):
        shifts = [True if self.shifts_rate > torch.rand(1) else False for _ in range(len(self.orders))]
        orders = self.orders

        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))

        # 每个 batch 补齐到 max_voxels
        num_per_batch_p = torch.full((batch_size,), max_voxels, dtype=torch.long, device=coords.device)

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))

        total_padded = batch_start_indices_p[-1]
        flat2win = torch.empty(total_padded, dtype=torch.long, device=coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)

        # mask（False: real points, True: padding points）
        mask = torch.ones(total_padded, dtype=torch.bool, device=coords.device)

        for i in range(batch_size):
            start = batch_start_indices[i]
            end = batch_start_indices[i + 1]
            padded_start = batch_start_indices_p[i]
            padded_end = batch_start_indices_p[i + 1]

            num_real = end - start
            num_padded = padded_end - padded_start

            # 把真实点填入 flat2win
            flat2win[padded_start: padded_start + num_real] = win2flat[start:end]

            # 设置真实点 mask 为 True
            mask[padded_start: padded_start + num_real] = False

            if num_real < num_padded:
                num_missing = num_padded - num_real

                # 构造镜像 padding 索引（例如: 0,1,2,3 → 2,1,0,1,2,3,...）
                reflect_base = torch.cat([
                    torch.arange(num_real - 2, -1, -1, device=coords.device),
                    torch.arange(1, num_real, device=coords.device)
                ])
                # 重复并裁剪到 num_missing 长度
                repeat_count = (num_missing + reflect_base.numel() - 1) // reflect_base.numel()
                mirror = reflect_base.repeat(repeat_count)[:num_missing]

                # 取真实点映射过来的值做映射
                mirror_extended = win2flat[start:end][mirror]

                # 填入 flat2win 的 padding 区域
                flat2win[padded_start + num_real: padded_end] = mirror_extended

            # 修改索引
            win2flat[start:end] += (padded_start - start)
            # flat2win[padded_start:padded_end] -= (padded_start - start)

        # 输出 mappings
        mappings = {
            "flat2win": flat2win,
            "win2flat": win2flat,
            "mask": mask
        }

        for order, shift in zip(orders, shifts):
            mappings.update(self.encoder(coords, sparse_shape, shift, order))

        return mappings

    def encoder(self, coords, sparse_shape, shifts, order):
        """
        :param coords: [bs, y, x]
        :return:
        """
        meta = {}
        n, ndim = coords.shape
        #                   z order                hilbert                 flatten windows
        assert order in {"z", "z-trans", "hilbert", "hilbert-trans", "x", "y", "xy", "yx", "xx", "yy"}
        if order == "z":
            coords2curve = self.serialization(
                coords, sparse_shape, shifts
            )
        elif order == "z-trans":
            coords2curve = self.serialization(
                coords[..., [0, 1, 3, 2]] if ndim == 4 else coords[..., [0, 2, 1]],
                sparse_shape, shifts
            )
        elif order == "hilbert":
            coords2curve = self.serialization(
                coords, sparse_shape, shifts
            )
        elif order == "hilbert-trans":
            coords2curve = self.serialization(
                coords[..., [0, 1, 3, 2]] if ndim == 4 else coords[..., [0, 2, 1]],
                sparse_shape, shifts
            )
        elif order == "xy" or order == "x":
            coords2curve = self.serialization(
                coords, sparse_shape, shifts, mapping_name=order
            )
        elif order == "yx" or order == "y":
            coords2curve = self.serialization(
                coords, sparse_shape, shifts, mapping_name=order
            )
        elif order == "xx":
            coords2curve = self.serialization(
                coords, sparse_shape, shifts, mapping_name=order
            )
        elif order == "yy":
            coords2curve = self.serialization(
                coords, sparse_shape, shifts, mapping_name=order
            )
        else:
            raise NotImplementedError

        meta[order] = coords2curve

        return meta

class DeformableAttention(nn.Module):
    def __init__(
            self,
            input_channel=256,
            embed_dim=256,
            num_heads=8,
            num_points=32,
            dropout=0.1,
            im2col_step=64,
    ):
        super().__init__()
        self.in_channel = input_channel

        # attn
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.im2col_step = im2col_step

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # offset layer
        self.sampling_offsets = nn.Linear(self.embed_dim, self.num_heads * self.num_points)
        self.sampling_weights = nn.Linear(self.embed_dim, self.num_heads * self.num_points)

        # proj layer
        self.value_proj = nn.Linear(self.in_channel, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.in_channel)

        # layer norm
        self.layer_norm = nn.LayerNorm(self.in_channel)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""

        def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
            # If the module has a weight and the weight is not None, initialize the weight to a constant
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            # If the module has a bias and the bias is not None, initialize the bias to a constant
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        def xavier_init(module: nn.Module,
                        gain: float = 1,
                        bias: float = 0,
                        distribution: str = 'normal') -> None:
            assert distribution in ['uniform', 'normal']
            if hasattr(module, 'weight') and module.weight is not None:
                if distribution == 'uniform':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                else:
                    nn.init.xavier_normal_(module.weight, gain=gain)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init_base = torch.stack([thetas.cos()], -1)  # (num_heads, 1)
        constant_init(self.sampling_offsets, val=0.)

        grid_init = (grid_init_base / grid_init_base.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1).repeat(1, 1, self.num_points, 1)
        for head_index in range(self.num_heads):
            scale_factor = (head_index % (self.num_heads // 2) + 1)  # 计算缩放因子
            for i in range(self.num_points):
                grid_init[head_index, :, i, :] *= scale_factor * (i + 1)
        self.sampling_offsets.bias.data = grid_init.view(-1)

        # 消融实验
        xavier_init(self.sampling_offsets, distribution='uniform', bias=0.)
        # xavier_init(self.sampling_weights, distribution='uniform', bias=0.)
        constant_init(self.sampling_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            reference_points: Optional[torch.Tensor] = None,
            spatial_shapes: Optional[torch.Tensor] = None,
            level_start_index: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

         Args:
            query (torch.Tensor): Query of Transformer with shape
                 (bs, num_query, embed_dims).
            value (torch.Tensor | None): The value tensor with shape
                 `(bs, num_key, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                 same shape as `query`. Default None. If None,
                 `query` will be used.
            query_pos (torch.Tensor): The pospatial_shapessitional encoding for `query`.
                 Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                 shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                 points with shape (bs, num_query, num_levels, 1),
                 all elements is range in [0, 1], top-left (0,0),
                 bottom-right (1, 1), including padding area.
                 or (N, Length_{query}, num_levels, 4), add
                 additional two dimensions is (w, h) to
                 form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                 each different levels. With shape (num_levels, 2),
                 last dimension represents (h, w).
            level_start_index (torch.Tensor): It is used to indicate
                 the starting position of each scale feature
                 for cuda and code calculation
         Returns:
             torch.Tensor: forwarded results with shape
             [bs, num_query, embed_dims].
         """

        # self attn
        if key is not None and value is not None and torch.equal(query, key) and torch.equal(key, value):
            temp_key = temp_value = query

            query = self._forward(
                query=query,
                key=temp_key,
                value=temp_value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                pos_padding_mask=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            return query

        # cross attn
        else:
            query = self._forward(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                pos_padding_mask=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
            return query

    def _forward(self,
                 query: torch.Tensor,
                 key: Optional[torch.Tensor] = None,
                 value: Optional[torch.Tensor] = None,
                 identity: Optional[torch.Tensor] = None,
                 query_pos: Optional[torch.Tensor] = None,
                 key_padding_mask: Optional[torch.Tensor] = None,
                 pos_padding_mask: Optional[torch.Tensor] = None,
                 reference_points: Optional[torch.Tensor] = None,
                 spatial_shapes: Optional[torch.Tensor] = None,
                 level_start_index: Optional[torch.Tensor] = None,
                 **kwargs
                 ):
        """Forward Function of MultiScaleDeformAttention.

         Args:
             query (torch.Tensor): Query of Transformer with shape
                 (bs, num_query, embed_dims).
             value (torch.Tensor | None): The value tensor with shape
                 `(bs, num_key, embed_dims)`.
             identity (torch.Tensor): The tensor used for addition, with the
                 same shape as `query`. Default None. If None,
                 `query` will be used.
             query_pos (torch.Tensor): The pospatial_shapessitional encoding for `query`.
                 Default: None.
             key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                 shape [bs, num_key].
             reference_points (torch.Tensor):  The normalized reference
                 points with shape (bs, num_query, num_levels, 1),
                 all elements is range in [0, 1], top-left (0,0),
                 bottom-right (1, 1), including padding area.
                 or (N, Length_{query}, num_levels, 4), add
                 additional two dimensions is (w, h) to
                 form reference boxes.
             spatial_shapes (torch.Tensor): Spatial shape of features in
                 different levels. With shape (num_levels),
                 last dimension represents (l).
         Returns:
             torch.Tensor: forwarded results with shape
             [num_query, bs, embed_dims].
         """
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            if pos_padding_mask is not None:
                query = query + query_pos.masked_fill(pos_padding_mask[..., None], 0.0)
            else:
                query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert spatial_shapes.sum() == num_value

        # value: [bs, num_value, embed_dims]
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.view(bs, num_value, self.num_heads, -1)

        # (bs, num_query, num_heads, sum(num_points), 1)
        sampling_offsets = self.sampling_offsets(query)
        # (bs, num_query, num_heads, num_levels=1, num_points_l, 1)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, 1, self.num_points, 1)

        # (bs, num_query, num_heads, num_levels=1, num_points_l)
        sampling_weights = self.sampling_weights(query)
        sampling_weights = sampling_weights.view(bs, num_query, self.num_heads, 1, self.num_points)
        sampling_weights = F.softmax(sampling_weights, dim=-1)

        sampling_locations = (reference_points[:, :, None, :, None, :]
                              + sampling_offsets
                              / spatial_shapes[None, None, None, :, None, :])

        if torch.cuda.is_available() and value.is_cuda:
            output = MSDeformAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                sampling_weights, self.im2col_step
            )
        else:
            output = deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, sampling_weights
            )

        output = self.output_proj(output)

        return self.layer_norm(self.dropout(output) + identity)


def deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        sampling_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, embed_dims)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (1,),
            last dimension 1 represent (L)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_points, 1),
            the last dimension 1 represent (x).
        sampling_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = \
        sampling_locations.shape
    value_list = value.split([L_ for L_ in value_spatial_shapes], dim=1)
    # sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, L_ in enumerate(value_spatial_shapes):
        # bs, L_, num_heads, embed_dims ->
        # bs, L_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, L_ ->
        # bs*num_heads, embed_dims, L_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, L_
        )
        # bs, num_queries, num_heads, num_points, 1 ->
        # bs, num_heads, num_queries, num_points, 1 ->
        # bs*num_heads, num_queries, num_points, 1
        sampling_grid_l_ = sampling_locations[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_grid_l_ = sampling_grid_l_ * L_ - 1

        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = grid_sample(value_l_, sampling_grid_l_, padding_mode='zeros')
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = sampling_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


def grid_sample(value: torch.Tensor, sampling_grid: torch.Tensor, padding_mode: str = 'zeros') -> torch.Tensor:
    """
    PyTorch 版本的 1D 线性插值函数，模拟 grid_sample 接口。

    Args:
        value (torch.Tensor): 输入数据，形状为 (batch_size * num_heads, embed_dims, length)
        sampling_grid (torch.Tensor): 采样点位置，形状为 (batch_size * num_heads, num_queries, num_points, 1)
        padding_mode (str): 边界处理方式，默认为 'zeros'，可选 'border'

    Returns:
        torch.Tensor: 插值结果，形状为 (batch_size * num_heads, embed_dims, num_queries, num_points)
    """
    batch_heads, embed_dims, length = value.shape
    _, _, num_points, _ = sampling_grid.shape

    # 计算 x_low 和 x_high
    # here may cause a bug:
    # x = [1.0, 2.0, 3.0, 4.0]
    # x = torch.floor(x)
    # x = [1.0, 2.0, 3.0, 3.0]
    x = sampling_grid.squeeze(-1)  # 去掉最后一维
    x_low = torch.floor(x).long()
    x_high = x_low + 1

    lx = x - x_low.float()
    hx = 1.0 - lx

    # 处理边界
    if padding_mode == 'border':
        x_low = torch.clamp(x_low, 0, length - 1)
        x_high = torch.clamp(x_high, 0, length - 1)
    else:  # zeros padding
        mask_low = (x_low >= 0) & (x_low < length)
        mask_high = (x_high >= 0) & (x_high < length)
        x_low = torch.clamp(x_low, 0, length - 1)
        x_high = torch.clamp(x_high, 0, length - 1)

    # 索引数据
    x_low_expanded = x_low.reshape(batch_heads, -1).unsqueeze(-2).expand(-1, embed_dims, -1)
    x_high_expanded = x_high.reshape(batch_heads, -1).unsqueeze(-2).expand(-1, embed_dims, -1)

    v1 = torch.gather(value, -1, index=x_low_expanded).reshape(batch_heads, embed_dims, -1, num_points)
    v2 = torch.gather(value, -1, index=x_high_expanded).reshape(batch_heads, embed_dims, -1, num_points)

    # 处理 padding_mode='zeros' 的无效点
    if padding_mode == 'zeros':
        v1 = v1 * mask_low.unsqueeze(1).float()
        v2 = v2 * mask_high.unsqueeze(1).float()

    # 线性插值
    interpolated = hx.unsq
