import math
import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
from torch import nn

# from torch.nn.init import constant_, normal_, uniform_, xavier_uniform_
from util.misc import NestedTensor

# from models.ops.modules import MSDeformAttn
# from .backbone import build_backbone as build_resnet_backbone
from .deformable_transformer import (
    DeformableTransformer,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    # DeformableTransformerEncoder,
    # DeformableTransformerEncoderLayer,
)
from .matcher import build_matcher
from .position_encoding import build_position_encoding
from .roomformer import MLP, SetCriterion, _get_clones

sys.path.append("LitePT")
import sys

from LitePT.litept.model import LitePT  # , PointSequential


def build_feature_map_lexsort(
    coords: torch.Tensor,  # Shape: (N, 3) [x, y, z]
    feats: torch.Tensor,  # Shape: (N, embed_dim)
    batch_idxs: torch.Tensor,  # Shape: (N,) batch index per point
    batch_size: int = 1,
    grid_size: int = 256,
    max_k: int = 20,
    embed_dim: int = 72,
) -> torch.Tensor:
    """
    Assume that the points xy-coordinates are mapped to range (256,256)
    """
    device = coords.device
    num_points = coords.shape[0]

    # 1. Round coordinates and build composite 1D pixel indices
    x = torch.round(coords[:, 0])
    y = torch.round(coords[:, 1])
    z = coords[:, 2]

    linear_pixel_idx = batch_idxs * (grid_size * grid_size) + y * grid_size + x
    keys = torch.stack([linear_pixel_idx, z], dim=1)

    # 2. Lexicographical Sort (column 1/Z first, then column 0/pixel_idx)
    idx = torch.arange(num_points, device=device)
    for col in [1, 0]:
        idx = idx[torch.argsort(keys[idx, col], stable=True)]

    # Re-order indices and features
    sorted_pixel_idx = keys[idx, 0].long()
    sorted_feats = feats[idx]

    # 3. Compute local slot index (0..k-1) per spatial pixel cell
    new_pixel_mask = torch.cat(
        [
            torch.tensor([True], device=device),
            sorted_pixel_idx[1:] != sorted_pixel_idx[:-1],
        ]
    )

    group_starts = torch.nonzero(new_pixel_mask, as_tuple=False).squeeze(-1)
    group_lengths = torch.diff(
        torch.cat([group_starts, torch.tensor([num_points], device=device)])
    )

    # Sequence position inside each pixel group
    slot_idx = torch.arange(num_points, device=device) - torch.repeat_interleave(
        group_starts, group_lengths
    )

    # 4. Cap at max_k points per pixel
    valid_mask = slot_idx < max_k
    # valid_mask = torch.from_numpy(np.sort(np.random.choice(slot_idx.cpu().numpy(), max_k)))
    valid_pixel_idx = sorted_pixel_idx[valid_mask]
    valid_slot_idx = slot_idx[valid_mask]
    valid_feats = sorted_feats[valid_mask]

    # print((~valid_mask).sum())

    # 5. Scatter sorted features into dense output buffer
    buffer = torch.zeros(
        (batch_size * grid_size * grid_size, max_k, embed_dim),
        dtype=feats.dtype,
        device=device,
    )
    buffer[valid_pixel_idx, valid_slot_idx] = valid_feats

    # 6. Reshape to target shape: (B, max_k * embed_dim, H, W)
    feat_map = buffer.view(batch_size, grid_size, grid_size, max_k * embed_dim)
    feat_map = feat_map.permute(0, 3, 1, 2).contiguous()

    return feat_map


def build_feature_map_density(
    coords: torch.Tensor,  # (N, 3) [x, y, z]
    feats: torch.Tensor,  # (N, embed_dim)
    batch_idxs: torch.Tensor,  # (N,)
    batch_size: int = 1,
    grid_size: int = 256,
    embed_dim: int = 72,
    eps: float = 1e-6,
):
    device = coords.device
    coords = coords[:, :2].clone().detach()  # never mutate caller's tensor
    batch_idxs = batch_idxs.long()
    num_points = coords.shape[0]

    grid_res = torch.tensor((grid_size, grid_size), device=device, dtype=coords.dtype)

    for b in range(batch_size):
        idx = batch_idxs == b
        pts = coords[idx]
        min_c = pts.min(0).values
        max_c = pts.max(0).values
        span = (max_c - min_c).clamp_min(eps)

        pad = 0.1 * span
        min_c = min_c - pad
        max_c = max_c + pad
        padded_span = (max_c - min_c).clamp_min(eps)

        coords[idx] = (pts - min_c[None]) / padded_span[None] * grid_res[None]

    x = coords[:, 0].round().long().clamp_(0, grid_size - 1)
    y = coords[:, 1].round().long().clamp_(0, grid_size - 1)
    linear_idx = batch_idxs * (grid_size * grid_size) + y * grid_size + x

    feat_buf = torch.zeros(
        batch_size * grid_size * grid_size, embed_dim, device=device, dtype=feats.dtype
    )
    feat_buf.index_add_(0, linear_idx, feats)

    count_buf = torch.zeros(
        batch_size * grid_size * grid_size, 1, device=device, dtype=feats.dtype
    )
    count_buf.index_add_(
        0, linear_idx, torch.ones(num_points, 1, device=device, dtype=feats.dtype)
    )

    mean_feat = feat_buf / count_buf.clamp_min(1.0)  # avg
    log_count = torch.log1p(count_buf)  # explicit density channel

    feat_map = torch.cat([mean_feat, log_count], dim=-1)
    feat_map = (
        feat_map.view(batch_size, grid_size, grid_size, -1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    density = (
        log_count.view(batch_size, grid_size, grid_size, -1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return feat_map, density


def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


class Dense2DBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, in_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, in_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out


class LitePTBackbone(nn.Module):
    def __init__(
        self,
        litept: nn.Module,
        # resnet_backbone: nn.Module,
        enc_out_channels,
        enc_mlp_ratio,
        out_channels=16,
        grid_size=256,
    ):
        super().__init__()
        self.litept = litept

        self.enc_out_channels = enc_out_channels
        self.grid_size = grid_size
        self.embed_dim = out_channels
        self.out_channels = out_channels
        # self.resnet_backbone = resnet_backbone

        self.num_feature_levels = num_feature_levels = 4
        hidden_dim = 256

        # if num_feature_levels > 1:
        #     num_backbone_outs = len(resnet_backbone.strides)
        #     input_proj_list = []
        #     for _ in range(num_backbone_outs):
        #         in_channels = resnet_backbone.num_channels[_]
        #         input_proj_list.append(
        #             nn.Sequential(
        #                 nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        #                 nn.GroupNorm(32, hidden_dim),
        #             )
        #         )
        #     for _ in range(num_feature_levels - num_backbone_outs):
        #         input_proj_list.append(
        #             nn.Sequential(
        #                 nn.Conv2d(
        #                     in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
        #                 ),
        #                 nn.GroupNorm(32, hidden_dim),
        #             )
        #         )
        #         in_channels = hidden_dim
        #     self.input_proj = nn.ModuleList(input_proj_list)
        # else:
        #     self.input_proj = nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 nn.Conv2d(
        #                     resnet_backbone.num_channels[0], hidden_dim, kernel_size=1
        #                 ),
        #                 nn.GroupNorm(32, hidden_dim),
        #             )
        #         ]
        #     )

        channels = [64, 128, 256, 256, 256, 256]

        height = width = self.grid_size // 2
        conv = [
            nn.Sequential(
                nn.Conv2d(
                    self.enc_out_channels + 1,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.GroupNorm(32, 64),
                # nn.LayerNorm([64, height, width]),
                # Dense2DBasicBlock(64),
                # Dense2DBasicBlock(64),
                nn.ReLU(),
            )
        ]
        for i in range(len(channels) - 1):
            height = height // 2
            width = width // 2
            conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[i], channels[i + 1], 3, 2, padding=1, bias=False
                    ),
                    nn.GroupNorm(32, channels[i + 1]),
                    # nn.LayerNorm([channels[i + 1], height, width]),
                    # nn.BatchNorm2d(channels[i+1], momentum=0.01, eps=1e-3),
                    nn.ReLU(),
                    Dense2DBasicBlock(channels[i + 1]),
                    Dense2DBasicBlock(channels[i + 1]),
                )
            )

        self.conv = nn.ModuleList(conv)

        # Possibly freezing LitePT?

    # TODO:
    # - Difference between using full LitePT (encoder + decoder) vs encoder only
    # - Try incorporating S2C operation on voxelized feature cloud (https://xieenze.github.io/projects/m2bev/) - No source
    # - Compare S2C approach vs sum approach (ours)
    # - But for now lets try linear layer first
    def forward(self, x):
        out = self.litept(x)
        coords = out["coord"]

        feat, density = build_feature_map_density(
            coords=coords,
            feats=out.feat,
            batch_idxs=out.batch,
            batch_size=int(out.batch.max().item() + 1),
            grid_size=self.grid_size,
            embed_dim=self.enc_out_channels,
        )
        # feat = torch.stack(x["image"]).to(torch.device("cuda"))

        feats = []
        for i, layer in enumerate(self.conv):
            conv_out = layer(feat)
            if i > 1:
                if i > 2:
                    feat = F.relu(
                        conv_out
                        + F.interpolate(feat, size=conv_out.shape[-1], mode="bilinear")
                    )
                else:
                    feat = conv_out
                feats.append(feat)
            else:
                feat = conv_out

        # b, _, h, w = feat.shape
        # device = feat.device
        # mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        # memory, pos = self.resnet_backbone(NestedTensor(feat, mask))
        # feats = []
        # masks = []
        # for l, m in enumerate(memory):
        #     feat, mask = m.decompose()
        #     feats.append(self.input_proj[l](feat))
        #     masks.append(mask)

        # if self.num_feature_levels > len(memory):
        #     _len_srcs = len(memory)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](memory[-1].tensors)
        #         else:
        #             src = self.input_proj[l](feats[-1])
        #         m = mask
        #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
        #             torch.bool
        #         )[0]
        #         pos_l = self.resnet_backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #         feats.append(src)
        #         masks.append(mask)
        #         pos.append(pos_l)

        return feats, density  # masks, pos, density


class DeformableTransformerWrapper(DeformableTransformer):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        poly_refine=True,
        return_intermediate_dec=False,
        aux_loss=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type="none",
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            poly_refine=poly_refine,
            return_intermediate_dec=return_intermediate_dec,
            aux_loss=aux_loss,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            query_pos_type=query_pos_type,
        )

        self.proj = nn.Linear(d_model * 2, d_model)
        # self.proj = nn.Conv2d(d_model * 2, d_model, 3, padding=1, bias=False)

    def forward(
        self,
        litept_srcs,
        resnet_srcs,
        masks,
        pos_embeds,
        query_embed=None,
        tgt=None,
        tgt_masks=None,
    ):
        """
        Separate srcs from litept (goes into decoder) and srcs from resnet (goes into encoder)
        """
        assert query_embed is not None
        # srcs from litept and resnet must be of the same shape

        litept_src_flatten = []
        resnet_src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (litept_src, resnet_src, mask, pos_embed) in enumerate(
            zip(litept_srcs, resnet_srcs, masks, pos_embeds)
        ):
            bs, _, h, w = litept_src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            litept_src = litept_src.flatten(2).transpose(1, 2)
            resnet_src = resnet_src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            litept_src_flatten.append(litept_src)
            resnet_src_flatten.append(resnet_src)
            mask_flatten.append(mask)
        resnet_src_flatten = torch.cat(resnet_src_flatten, 1)
        litept_src_flatten = torch.cat(litept_src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=resnet_src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)

        _resnet_memory = self.encoder(
            resnet_src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        bs, _, _ = litept_src_flatten.shape

        # TODO: Add some (learnable) linear layer to join the memory from resnet and litept
        # Channel-wise concatenate
        shared_memory = torch.concat(
            [litept_src_flatten, _resnet_memory], axis=2
        ).reshape((-1, self.d_model * 2))
        shared_memory = self.proj(shared_memory).reshape((bs, -1, self.d_model))

        # Uncomment to replicate RoomFormer
        # shared_memory = _resnet_memory

        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = query_embed.sigmoid()
        init_reference_out = reference_points

        hs, inter_references, inter_classes = self.decoder(
            tgt=tgt,
            reference_points=reference_points,
            src=shared_memory,
            src_flatten=None,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=query_embed,
            src_padding_mask=mask_flatten,
        )

        return hs, init_reference_out, inter_references, inter_classes


class DeformableTransformerDecoderWrapper(nn.Module):
    """
    Act as DeformableTransformer class but for decoder only
    """

    def __init__(
        self,
        d_model=256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        poly_refine: bool = True,
        return_intermediate_dec: bool = False,
        aux_loss: bool = False,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        query_pos_type: int = "none",
    ):
        super().__init__()
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            poly_refine=poly_refine,
            return_intermediate=return_intermediate_dec,
            aux_loss=aux_loss,
            query_pos_type=query_pos_type,
        )
        self.num_layers = num_decoder_layers

        if query_pos_type == "sine":
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        srcs,
        masks,
        query_embed=None,
        tgt=None,
        tgt_masks=None,
    ):
        assert query_embed is not None

        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)

        bs, _, _ = src_flatten.shape

        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = query_embed.sigmoid()
        init_reference_out = reference_points

        hs, inter_references, inter_classes = self.decoder(
            tgt=tgt,
            reference_points=reference_points,
            src=src_flatten,
            src_flatten=None,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=query_embed,
            src_padding_mask=mask_flatten,
        )

        return hs, init_reference_out, inter_references, inter_classes


class LitePTDeformableTransformer(nn.Module):
    """
    Equivalent to RoomFormer class
    """

    def __init__(
        self,
        litept_backbone: nn.Module,
        position_embedding: nn.Module,
        # resnet_backbone: nn.Module,
        transformer: nn.Module,
        num_classes: int,
        num_queries: int,
        num_polys: int,
        num_feature_levels: int = 4,
        with_poly_refine: bool = True,
        return_intermediate_dec: bool = False,
        aux_loss: bool = False,
        semantic_classes: int = -1,
    ):
        super().__init__()
        self.litept_backbone = litept_backbone
        # self.resnet_backbone = resnet_backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.num_polys = num_polys
        assert num_queries % num_polys == 0
        self.num_queries_per_poly = num_queries // num_polys

        _enc_out_channels = hidden_dim = 256

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.num_feature_levels = num_feature_levels
        self.position_embedding = position_embedding

        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        # if num_feature_levels > 1:
        #     num_backbone_outs = len(resnet_backbone.strides)
        #     input_proj_list = []
        #     for _ in range(num_backbone_outs):
        #         in_channels = resnet_backbone.num_channels[_]
        #         input_proj_list.append(
        #             nn.Sequential(
        #                 nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        #                 nn.GroupNorm(32, hidden_dim),
        #             )
        #         )
        #     for _ in range(num_feature_levels - num_backbone_outs):
        #         input_proj_list.append(
        #             nn.Sequential(
        #                 nn.Conv2d(
        #                     in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
        #                 ),
        #                 nn.GroupNorm(32, hidden_dim),
        #             )
        #         )
        #         in_channels = hidden_dim
        #     self.input_proj = nn.ModuleList(input_proj_list)
        # else:
        #     self.input_proj = nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 nn.Conv2d(
        #                     resnet_backbone.num_channels[0], hidden_dim, kernel_size=1
        #                 ),
        #                 nn.GroupNorm(32, hidden_dim),
        #             )
        #         ]
        #     )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)

        num_pred = self.transformer.decoder.num_layers

        if with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.coords_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.coords_embed = nn.ModuleList(
                [self.coords_embed for _ in range(num_pred)]
            )

        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed

        # Semantically-rich floorplan
        self.room_class_embed = None
        if semantic_classes > 0:
            self.room_class_embed = nn.Linear(hidden_dim, semantic_classes)

        self.aux_loss = aux_loss

    def forward(self, samples: dict):
        """
        :param samples: a dictionary containing:
        - file_name: density map file name (redundant)
        - height: density map height (redundant)
        - width: density map width (redundant)
        - image_id: density map id in the dataset (redundant)
        - image: the actual density map (batch_size, 256, 256) (redundant)
        - instances: target corners coordinates (in counter-clockwise order)
        - coord: 3D coordinates of the (batched) point clouds
        - grid_coord: corresponding coordinates of the voxelized point clouds
        - grid_size: sample rate for coord and grid_coord
        - offset: mark end of each point cloud in batch
        - feat: [rgb, normal] stacked rgb + normal values for each point
        """
        litept_feats, _ = self.litept_backbone(samples)
        bs = litept_feats[0].shape[0]
        device = litept_feats[0].device

        # density = torch.stack(samples["image"]).to(device)

        tgt = self.tgt_embed.weight
        query_embed = self.query_embed.weight
        # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        _mask = torch.zeros(
            bs,
            256,
            256,
            dtype=torch.bool,
            device=device,
        )

        # nested_density = NestedTensor(density, _mask)
        # resnet_srcs, pos = self.resnet_backbone(nested_density)
        # Probably use these position embedding + mask for both

        # mask = torch.zeros(
        #     bs,
        #     litept_feats[0].shape[-2],
        #     litept_feats[0].shape[-1],
        #     dtype=torch.bool,
        #     device=device
        # )
        # pos = self.position_embedding(_mask)
        pos = []

        # TODO: Currently this is just a placeholder.
        # Maybe add some convolution to replace feat interpolation?
        masks = []
        # resnet_feats = []
        for ifeat in litept_feats:
            # ifeat = F.interpolate(feat, size=size, mode="bilinear")
            size = ifeat.shape[-2:]
            imask = (
                F.interpolate(_mask[None].float(), size=size, mode="nearest")
                .squeeze(1)
                .bool()
            )[0]
            ipos = self.position_embedding(NestedTensor(ifeat, imask))

            # litept_feats.append(ifeat)
            masks.append(imask)
            pos.append(ipos)

        # for l, feat in enumerate(resnet_srcs):
        #     src, mask = feat.decompose()
        #     resnet_feats.append(self.input_proj[l](src))
        #     masks.append(mask)
        #     assert mask is not None
        # if self.num_feature_levels > len(resnet_feats):
        #     _len_srcs = len(resnet_feats)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](resnet_srcs[-1].tensors)
        #         else:
        #             src = self.input_proj[l](resnet_feats[-1])
        #         m = _mask
        #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
        #             torch.bool
        #         )[0]
        #         pos_l = self.resnet_backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #         resnet_feats.append(src)
        #         masks.append(mask)
        #         pos.append(pos_l)

        hs, _init_reference, inter_references, inter_classes = self.transformer(
            litept_feats,
            # resnet_feats,
            masks,
            pos,
            query_embed,
            tgt,
        )
        num_layer = hs.shape[0]
        outputs_class = inter_classes.reshape(
            num_layer, bs, self.num_polys, self.num_queries_per_poly
        )
        outputs_coord = inter_references.reshape(
            num_layer, bs, self.num_polys, self.num_queries_per_poly, 2
        )
        out = {"pred_logits": outputs_class[-1], "pred_coords": outputs_coord[-1]}

        # hack implementation of room label prediction, not compatible with auxiliary loss
        if self.room_class_embed is not None:
            outputs_room_class = self.room_class_embed(
                hs[-1]
                .view(bs, self.num_polys, self.num_queries_per_poly, -1)
                .mean(axis=2)
            )
            out = {
                "pred_logits": outputs_class[-1],
                "pred_coords": outputs_coord[-1],
                "pred_room_logits": outputs_room_class,
            }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out


def build_litept(args):
    litept = LitePT(
        in_channels=args.litept_in_channels,  # 6,
        order=args.litept_order,  # ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=args.litept_stride,  # (2, 2, 2, 2),
        enc_depths=args.litept_enc_depths,  # (2, 2, 2, 6, 2),
        enc_channels=args.litept_enc_channels,  # (36, 72, 144, 252, 504),
        enc_num_head=args.litept_enc_num_head,  # (2, 4, 8, 14, 28),
        enc_patch_size=args.litept_enc_patch_size,  # (1024, 1024, 1024, 1024, 1024),
        enc_conv=args.litept_enc_conv,  # (True, True, True, False, False),
        enc_attn=args.litept_enc_attn,  # (False, False, False, True, True),
        enc_rope_freq=args.litept_enc_rope_freq,  # (100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=args.litept_dec_depths,  # (0, 0, 0, 0),
        dec_channels=args.litept_dec_channels,  # (72, 72, 144, 252),
        dec_num_head=args.litept_dec_num_head,  # (4, 4, 8, 14),
        dec_patch_size=args.litept_dec_patch_size,  # (1024, 1024, 1024, 1024),
        dec_conv=args.litept_dec_conv,  # (False, False, False, False),
        dec_attn=args.litept_dec_attn,  # (False, False, False, False),
        dec_rope_freq=args.litept_dec_rope_freq,  # (100.0, 100.0, 100.0, 100.0),
        mlp_ratio=args.litept_mlp_ratio,  # 4,
        qkv_bias=args.litept_qkv_bias,  # True,
        qk_scale=args.litept_qk_scale,  # None,
        attn_drop=args.litept_attn_drop,  # 0.0,
        proj_drop=args.litept_proj_drop,  # 0.0,
        drop_path=args.litept_drop_path,  # 0.3,
        shuffle_orders=args.litept_shuffle_orders,  # True,
        pre_norm=args.litept_prenorm,  # True,
        enc_mode=args.litept_enc_mode,  # True,
    )

    if args.litept_checkpoint is not None:
        weights = torch.load(args.litept_checkpoint, weights_only=False)
        model_dict = litept.state_dict()
        filtered_weights = {}
        for k, v in weights["state_dict"].items():
            # 16 == len("module.backbone.")
            # Just a hack to accommodate difference in layer naming convention
            if k[16:] in model_dict:
                if v.shape == model_dict[k[16:]].shape:
                    filtered_weights[k[16:]] = v
                else:
                    print(f"{v.shape} mismatch {model_dict[k[16:]].shape}")
            else:
                print(f"{k} missing")

        # Missing weight entries for transformer is expected since
        # we are working with LitePT encoder (no dec, no seg)

        model_dict.update(filtered_weights)
        litept.load_state_dict(model_dict)

        if args.litept_frozen:
            litept.requires_grad_(False)

    return litept


def build(args, train=True):
    litept = build_litept(args)

    litept_out_channels = (
        72 if not args.litept_enc_mode else args.litept_enc_channels[-1]
    )

    # resnet_backbone = build_resnet_backbone(args)
    position_embedding = build_position_encoding(args)

    litept_backbone = LitePTBackbone(
        litept=litept,
        # resnet_backbone=resnet_backbone,
        enc_out_channels=litept_out_channels,
        enc_mlp_ratio=args.litept_mlp_ratio,
        out_channels=args.litept_mlp_out_channels,
        grid_size=args.litept_grid_size,
    )

    # transformer = DeformableTransformerDecoderWrapper(
    #     d_model=args.hidden_dim,
    #     nhead=args.nheads,
    #     num_decoder_layers=args.dec_layers,
    #     dim_feedforward=args.dim_feedforward,
    #     dropout=args.dropout,
    #     activation="relu",
    #     poly_refine=args.with_poly_refine,
    #     return_intermediate_dec=True,
    #     aux_loss=args.aux_loss,
    #     num_feature_levels=args.num_feature_levels,
    #     dec_n_points=args.dec_n_points,
    #     query_pos_type=args.query_pos_type,
    # )

    transformer = DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        poly_refine=args.with_poly_refine,
        return_intermediate_dec=True,
        aux_loss=args.aux_loss,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.dec_n_points,
        query_pos_type=args.query_pos_type,
    )

    num_classes = 1
    model = LitePTDeformableTransformer(
        litept_backbone,
        position_embedding,
        # resnet_backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_polys=args.num_polys,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_poly_refine=args.with_poly_refine,
        semantic_classes=args.semantic_classes,
    )

    if not train:
        return model

    # device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_ce_room": args.room_cls_loss_coef,
        "loss_coords": args.coords_loss_coef,
        "loss_raster": args.raster_loss_coef,
    }
    weight_dict["loss_dir"] = 1

    enc_weight_dict = {}
    enc_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(enc_weight_dict)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "polys", "cardinality"]
    # num_classes, matcher, weight_dict, losses
    criterion = SetCriterion(
        num_classes, args.semantic_classes, matcher, weight_dict, losses
    )

    return model, criterion
