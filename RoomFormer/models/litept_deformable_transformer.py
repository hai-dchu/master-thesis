import math
import sys

import torch
from torch import nn

from .deformable_transformer import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
)
from .matcher import build_matcher
from .roomformer import MLP, SetCriterion, _get_clones

sys.path.append("LitePT")
from LitePT.litept.model import MLP as FFN
from LitePT.litept.model import LitePT, PointSequential


def build_feature_map_lexsort(
    coords: torch.Tensor,  # Shape: (N, 3) [x, y, z]
    feats: torch.Tensor,  # Shape: (N, embed_dim)
    batch_idxs: torch.Tensor,  # Shape: (N,) batch index per point
    batch_size: int = 1,
    grid_size: int = 256,
    max_k: int = 20,
    embed_dim: int = 72,
) -> torch.Tensor:
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


def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


class LitePTEncoder(nn.Module):
    def __init__(
        self,
        litept: nn.Module,
        enc_out_channels,
        enc_mlp_ratio,
        out_channels=16,
        grid_size=256,
        max_k=16,
    ):
        super().__init__()
        self.backbone = litept
        self.mlp = PointSequential(
            FFN(
                in_channels=enc_out_channels,
                hidden_channels=enc_out_channels * enc_mlp_ratio,
                out_channels=out_channels,
            )
        )

        self.grid_size = grid_size
        self.max_k = max_k
        self.embed_dim = out_channels
        self.out_channels = out_channels * max_k

        # Possibly freezing LitePT?

    def forward(self, x):
        out = self.mlp(self.backbone(x))
        feat = build_feature_map_lexsort(
            coords=out.coord,
            feats=out.feat,
            batch_idxs=out.batch,
            batch_size=int(out.batch.max().item() + 1),
            grid_size=self.grid_size,
            max_k=self.max_k,
            embed_dim=self.embed_dim,
        )

        return feat


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
            src=src,
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
        encoder: nn.Module,
        decoder: nn.Module,
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
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = num_queries
        self.num_polys = num_polys
        assert num_queries % num_polys == 0
        self.num_queries_per_poly = num_queries // num_polys

        _enc_out_channels = hidden_dim = encoder.out_channels

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)

        num_pred = self.decoder.num_layers

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

        self.decoder.decoder.coords_embed = self.coords_embed
        self.decoder.decoder.class_embed = self.class_embed

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
        memory = self.encoder(samples)
        bs = memory.shape[0]
        device = memory.device

        tgt = self.tgt_embed.weight
        query_embed = self.query_embed.weight
        # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        masks = torch.zeros(
            bs,
            self.encoder.grid_size,
            self.encoder.grid_size,
            dtype=torch.bool,
            device=device,
        )

        hs, _init_reference, inter_references, inter_classes = self.decoder(
            [memory],
            [masks],
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


def build(args, train=True):
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

        # Missing weight entries for decoder is expected since
        # we are working with LitePT encoder (no dec, no seg)

        model_dict.update(filtered_weights)
        litept.load_state_dict(model_dict)

        if args.litept_frozen:
            litept.requires_grad_(False)

    encoder = LitePTEncoder(
        litept=litept,
        enc_out_channels=72
        if not args.litept_enc_mode
        else args.litept_enc_channels[-1],
        enc_mlp_ratio=args.litept_mlp_ratio,
        out_channels=args.litept_mlp_out_channels,
        grid_size=args.litept_grid_size,
        max_k=args.litept_max_keep,
    )

    decoder = DeformableTransformerDecoderWrapper(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        poly_refine=args.with_poly_refine,
        return_intermediate_dec=True,
        aux_loss=args.aux_loss,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        query_pos_type=args.query_pos_type,
    )

    num_classes = 1
    model = LitePTDeformableTransformer(
        encoder,
        decoder,
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
