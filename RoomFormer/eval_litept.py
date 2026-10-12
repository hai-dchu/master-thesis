import argparse
import os
import random
import sys
from pathlib import Path

sys.path.append("LitePT")


from collections.abc import Mapping, Sequence

import numpy as np
import torch
import util.misc as utils
from datasets import build_normal_dataset as build_dataset
from engine_litept import evaluate_floor
from models import build_model_litept as build_model
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def _config_litept(parser):
    group = parser.add_argument_group("config_litept")
    # LitePT
    group.add_argument(
        "--litept_in_channels",
        type=int,
        default=6,
        help="input channel of litept, default 6",
    )
    group.add_argument(
        "--litept_order",
        default=("z", "z-trans", "hilbert", "hilbert-trans"),
        help="serialization order following PointV3",
    )
    group.add_argument("--litept_stride", default=(2, 2, 2, 2))
    group.add_argument("--litept_enc_depths", default=(2, 2, 2, 6, 2))
    group.add_argument("--litept_enc_channels", default=(36, 72, 144, 252, 504))
    group.add_argument("--litept_enc_num_head", default=(2, 4, 8, 14, 28))
    group.add_argument(
        "--litept_enc_patch_size", default=(1024, 1024, 1024, 1024, 1024)
    )
    group.add_argument("--litept_enc_conv", default=(True, True, True, False, False))
    group.add_argument("--litept_enc_attn", default=(False, False, False, True, True))
    group.add_argument(
        "--litept_enc_rope_freq", default=(100.0, 100.0, 100.0, 100.0, 100.0)
    )
    group.add_argument("--litept_dec_depths", default=(0, 0, 0, 0))
    group.add_argument("--litept_dec_channels", default=(72, 72, 144, 252))
    group.add_argument("--litept_dec_num_head", default=(4, 4, 8, 14))
    group.add_argument("--litept_dec_patch_size", default=(1024, 1024, 1024, 1024))
    group.add_argument("--litept_dec_conv", default=(False, False, False, False))
    group.add_argument("--litept_dec_attn", default=(False, False, False, False))
    group.add_argument("--litept_dec_rope_freq", default=(100.0, 100.0, 100.0, 100.0))
    group.add_argument("--litept_mlp_ratio", default=4)
    group.add_argument("--litept_qkv_bias", default=True)
    group.add_argument("--litept_qk_scale", default=None)
    group.add_argument("--litept_attn_drop", default=0.0)
    group.add_argument("--litept_proj_drop", default=0.0)
    group.add_argument("--litept_drop_path", default=0.3)
    group.add_argument("--litept_shuffle_orders", default=True)
    group.add_argument("--litept_prenorm", default=True)
    group.add_argument("--litept_enc_mode", default=False, action="store_true")

    # Grid size for density creation
    group.add_argument("--litept_grid_size", default=256, type=int)

    # Grid size for voxelization
    group.add_argument("--grid_sample_size", default=0.5, type=float)

    # MLP layer after LitePT
    group.add_argument(
        "--litept_mlp_out_channels",
        default=16,
        type=int,
        help="output channel dim for MLP layer after LitePT",
    )
    group.add_argument(
        "--litept_max_keep",
        default=16,
        type=int,
        help="maximum number of points in each height cell to keep",
    )

    group.add_argument(
        "--litept_checkpoint",
        type=str,
        help="litept checkpoint if available, deciding if litept is frozen or not",
    )
    group.add_argument(
        "--litept_frozen",
        default=False,
        action="store_true",
        help="if included, freeze LitePT",
    )
    group.add_argument(
        "--litept_start_epoch",
        default=0,
        type=int,
        help="freeze litept until this epoch",
    )
    return parser


def _config_deformable_transformer(parser):
    group = parser.add_argument_group("config_deformable_decoder")
    # DeformableTransformerDecoder
    group.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    group.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    group.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    group.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    group.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    group.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    group.add_argument(
        "--with_poly_refine",
        default=True,
        action="store_true",
        help="iteratively refine reference points (i.e. positional part of polygon queries)",
    )
    group.add_argument(
        "--aux_loss",
        default=False,
        action="store_true",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    group.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )
    group.add_argument("--dec_n_points", default=4, type=int)
    group.add_argument(
        "--query_pos_type",
        default="sine",
        type=str,
        choices=("static", "sine", "none"),
        help="Type of query pos in decoder - \
            1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
            2. sine: sine embedding from reference points (so if references points update, query_pos also \
            3. none: remove query_pos",
    )

    # backbone
    group.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    group.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    group.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    group.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )

    return parser


def _config_litept_deformable_transformer(parser):
    group = parser.add_argument_group("config_litept_deformable_transformer")
    # LitePTDeformableTransformer
    group.add_argument(
        "--num_queries",
        default=800,
        type=int,
        help="Number of query slots (num_polys * max. number of corner per poly)",
    )
    group.add_argument(
        "--num_polys",
        default=20,
        type=int,
        help="Number of maximum number of room polygons",
    )
    group.add_argument(
        "--masked_attn",
        default=False,
        action="store_true",
        help="if true, the query in one room will not be allowed to attend other room (Unused settings, although it could be use in the future)",
    )
    group.add_argument(
        "--semantic_classes",
        default=-1,
        type=int,
        help="Number of classes for semantically-rich floorplan:  \
            1. default -1 means non-semantic floorplan \
            2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty",
    )

    return parser


def _config_roomformer_criterion(parser):
    group = parser.add_argument_group("config_roomformer_criterion")

    # From RoomFormer original
    # matcher
    group.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    group.add_argument(
        "--set_cost_coords",
        default=5,
        type=float,
        help="L1 coords coefficient in the matching cost",
    )

    # loss coefficients
    group.add_argument("--cls_loss_coef", default=2, type=float)
    group.add_argument("--room_cls_loss_coef", default=0.2, type=float)
    group.add_argument("--coords_loss_coef", default=5, type=float)
    group.add_argument("--raster_loss_coef", default=1, type=float)

    return parser


def config():
    parser = argparse.ArgumentParser(prog="LitePTDeformableTransformer", add_help=False)

    # Just for the ease of comprehension
    parser = _config_litept(parser)
    parser = _config_deformable_transformer(parser)
    parser = _config_litept_deformable_transformer(parser)
    parser = _config_roomformer_criterion(parser)

    # Some training hyperparameters
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument("--lr_litept_mlp_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--lr_drop", default=[400], type=list)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Dataset
    parser.add_argument("--dataset_name", default="stru3d")
    parser.add_argument(
        "--dataset_root",
        default="data/stru3d_processed",
        help="dataset root folder, in which include train, test, val, annotation.json",
    )
    parser.add_argument("--eval_set", default="test", type=str)

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/deform_litept_stru3d.pth",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_stru3d",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--num_workers", default=2, type=int)

    # visualization options
    parser.add_argument(
        "--plot_pred", default=True, type=bool, help="plot predicted floorplan"
    )
    parser.add_argument(
        "--plot_density",
        default=True,
        type=bool,
        help="plot predicited room polygons overlaid on the density map",
    )
    parser.add_argument(
        "--plot_gt", default=True, type=bool, help="plot ground truth floorplan"
    )
    # parser.add_argument(
    #     "--dry_run",
    #     default=False,
    #     action="store_true",
    #     help="Experiment mode, will run for 1 epoch (nothing created)",
    # )

    return parser


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    # scene_ids = [x["image_id"] for x in batch]
    # samples = [x["image"] for x in batch]
    # gt_instances = [x["instances"] for x in batch]

    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        cocos_keys = [
            "image_id",
            "image",
            "instances",
            "file_name",
            "height",
            "width",
            "annotations",
        ]
        collate_batch = {}
        for key in batch[0]:
            if key in cocos_keys:
                collate_batch[key] = [x[key] for x in batch]
            # elif key in "point_cloud":
            #     collate_batch["coord"] = collate_fn([d[key] for d in batch])
            # elif key in "colors":
            #     collate_batch["color"] = collate_fn([d[key] for d in batch])
            # elif key in "normals":
            #     collate_batch["normal"] = collate_fn([d[key] for d in batch])
            else:
                collate_batch[key] = (
                    collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
        return collate_batch
    else:
        return default_collate(batch)


def main(args):
    print(f"git:\n {utils.get_sha()}\n")

    print(args)
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args, train=False)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # build dataset and dataloader
    dataset_eval = build_dataset(image_set=args.eval_set, args=args)
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    data_loader_eval = DataLoader(
        dataset_eval,
        args.batch_size,
        sampler=sampler_eval,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    for n, p in model.named_parameters():
        print(n)

    output_dir = Path(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=False
    )
    unexpected_keys = [
        k for k in unexpected_keys if not (k.endswith(("total_params", "total_ops")))
    ]
    if len(missing_keys) > 0:
        print(f"Missing Keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected Keys: {unexpected_keys}")

    save_dir = os.path.join(os.path.dirname(args.checkpoint), output_dir)
    evaluate_floor(
        model,
        args.dataset_name,
        data_loader_eval,
        device,
        save_dir,
        plot_pred=args.plot_pred,
        plot_density=args.plot_density,
        plot_gt=args.plot_gt,
        semantic_rich=args.semantic_classes > 0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable LitePT evaluation script", parents=[config()]
    )
    args = parser.parse_args()

    main(args)
