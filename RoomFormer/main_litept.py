import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.append("LitePT")


from collections.abc import Mapping, Sequence

import numpy as np
import torch
import util.misc as utils
import wandb
from datasets import build_normal_dataset as build_dataset
from engine_litept import evaluate, train_one_epoch
from models import build_model_litept as build_model
from torch.utils.data import DataLoader, Subset
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
    group.add_argument("--litept_enc_mode", default=False)

    # Grid size for voxelization
    group.add_argument("--litept_grid_size", default=256, type=int)

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
    return parser


def _config_deformable_decoder(parser):
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
    parser = _config_deformable_decoder(parser)
    parser = _config_litept_deformable_transformer(parser)
    parser = _config_roomformer_criterion(parser)

    # Some training hyperparameters
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument("--lr_litept_mlp_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
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

    parser.add_argument(
        "--output_dir", default="output", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--job_name", default="train", type=str)

    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="if added, initiate remote logging",
    )

    # For experimenting
    parser.add_argument(
        "--subset_length",
        default=-1,
        help="If subset_length > 0, train on subset_length samples instead of the full dataset",
        type=int,
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="Experiment mode, will run for 1 epoch (nothing created)",
    )

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

    # setup wandb for logging
    if args.wandb:
        utils.setup_wandb()
        wandb.init(project="RoomFormer")
        wandb.run.name = args.run_name

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.subset_length > 0:
        indices = range(args.subset_length)
        dataset_train = Subset(dataset_train, indices)
        dataset_val = Subset(dataset_val, indices)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, ["encoder"]) and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["encoder.backbone"]) and p.requires_grad
            ],
            "lr": args.lr_backbone if args.litept_checkpoint is not None else args.lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["encoder.mlp"]) and p.requires_grad
            ],
            "lr": args.lr * args.lr_litept_mlp_mult,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=2e-2,
    #     epochs=args.epochs,
    #     steps_per_epoch=len(data_loader_train),
    # )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [400])  # args.lr_drop)

    output_dir = Path(args.output_dir)

    # TODO: Write resume training weight here (not now)

    for n, p in model.named_parameters():
        param_state = "[Active]" if p.requires_grad else ""
        print(f"{param_state} {n}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            lr_scheduler,
            args.clip_max_norm,
        )
        # lr_scheduler.step()
        if args.output_dir and not args.dry_run:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 20 epochs
            if (epoch + 1) in args.lr_drop or (epoch + 1) % 20 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(
            model, criterion, args.dataset_name, data_loader_val, device
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.wandb:
            wandb.log({"epoch": epoch})
            wandb.log({"lr_rate": train_stats["lr"]})

        train_log_dict = {
            "train/loss": train_stats["loss"],
            "train/loss_ce": train_stats["loss_ce"],
            "train/loss_coords": train_stats["loss_coords"],
            "train/loss_coords_unscaled": train_stats["loss_coords_unscaled"],
            "train/cardinality_error": train_stats["cardinality_error_unscaled"],
        }

        val_log_dict = {
            "val/loss": test_stats["loss"],
            "val/loss_ce": test_stats["loss_ce"],
            "val/loss_coords": test_stats["loss_coords"],
            "val/loss_coords_unscaled": test_stats["loss_coords_unscaled"],
            "val/cardinality_error": test_stats["cardinality_error_unscaled"],
            "val_metrics/room_prec": test_stats["room_prec"],
            "val_metrics/room_rec": test_stats["room_rec"],
            "val_metrics/corner_prec": test_stats["corner_prec"],
            "val_metrics/corner_rec": test_stats["corner_rec"],
            "val_metrics/angles_prec": test_stats["angles_prec"],
            "val_metrics/angles_rec": test_stats["angles_rec"],
        }

        if args.semantic_classes > 0:
            # need to log additional metrics for semantically-rich floorplans
            train_log_dict["train/loss_ce_room"] = train_stats["loss_ce_room"]
            val_log_dict["val/loss_ce_room"] = test_stats["loss_ce_room"]
            val_log_dict["val_metrics/room_sem_prec"] = test_stats["room_sem_prec"]
            val_log_dict["val_metrics/room_sem_rec"] = test_stats["room_sem_rec"]
            val_log_dict["val_metrics/window_door_prec"] = test_stats[
                "window_door_prec"
            ]
            val_log_dict["val_metrics/window_door_rec"] = test_stats["window_door_rec"]

        else:
            # only apply the rasterization loss for non-semantic floorplans
            train_log_dict["train/loss_raster"] = train_stats["loss_raster"]
            val_log_dict["val/loss_raster"] = test_stats["loss_raster"]

        if "room_iou" in test_stats:
            val_log_dict["val_metrics/room_iou"] = test_stats["room_iou"]

        if args.wandb:
            wandb.log(train_log_dict)
            wandb.log(val_log_dict)

        if args.output_dir and not args.dry_run:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "LitePT Deformable Transformer training script", parents=[config()]
    )
    args = parser.parse_args()
    now = datetime.datetime.now()  # noqa: DTZ005
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = run_id + "_" + args.job_name
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    if args.output_dir and not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
