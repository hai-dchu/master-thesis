import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import util.misc as utils
import wandb
from datasets import build_mixed_dataset as build_dataset
from engine_v3 import evaluate, train_one_epoch
from models import build_model_v3 as build_model
from models.dino_bev import (
    extract_patch_grid,
    make_transform,
)
from torch.utils.data import DataLoader, Subset
from util.poly_ops import pad_gt_polys

FACES = sorted(["U", "F", "R", "L", "B", "D"])


def get_args_parser():
    parser = argparse.ArgumentParser("RoomFormer", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--lr_drop", default=[400], type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--sgd", action="store_true")

    # backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries",
        default=800,
        type=int,
        help="Number of query slots (num_polys * max. number of corner per poly)",
    )
    parser.add_argument(
        "--num_polys",
        default=20,
        type=int,
        help="Number of maximum number of room polygons",
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument(
        "--query_pos_type",
        default="sine",
        type=str,
        choices=("static", "sine", "none"),
        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos",
    )
    parser.add_argument(
        "--with_poly_refine",
        default=True,
        action="store_true",
        help="iteratively refine reference points (i.e. positional part of polygon queries)",
    )
    parser.add_argument(
        "--masked_attn",
        default=False,
        action="store_true",
        help="if true, the query in one room will not be allowed to attend other room",
    )
    parser.add_argument(
        "--semantic_classes",
        default=-1,
        type=int,
        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty",
    )

    # loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_true",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_coords",
        default=5,
        type=float,
        help="L1 coords coefficient in the matching cost",
    )

    # loss coefficients
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--room_cls_loss_coef", default=0.2, type=float)
    parser.add_argument("--coords_loss_coef", default=5, type=float)
    parser.add_argument("--raster_loss_coef", default=1, type=float)

    # dataset parameters
    parser.add_argument("--dataset_name", default="stru3d")
    parser.add_argument("--dataset_root", default="data/stru3d_processed", type=str)

    parser.add_argument(
        "--output_dir",
        default="output-v1",
        help="path where to save, empty for no saving",
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
    parser.add_argument("--job_name", default="train_stru3d", type=str)

    # log monitor wandb
    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="if true, init and log to wandb",
    )

    # dinov3
    parser.add_argument("--dinov3_repo", help="root directory of dinov3")
    parser.add_argument("--dinov3_checkpoint", help="checkpoint directory of dinov3")
    parser.add_argument(
        "--dinov3_n_last_layers",
        default=1,
        help="number of last layers to be extracted from dinov3 backbone",
        type=int,
    )
    parser.add_argument(
        "--dinov3_model", default="dinov3_vits16", help="dinov3 exact model name"
    )

    # dino-bev
    parser.add_argument(
        "--pca_outdim", default=64, help="PCA reduction dimension", type=int
    )

    parser.add_argument("--pca", default="checkpoints/pca.pth")

    parser.add_argument(
        "--lr_dino_multilayer_proj",
        default=1e-3,
        help="Learning rate for projection layer between DINO_BEV and ResNet output",
        type=float,
    )

    parser.add_argument(
        "--dino_bev_aggregation",
        default="average",
        help="aggregation strategy for multiple features 3D points",
    )

    # For experimenting
    parser.add_argument(
        "--subset_length",
        default=-1,
        help="If subset_length > 0, train on subset_length samples instead of the full dataset",
        type=int,
    )

    parser.add_argument(
        "--DINO_BEV",
        default=False,
        action="store_true",
        help="Include if DINO_BEV output is precalculated",
    )

    return parser


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self, parent: torch.utils.data.Dataset, dino: torch.nn.Module, device: str
    ):
        super().__init__()
        self.parent = parent
        self.model = dino
        self.device = device
        self.transform = make_transform()

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        sample = self.parent.__getitem__(idx)
        rooms = sample["cubes"].keys()

        room_faces = []
        for room in rooms:
            for face in FACES:
                room_faces.append(
                    torch.moveaxis(sample["cubes"][room][face], -1, 0).to(self.device)
                )

        room_faces = torch.stack(room_faces)
        patches = extract_patch_grid(
            self.model, self.transform(room_faces)
        )  # .repeat(1, 1, 16, 16)
        return patches


def build_batch_collator(args, mode="model", num_queries_per_poly=None):
    if mode == "model":
        assert num_queries_per_poly is not None

    assert not (mode == "pca" and args.DINO_BEV), (
        "if args.DINO_BEV is included (i.e. DINO_BEV output is precalculated, then it is redundant to have pca_batch_collator)"
    )

    def batch_collator(batch):
        """
        A batch collator that does something.
        """
        scene_ids = [x["image_id"] for x in batch]
        samples = [x["image"] for x in batch]
        gt_instances = [x["instances"] for x in batch]
        room_targets = pad_gt_polys(gt_instances, num_queries_per_poly, device="cpu")

        batched_room_faces = []  # [[x['cubes'][room] for room in x['cubes']] for x in batch]
        batched_room_depths = []
        batched_point_clouds = []
        batched_prj_points = []
        batched_masks = []

        for x in batch:
            rooms = sorted(x["cubes"].keys())
            tmp_cube = []
            tmp_depth = []
            tmp_prj_points = []
            tmp_masks = []
            for room in rooms:
                room_face = []
                room_depth = []
                room_prj_points = []
                room_mask = []
                for face in FACES:
                    room_face.append(torch.moveaxis(x["cubes"][room][face], -1, 0))
                    room_depth.append(torch.moveaxis(x["depths"][room][face], -1, 0))
                    room_prj_points.append(x["project_points"][room][face])
                    room_mask.append(x["masks"][room][face])
                tmp_cube.append(torch.stack(room_face))
                tmp_depth.append(torch.stack(room_depth))
                tmp_prj_points.append(room_prj_points)
                tmp_masks.append(room_mask)

            batched_room_faces.append(
                torch.flatten(torch.stack(tmp_cube), start_dim=0, end_dim=1)
            )
            batched_room_depths.append(
                torch.flatten(torch.stack(tmp_depth), start_dim=0, end_dim=1)
            )
            batched_prj_points.append(tmp_prj_points)
            batched_masks.append(tmp_masks)
            batched_point_clouds.append(x["point_cloud"])

        return (
            scene_ids,
            samples,
            gt_instances,
            room_targets,
            batched_room_faces,
            batched_room_depths,
            batched_point_clouds,
            batched_prj_points,
            batched_masks,
        )

    def DINO_BEV_bc(batch):
        """
        A batch collator that does something.
        """
        scene_ids = [x["image_id"] for x in batch]
        samples = [x["image"] for x in batch]
        gt_instances = [x["instances"] for x in batch]
        room_targets = pad_gt_polys(gt_instances, num_queries_per_poly, device="cpu")

        batched_scene_bevs = torch.stack([x["bev"] for x in batch])
        batched_scene_masks = torch.stack([x["mask"] for x in batch])
        batched_scene_cnts = torch.stack([x["cnt"] for x in batch])
        batched_scene_agrees = torch.stack([x["agree"] for x in batch])

        return (
            scene_ids,
            samples,
            gt_instances,
            room_targets,
            batched_scene_bevs,
            batched_scene_masks,
            batched_scene_cnts,
            batched_scene_agrees,
        )

    def pca_batch_collator(batch):
        """
        batch collator for pca dedicated dataloader
        """
        return torch.cat(batch, dim=0)

    if mode == "model":
        if args.DINO_BEV:
            return DINO_BEV_bc
        return batch_collator
    else:
        return pca_batch_collator


def main(args):
    print(f"git:\n  {utils.get_sha()}\n")

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
    model, pca, criterion = build_model(args)
    model.to(device)

    # dino_bev, pca = build_dino_bev(args)
    # dino_bev.to(device)
    # dino_bev.requires_grad_(False)

    # model = DINORoomFormer(model, dino_bev).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # build dataset and dataloader
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

    batch_collator = build_batch_collator(
        args, mode="model", num_queries_per_poly=model.num_queries_per_poly
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=batch_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=batch_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if not args.DINO_BEV:
        pca_batch_collator = build_batch_collator(mode="pca")
        dataset_pca = DatasetWrapper(
            dataset_train, model.dino_multilayer.dino_bev.DINO, device
        )
        sampler_pca = torch.utils.data.RandomSampler(dataset_pca)
        batch_sampler_pca = torch.utils.data.BatchSampler(
            sampler_pca, args.batch_size, drop_last=True
        )
        data_loader_pca = DataLoader(
            dataset_pca, batch_sampler=batch_sampler_pca, collate_fn=pca_batch_collator
        )

        print("Fitting PCA")
        pca.fit(data_loader_pca)
        # torch.save(pca.state_dict(), args.pca)
        pca.to(device)
    else:
        state = torch.load(args.pca)
        pca.components = state["components"]
        pca.mean = state["mean"]
        pca.is_fitted = True

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model.named_parameters():
        param_state = "[Active]" if p.requires_grad else ""
        print(f"{param_state} {n}")

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, ["dino_multilayer.input_proj"])
                and not match_name_keywords(n, ["dino_scales"])
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
        # higher learning rate for multilayer DINO_BEV wrapper (Conv2d + GroupNorm) to match ResNet multilayered output
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["dino_multilayer.input_proj"])
                and p.requires_grad
            ],
            "lr": args.lr_dino_multilayer_proj,
        },
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith(("total_params", "total_ops")))
        ]
        if len(missing_keys) > 0:
            print(f"Missing Keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected Keys: {unexpected_keys}")
        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy

            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = False
            if args.override_resumed_lr_drop:
                print(
                    "Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler."
                )
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = [
                    group["initial_lr"] for group in optimizer.param_groups
                ]
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1
        # check the resumed model
        test_stats = evaluate(
            model, criterion, args.dataset_name, data_loader_val, device
        )

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
            args.clip_max_norm,
        )
        lr_scheduler.step()
        if args.output_dir:
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

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "RoomFormer training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    now = datetime.datetime.now()  # noqa: DTZ005
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = run_id + "_" + args.job_name
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
