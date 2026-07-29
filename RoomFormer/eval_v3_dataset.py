import argparse

# import datetime
# import json
import os
import random

# import time
from pathlib import Path

import numpy as np
import torch

# import util.misc as utils
from datasets import build_mixed_dataset as build_dataset
from engine_v3 import evaluate_floor
from models import build_model_v0 as build_model
from torch.utils.data import DataLoader
from util.poly_ops import pad_gt_polys

FACES = sorted(["U", "F", "R", "L", "B", "D"])


def get_args_parser():
    parser = argparse.ArgumentParser("RoomFormer", add_help=False)
    parser.add_argument("--batch_size", default=10, type=int)

    # backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument("--lr_backbone", default=0, type=float)
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

    # aux
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_true",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # dataset parameters
    parser.add_argument("--dataset_name", default="stru3d")
    parser.add_argument("--dataset_root", default="data/stru3d", type=str)
    parser.add_argument("--eval_set", default="test", type=str)

    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/roomformer_scenecad.pth",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--output_dir", default="eval_stru3d", help="path where to save result"
    )

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

    return parser


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
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

    def batch_collator(batch):
        """
        A batch collator that does something.
        """
        scene_ids = [x["image_id"] for x in batch]
        samples = [x["image"] for x in batch]
        gt_instances = [x["instances"] for x in batch]
        room_targets = pad_gt_polys(
            gt_instances, model.num_queries_per_poly, device="cpu"
        )

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

    data_loader_eval = DataLoader(
        dataset_eval,
        args.batch_size,
        sampler=sampler_eval,
        drop_last=False,
        collate_fn=batch_collator,
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
        "RoomFormer evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    main(args)
