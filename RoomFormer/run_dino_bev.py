# Purpose: Precalculate DINO_BEV output for faster training
# since DINOv3 is frozen, it is not required to be called
# multiple time during the training. Therefore, it is reasonable
# to precalculate the output for faster training and testing.abs
#
# Author: Hai Chu

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import build_mixed_dataset as build_dataset
from models.dino_bev import build as build_model
from models.dino_bev import extract_patch_grid, make_transform
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from util.poly_ops import pad_gt_polys

FACES = sorted(["U", "F", "R", "L", "B", "D"])


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


def config():
    a = argparse.ArgumentParser(description="Calculate DINO-BEV output")
    # dataset parameters
    a.add_argument("--dataset_name", default="stru3d")
    a.add_argument("--dataset_root", default="data/stru3d_processed", type=str)
    a.add_argument(
        "--semantic_classes",
        default=-1,
        type=int,
        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty",
    )
    a.add_argument("--batch_size", default=10, type=int)
    a.add_argument(
        "-d", "--dry_run", default=False, action="store_true", help="test run"
    )
    a.add_argument(
        "-v", "--verbose", default=False, action="store_true", help="print log if true"
    )
    a.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    a.add_argument("--num_workers", default=2, type=int)

    a.add_argument("--seed", default=42, type=int)
    a.add_argument("--dinov3_repo", help="root directory of dinov3")
    a.add_argument("--dinov3_checkpoint", help="checkpoint directory of dinov3")
    a.add_argument(
        "--dinov3_n_last_layers",
        default=1,
        help="number of last layers to be extracted from dinov3 backbone",
        type=int,
    )
    a.add_argument(
        "--dinov3_model", default="dinov3_vits16", help="dinov3 exact model name"
    )

    # dino-bev
    a.add_argument("--pca_outdim", default=64, help="PCA reduction dimension", type=int)

    a.add_argument(
        "--dino_bev_aggregation",
        default="average",
        help="aggregation strategy for multiple features 3D points",
    )

    a.add_argument(
        "--DINO_BEV",
        default=False,
        action="store_true",
        help="Include if DINO_BEV output is precalculated",
    )
    args = a.parse_args()
    return args


def build_batch_collator(mode="model"):
    def batch_collator(batch):
        """
        A batch collator that does something.
        """
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
            batched_room_faces,
            batched_room_depths,
            batched_point_clouds,
            batched_prj_points,
            batched_masks,
        )

    def pca_batch_collator(batch):
        """
        batch collator for pca dedicated dataloader
        """
        return torch.cat(batch, dim=0)

    if mode == "model":
        return batch_collator
    else:
        return pca_batch_collator


def _batch_to(batch, device="cpu"):
    (
        batched_room_faces,
        batched_room_depths,
        batched_point_clouds,
        batched_prj_points,
        batched_masks,
    ) = batch
    batched_room_faces = [
        faces.to(device, non_blocking=True) for faces in batched_room_faces
    ]
    batched_room_depths = [
        faces.to(device, non_blocking=True) for faces in batched_room_depths
    ]
    batched_point_clouds = [
        faces.to(device, non_blocking=True) for faces in batched_point_clouds
    ]

    for scene_idx in range(len(batched_prj_points)):
        for room_idx in range(len(batched_prj_points[scene_idx])):
            for face_idx in range(len(FACES)):
                batched_prj_points[scene_idx][room_idx][face_idx] = batched_prj_points[
                    scene_idx
                ][room_idx][face_idx].to(device, non_blocking=True)

                batched_masks[scene_idx][room_idx][face_idx] = batched_masks[scene_idx][
                    room_idx
                ][face_idx].to(device, non_blocking=True)

    return (
        batched_room_faces,
        batched_room_depths,
        batched_point_clouds,
        batched_prj_points,
        batched_masks,
    )


def main(args):
    model, pca = build_model(args)
    if args.verbose or args.dry_run:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable parameters: {n_parameters}")
        for n, p in model.named_parameters():
            param_state = "[Active]" if p.requires_grad else ""
            print(f"{param_state} {n}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = args.device
    model.to(device)
    pca.to(device)

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_test = build_dataset(image_set="test", args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    dataset_pca = DatasetWrapper(dataset_train, model.DINO, device)

    # sampler_train = RandomSampler(dataset_train)
    # sampler_val = SequentialSampler(dataset_val)
    sampler_pca = RandomSampler(dataset_pca)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True
    # )
    batch_sampler_pca = torch.utils.data.BatchSampler(
        sampler_pca, args.batch_size, drop_last=True
    )

    batch_collator = build_batch_collator("model")
    pca_batch_collator = build_batch_collator("pca")

    # data_loader_train = DataLoader(
    #     dataset_train,
    #     batch_sampler=batch_sampler_train,
    #     collate_fn=batch_collator,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )
    # data_loader_val = DataLoader(
    #     dataset_val,
    #     args.batch_size,
    #     sampler=sampler_val,
    #     drop_last=False,
    #     collate_fn=batch_collator,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    data_loader_pca = DataLoader(
        dataset_pca, batch_sampler=batch_sampler_pca, collate_fn=pca_batch_collator
    )

    print("\nfitting PCA")
    start = time.time()
    pca.fit(data_loader_pca)
    end = time.time() - start
    if args.verbose or args.dry_run:
        print(f"\telapsed time: {end:.4f}s")

    torch.save(pca.state_dict(), "checkpoints/pca.pth")
    pca.to(device)

    output_dir = Path(args.dataset_root)
    print("\ncalculating DINO_BEV output")
    for dataset, mode in zip(
        [dataset_train, dataset_test, dataset_val], ["train", "test", "val"]
    ):
        scene_ids = dataset.scene_ids
        for idx in range(len(scene_ids)):
            scene_id = scene_ids[idx]
            if args.verbose or args.dry_run:
                print(f"\nprocessing {scene_id}")

            record = {}
            start = time.time()
            point_cloud, _, idxs = dataset._load_point_cloud(scene_id)
            record["point_cloud"] = point_cloud
            prj_points, masks = dataset._load_mask_prj_points(scene_id, idxs)
            record["project_points"] = prj_points
            record["masks"] = masks
            record["cubes"] = dataset._load_cubes(scene_id)
            record["depths"] = dataset._load_depths(scene_id)

            (
                batched_room_faces,
                batched_room_depths,
                batched_point_clouds,
                batched_prj_points,
                batched_masks,
            ) = _batch_to(batch_collator([record]), device=device)
            scene_bev, scene_mask, scene_cnt, scene_agree = model(
                batched_room_faces,
                batched_masks,
                batched_point_clouds,
                batched_prj_points,
            )
            output_path = output_dir / mode / scene_id
            if args.verbose or args.dry_run:
                print(f"\tsaving outputs to {output_path}")
            if not args.dry_run:
                # save using torch.save
                torch.save(scene_bev.squeeze(), output_path / "scene_bev.pt")
                torch.save(scene_mask.squeeze(), output_path / "scene_mask.pt")
                torch.save(scene_cnt.squeeze(), output_path / "scene_cnt.pt")
                torch.save(scene_agree.squeeze(), output_path / "scene_agree.pt")
            end = time.time() - start
            if args.verbose or args.dry_run:
                print(f"\telapsed time: {end:.4f}s")


if __name__ == "__main__":
    main(config())
