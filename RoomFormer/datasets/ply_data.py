# ---------------------------------------------------------------------------------
# Modified from Edge Prediction for Roof Wireframe Reconstruction with Transformers
# https://github.com/ghanning/S23DR2026/blob/main/s23dr/nn/dataset.py

# Now I have to adapt this to COCO style, or else the whole model doesn't work :((
#
# Hai Chu
# ---------------------------------------------------------------------------------

import os
import json
from typing import Tuple

import numpy as np
import torch
from plyfile import PlyData
from pycocotools.coco import COCO
from scipy.spatial.transform import Rotation
from PIL import Image

from util.poly_ops import resort_corners
from detectron2.data import transforms as T
from detectron2.data.detection_utils import (
    annotations_to_instances,
    transform_instance_annotations,
)
from detectron2.structures import BoxMode


# Each folder consists of:
# |- train/test/val
#   |- scene_<scene_id>
#     |- density.png
#     |- point_cloud.ply
class PlyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        num_points: int,
        transforms,
        aug_rotate: bool = False,
        aug_flip: bool = False,
        semantic_classes: int = -1,
        mode="train",
    ):
        assert os.path.exists(os.path.abspath(data_dir)), "data folder does not exist"
        assert mode in ["train", "test", "val"], (
            "mode should be one of (train, test, val), default=train"
        )
        super(PlyDataset, self).__init__()
        self.mode = mode
        self.data_dir = os.path.abspath(data_dir)
        self.data_root = os.path.join(self.data_dir, mode)
        self.scene_ids = sorted(os.listdir(self.data_root))
        self.num_points = num_points

        self.aug_rotate = aug_rotate
        self.aug_flip = aug_flip
        self.semantic_classes = semantic_classes

        ann_file = os.path.join(self.data_dir, "annotations", f"{self.mode}.json")
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self._transforms = transforms
        self.prepare = ConvertToCocoDict(self.data_root, self._transforms)

    def __len__(self):
        return len(self.scene_ids)

    def _get_image(self, path):
        return Image.open(os.path.join(self.data_root, path))

    def _load_point_cloud(self, scene_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scene_id in self.scene_ids, "scene_id not found"
        ply_path = os.path.join(self.data_root, scene_id, "point_cloud.ply")
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        xyz = torch.from_numpy(
            np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        ).float()

        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
        features = colors_to_features(colors)

        return xyz, features

    def __getitem__(self, index):
        scene_id = self.scene_ids[index]

        # Load point cloud
        # for debugging purpose
        view_transform = torch.eye(3, dtype=torch.float32)

        xyz, features = self._load_point_cloud(scene_id)

        if self.aug_rotate:
            R = Rotation.from_euler("y", np.random.rand() * 360, degrees=True)
            R = torch.from_numpy(R.as_matrix()).float()
            xyz = xyz @ R.T
            view_transform = R @ view_transform

        if self.aug_flip and np.random.rand() < 0.5:
            xyz[:, 0] = -xyz[:, 0]
            flip = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32))
            view_transform = flip @ view_transform

        min_xyz, max_xyz, q_diff = compute_norm_params(xyz)
        xyz = (xyz - min_xyz) / q_diff
        idx = sample_indices(xyz, scene_id, self.num_points)
        xyz = xyz[idx]
        features = features[idx]

        item = {
            "xyz": xyz.float(),  # [num_pts, 3]
            "features": features.float(),  # [num_pts, 3] # RGB
            "min_xyz": min_xyz.float(),  # [3]
            "max_xyz": max_xyz.float(),  # [3]
        }
        # Load density map
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        if self.semantic_classes == -1:
            target = [t for t in target if t["category_id"] not in [16, 17]]

        path = coco.loadImgs(img_id)[0]["file_name"]

        record = self.prepare(img_id, path, target)
        record["point_cloud"] = item

        return record


def collate_fn(batch: list[dict]) -> dict:
    collated = {
        "xyz": torch.stack([item["xyz"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "min_xyz": torch.stack([item["min_xyz"] for item in batch]),
        "max_xyz": torch.stack([item["max_xyz"] for item in batch]),
    }

    return collated


def compute_norm_params(
    xyz: torch.Tensor, pad: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_xyz = xyz.min(dim=0).values - pad
    max_xyz = xyz.max(dim=0).values + pad
    q_diff = max_xyz - min_xyz + 1e-6
    return min_xyz, max_xyz, q_diff


def sample_indices(xyz: torch.Tensor, name: str, num_points: int) -> torch.Tensor:
    if len(xyz) == 0:
        raise ValueError(f"No points found in {name} point cloud.")

    replace = len(xyz) < num_points
    if replace:
        return torch.randint(0, len(xyz), (num_points,))
    return torch.randperm(len(xyz))[:num_points]


def sample_unique_indices(
    xyz: torch.Tensor, name: str, num_points: int
) -> torch.Tensor:
    if num_points <= 0:
        return torch.empty(0, dtype=torch.long)
    if num_points >= len(xyz):
        return torch.arange(len(xyz), dtype=torch.long)
    return sample_indices(xyz, name, num_points)


def colors_to_features(rgb) -> torch.Tensor:
    assert rgb is not None, "RGB data not available"

    features = np.asarray(rgb, dtype=np.float32) / 255.0
    features = torch.from_numpy(features).float()

    return features


class ConvertToCocoDict(object):
    def __init__(self, root, augmentations):
        self.root = root
        self.augmentations = augmentations

    def __call__(self, img_id, path, target):

        file_name = os.path.join(self.root, path)

        img = np.array(Image.open(file_name))
        w, h = img.shape

        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record["image_id"] = img_id

        for obj in target:
            obj["bbox_mode"] = BoxMode.XYWH_ABS

        record["annotations"] = target

        if self.augmentations is None:
            record["image"] = (1 / 255) * torch.as_tensor(
                np.ascontiguousarray(np.expand_dims(img, 0))
            )
            record["instances"] = annotations_to_instances(
                target, (h, w), mask_format="polygon"
            )
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record["image"] = (1 / 255) * torch.as_tensor(
                np.array(np.expand_dims(image, 0))
            )

            annos = [
                transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in record.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # resort corners after augmentation: so that all corners start from upper-left counterclockwise
            for anno in annos:
                anno["segmentation"][0] = resort_corners(anno["segmentation"][0])

            record["instances"] = annotations_to_instances(
                annos, (h, w), mask_format="polygon"
            )

        return record


def make_poly_transforms(image_set):

    if image_set == "train":
        return T.AugmentationList(
            [
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomRotation(
                    [0.0, 90.0, 180.0, 270.0],
                    expand=False,
                    center=None,
                    sample_style="choice",
                ),
            ]
        )

    if image_set == "val" or image_set == "test":
        return None

    raise ValueError(f"unknown {image_set}")


def build(mode, args):
    assert os.path.exists(os.path.abspath(args.dataset_root)), (
        f"{args.dataset_root} does not exist"
    )
    dataset_root = os.path.abspath(args.dataset_root)

    dataset = PlyDataset(
        dataset_root,
        num_points=args.num_points,
        transforms=make_poly_transforms(mode),
        aug_rotate=True,
        aug_flip=True,
        semantic_classes=args.semantic_classes,
        mode=mode,
    )

    return dataset
