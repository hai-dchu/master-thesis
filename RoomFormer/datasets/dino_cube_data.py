# ---------------------------------------------------------------------------------
# Modified from Edge Prediction for Roof Wireframe Reconstruction with Transformers
# https://github.com/ghanning/S23DR2026/blob/main/s23dr/nn/dataset.py

# Now I have to adapt this to COCO style, or else the whole model doesn't work :((
#
# Hai Chu
# ---------------------------------------------------------------------------------

import os

import numpy as np
import torch
from detectron2.data import transforms as T
from detectron2.data.detection_utils import (
    annotations_to_instances,
    transform_instance_annotations,
)
from detectron2.structures import BoxMode
from PIL import Image
from plyfile import PlyData
from pycocotools.coco import COCO
from util.poly_ops import resort_corners

FACES = ["U", "F", "R", "L", "B", "D"]


# Each folder consists of:
# |- train/test/val
#   |- scene_<scene_id>
#     |- <room_id>
#       |- img_<orientation>.png for orientation in (U, D, R, L, F, B)
#       |- dep_<orientation>.png for orientation in (U, D, R, L, F, B)
#       |- mask_<orientation>.npy for orientation in (U, D, R, L, F, B)
#     |- density.png
#     |- point_cloud.ply
class CubePolyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
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
        super().__init__()
        self.mode = mode
        self.data_dir = os.path.abspath(data_dir)
        self.data_root = os.path.join(self.data_dir, mode)

        self.aug_rotate = aug_rotate
        self.aug_flip = aug_flip
        self.semantic_classes = semantic_classes

        ann_file = os.path.join(self.data_dir, "annotations", f"{self.mode}.json")
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

        self._transforms = transforms
        self.prepare = ConvertToCocoDict(self.data_root, self._transforms)

        # TODO: Fix dataset installation, since the current dataset (stru3d_processed) has missing scenes
        self.scene_ids = [
            self.coco.imgs[i]["file_name"].split("/")[0] for i in self.coco.imgs.keys()
        ]  # sorted(os.listdir(self.data_root))

    def __len__(self):
        return len(self.scene_ids)

    def _get_image(self, path):
        return Image.open(os.path.join(self.data_root, path))

    def _load_point_cloud(self, scene_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert scene_id in self.scene_ids, "scene_id not found"
        ply_path = os.path.join(self.data_root, scene_id, "point_cloud.ply")
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        xyz = torch.from_numpy(
            np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        ).float()

        colors = torch.from_numpy(
            np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
        )

        # sample every 100th point
        # TODO: change this into a hyperparameter
        idxs = [i for i in range(0, len(xyz), 1)]
        xyz = xyz[idxs]
        colors = colors[idxs]

        return xyz, colors, idxs

    def _load_masks(self, scene_id: str, idxs: list[int]):
        assert scene_id in self.scene_ids, "scene_id not found"
        rooms = os.listdir(os.path.join(self.data_root, scene_id))
        masks = {}
        for room in sorted(rooms):
            path = os.path.join(self.data_root, scene_id, room)
            if not os.path.isdir(path):
                continue
            tmp = {}
            for face in FACES:
                tmp[face] = torch.from_numpy(
                    np.load(os.path.join(path, f"mask_{face}.npy"))[idxs]
                )
            masks[room] = tmp
        return masks

    def _load_cubes(self, scene_id: str):
        assert scene_id in self.scene_ids, "scene_id not found"
        rooms = os.listdir(os.path.join(self.data_root, scene_id))
        faces = {}
        for room in sorted(rooms):
            path = os.path.join(self.data_root, scene_id, room)
            if not os.path.isdir(path):
                continue
            tmp = {}
            for face in FACES:
                tmp[face] = torch.from_numpy(
                    np.array(Image.open(os.path.join(path, f"img_{face}.png")))
                )
            faces[room] = tmp

        return faces

    def _load_depths(self, scene_id: str):
        assert scene_id in self.scene_ids, "scene_id not found"
        rooms = os.listdir(os.path.join(self.data_root, scene_id))
        depths = {}
        for room in sorted(rooms):
            path = os.path.join(self.data_root, scene_id, room)
            if not os.path.isdir(path):
                continue
            tmp = {}
            for face in FACES:
                tmp[face] = torch.from_numpy(
                    np.array(Image.open(os.path.join(path, f"dep_{face}.png")))
                )
            depths[room] = tmp

        return depths

    def _load_prj_points(self, scene_id: str):
        assert scene_id in self.scene_ids, "scene_id not found"
        rooms = os.listdir(os.path.join(self.data_root, scene_id))
        points = {}
        for room in sorted(rooms):
            path = os.path.join(self.data_root, scene_id, room)
            if not os.path.isdir(path):
                continue
            tmp = {}
            for face in FACES:
                tmp[face] = torch.from_numpy(
                    np.load(os.path.join(path, f"prj_points_{face}.npy"))
                )

            points[room] = tmp

        return points

    def _load_mask_prj_points(self, scene_id: str, idxs: list[int]):
        assert scene_id in self.scene_ids, "scene_id not found"
        rooms = os.listdir(os.path.join(self.data_root, scene_id))
        masks = {}
        points = {}
        for room in sorted(rooms):
            path = os.path.join(self.data_root, scene_id, room)
            if not os.path.isdir(path):
                continue
            tmp_masks = {}
            tmp_points = {}
            for face in FACES:
                # tmp_mask[face]
                mask = torch.from_numpy(np.load(os.path.join(path, f"mask_{face}.npy")))
                old_mask = torch.where(mask > 0)
                plh = torch.zeros_like(mask)
                plh[idxs] = 1
                keep_points = plh[old_mask]
                tmp_masks[face] = mask[idxs]
                tmp_points[face] = torch.from_numpy(
                    np.load(os.path.join(path, f"prj_points_{face}.npy"))
                )[keep_points]
            masks[room] = tmp_masks
            points[room] = tmp_points
        return points, masks

    def _point_cloud_augmentation(
        self,
        point_cloud: torch.Tensor,
        horizontal: bool = False,
        vertical: bool = False,
        rotate: float = 0.0,
    ):
        if horizontal:
            point_cloud[:, 0] = -point_cloud[:, 0]
        if vertical:
            point_cloud[:, 1] = -point_cloud[:, 1]
        if rotate > 0:
            # rotate in deg
            rad = rotate / 180.0 * np.pi
            c, s = np.cos(rad), np.sin(rad)
            rot = torch.Tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]) # maybe not this?

            # The point cloud is centered around camera center
            point_cloud = torch.mm(point_cloud, rot.T)
            
            # suppress floating point errors
            point_cloud[point_cloud.abs() < 1e-12] = 0 

        return point_cloud

    def __getitem__(self, index):
        """
        Each item return consists of:
        - COCO object for density map
        - Faces: cube map faces of shape (6) (U, D, R, L, F, B) from panorama
        - Masks: dict of shape (point_cloud_size, ), keys in (U, D, R, L, F, B) containing which (3D) points are visible in each face
        """
        # Load density map
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        if self.semantic_classes == -1:
            target = [t for t in target if t["category_id"] not in [16, 17]]

        path = coco.loadImgs(img_id)[0]["file_name"]

        # get random rotations and flip
        _hor = np.random.randn() > 0.5
        _ver = np.random.randn() > 0.5
        _rotate = np.random.choice([0.0, 90.0, 180.0, 270.0])
        record = self.prepare(
            img_id, path, target, horizontal=_hor, vertical=_ver, rotate=_rotate
        )

        scene_id = self.scene_ids[index]
        point_cloud, _, idxs = self._load_point_cloud(scene_id)
        record["point_cloud"] = self._point_cloud_augmentation(
            point_cloud, horizontal=_hor, vertical=_ver, rotate=_rotate
        )
        # record["masks"] = self._load_masks(scene_id, idxs)
        # record["project_points"] = self._load_prj_points(scene_id)
        prj_points, masks = self._load_mask_prj_points(scene_id, idxs)
        record["project_points"] = prj_points
        record["masks"] = masks
        record["cubes"] = self._load_cubes(scene_id)
        record["depths"] = self._load_depths(scene_id)

        return record


class ConvertToCocoDict:
    def __init__(
        self,
        root,
        augmentations,
    ):
        self.root = root
        self.augmentations = augmentations

    def __call__(
        self,
        img_id,
        path,
        target,
        horizontal: bool = False,
        vertical: bool = False,
        rotate: float = 0,
    ):
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
            aug_list = self.augmentations(
                w,
                h,
                horizontal=horizontal,
                vertical=vertical,
                rotate=rotate,
            )
            transforms = aug_list(aug_input)
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


# TODO: Replace the whole transform pipeline to also rotate and flip the point cloud
# So the idea is to record the set of transformation returned from AugmentationList
# which include several booleans and angles:
# - horizontal flip
# - vertical flip
# - rotation (0, 90, 180, 270)


def _random_transform_wrapper(
    img_width: int = 256,
    img_height: int = 256,
    horizontal: bool = False,
    vertical: bool = False,
    rotate: float = 0,
):
    hor = T.NoOpTransform()
    ver = T.NoOpTransform()
    rot = T.NoOpTransform()

    if horizontal:
        hor = T.HFlipTransform(img_width)

    if vertical:
        hor = T.VFlipTransform(img_height)

    if rotate > 0:
        rot = T.RotationTransform(
            img_width, img_height, rotate, expand=False, center=None
        )

    return T.AugmentationList([hor, ver, rot])


def make_poly_transforms(image_set):
    if image_set == "train":
        # return None
        # return T.AugmentationList(
        #     [
        #         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        #         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        #         T.RandomRotation(
        #             [0.0, 90.0, 180.0, 270.0],
        #             expand=False,
        #             center=None,
        #             sample_style="choice",
        #         ),
        #     ]
        # )
        return _random_transform_wrapper

    if image_set == "val" or image_set == "test":
        return None

    raise ValueError(f"unknown {image_set}")


def build(mode, args):
    assert os.path.exists(os.path.abspath(args.dataset_root)), (
        f"{args.dataset_root} does not exist"
    )
    dataset_root = os.path.abspath(args.dataset_root)

    dataset = CubePolyDataset(
        dataset_root,
        transforms=make_poly_transforms(mode),
        aug_rotate=False,
        aug_flip=False,
        semantic_classes=args.semantic_classes,
        mode=mode,
    )

    return dataset
