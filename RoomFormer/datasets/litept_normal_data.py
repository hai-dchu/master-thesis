# ------------------------------------------------------------
# Basically RoomFormer's original dataset but with point cloud
# Hai Chu
# ------------------------------------------------------------

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


class PointCloudNormalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms,
        point_transforms,
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
        self._point_transforms = point_transforms
        self.prepare = ConvertToCocoDict(self.data_root, self._transforms)

        # TODO: Fix dataset installation, since the current dataset (stru3d_processed) has missing scenes
        self.scene_ids = [
            self.coco.imgs[i]["file_name"].split("/")[0] for i in self.coco.imgs
        ]  # sorted(os.listdir(self.data_root))

    def __len__(self):
        return len(self.scene_ids)

    def _get_image(self, path):
        return Image.open(os.path.join(self.data_root, path))

    def _load_point_cloud_normal(
        self, scene_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert scene_id in self.scene_ids, "scene_id not found"
        ply_path = os.path.join(self.data_root, scene_id, "point_cloud.ply")
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=-1)

        # TODO: Downsample point cloud (1/10, 1/20 etc.)
        idxs = [i for i in range(0, len(xyz), 10)]
        xyz = xyz[idxs]
        colors = colors[idxs]
        normals = normals[idxs]

        return xyz, colors, normals

    # TODO: Map points to grid (256x256)
    def _point_cloud_alignment(
        self,
        point_cloud: np.array,
        normal: np.array,
        width: int = 256,
        height: int = 256,
    ):
        image_res = np.array((width, height), dtype=np.float32)
        min_coords = np.min(point_cloud[:, :2], axis=0)
        max_coords = np.max(point_cloud[:, :2], axis=0)
        span = max_coords - min_coords

        max_coords = max_coords + 0.1 * span
        min_coords = min_coords - 0.1 * span

        padded_span = max_coords[None, :2] - min_coords[None, :2]

        point_cloud[:, :2] = np.round(
            (point_cloud[:, :2] - min_coords[None, :2]) / padded_span * image_res[None]
        )
        # point_cloud[:, :2] = np.minimum(
        #     np.maximum(point_cloud[:, :2], np.zeros_like(image_res)), image_res - 1
        # )

        point_cloud[:, 2] = np.round(
            (point_cloud[:, 2] - point_cloud[:, 2].min())
            / (point_cloud[:, 2].max() - point_cloud[:, 2].min())
            * (width - 1)
        )

        scale_x, scale_y = image_res / padded_span.squeeze()
        scale_z = width / (point_cloud[:, 2].max() - point_cloud[:, 2].min())

        inv_scale = np.array([1.0 / scale_x, 1.0 / scale_y, 1.0 / scale_z])
        normal = normal * inv_scale

        return point_cloud, normal

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
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            # The point cloud is centered around camera center
            point_cloud = point_cloud @ rot  # torch.mm(point_cloud, rot.T)

            # suppress floating point errors
            point_cloud[np.abs(point_cloud) < 1e-12] = 0

        return point_cloud

    def _normal_augmentation(
        self,
        normals: torch.Tensor,
        horizontal: bool = False,
        vertical: bool = False,
        rotate: float = 0.0,
    ):
        if horizontal:
            normals[:, 0] = -normals[:, 0]
        if vertical:
            normals[:, 1] = -normals[:, 1]

        if horizontal ^ vertical:
            normals = -normals

        if rotate > 0:
            rad = rotate / 180.0 * np.pi
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            normals = normals @ rot  # torch.mm(normals, rot.T)

            normals[np.abs(normals) < 1e-12] = 0

        return normals

    def __getitem__(self, index):
        """
        Each item return consists of:
        - COCO object for density map
        - Point cloud xyz
        - Point cloud normal
        - Point cloud rgb
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

        _, h, w = record["image"].shape

        scene_id = self.scene_ids[index]
        point_cloud, colors, normals = self._load_point_cloud_normal(scene_id)

        point_cloud = self._point_cloud_augmentation(
            point_cloud, horizontal=_hor, vertical=_ver, rotate=_rotate
        )
        normal = self._normal_augmentation(
            normals, horizontal=_hor, vertical=_ver, rotate=_rotate
        )
        point_cloud, normal = self._point_cloud_alignment(
            point_cloud, normal, width=256, height=256
        )
        points = dict(
            coord=point_cloud,
            color=colors,
            normal=normal,
        )

        points = self._point_transforms(points)

        for k, v in points.items():
            record[k] = v

        # record["coord"] = self._point_cloud_augmentation(
        #     point_cloud, horizontal=_hor, vertical=_ver, rotate=_rotate
        # )
        # record["color"] = colors
        # record["normal"] = self._normal_augmentation(
        #     normals, horizontal=_hor, vertical=_ver, rotate=_rotate
        # )

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


# Replace the whole transform pipeline to also rotate and flip the point cloud
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
        ver = T.VFlipTransform(img_height)

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


def build(mode, args, point_transforms):
    assert os.path.exists(os.path.abspath(args.dataset_root)), (
        f"{args.dataset_root} does not exist"
    )
    dataset_root = os.path.abspath(args.dataset_root)

    dataset = PointCloudNormalDataset(
        dataset_root,
        transforms=make_poly_transforms(mode),
        point_transforms=point_transforms,
        aug_rotate=False,
        aug_flip=False,
        semantic_classes=args.semantic_classes,
        mode=mode,
    )

    return dataset
