# --------------------------------------------------------------------------------------------
# Modified from Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries
# https://github.com/ywyue/RoomFormer/blob/main/data_preprocess/stru3d/generate_coco_stru3d.py
#
# Process raw .ply data and 3d annotations and store them in stru3d_processed in the form
# |-stru3d_processed
#   |-train (test, val)
#     |-scene_<scene_id>
#       |-point_cloud.ply
#       |-density.png
#   |-annotations
#     |-train.json (test, val)
#
# Hai Chu
# --------------------------------------------------------------------------------------------

import argparse
import json
import os
import sys

from stru3d_utils import (
    generate_coco_dict,
    generate_density,
    normalize_annotations,
    parse_floor_plan_polys,
)
from tqdm import tqdm

sys.path.append("../.")
from common_utils import export_density, read_scene_pc

with open("invalid_scenes.txt", "r") as file:
    INVALID_SCENES = file.read().split(",")

INVALID_SCENES = [int(x) for x in INVALID_SCENES]

TYPE_ID_MAPPING = {
    "living room": 0,
    "kitchen": 1,
    "bedroom": 2,
    "bathroom": 3,
    "balcony": 4,
    "corridor": 5,
    "dining room": 6,
    "study": 7,
    "studio": 8,
    "store room": 9,
    "garden": 10,
    "laundry room": 11,
    "office": 12,
    "basement": 13,
    "garage": 14,
    "undefined": 15,
    "door": 16,
    "window": 17,
}


def config():
    ap = argparse.ArgumentParser(description="Generate point cloud annotations")
    ap.add_argument(
        "--data_root",
        type=str,
        help="path to raw Structured3D folder",
    )
    ap.add_argument(
        "--output",
        default="stru3d_processed",
        type=str,
        help="path to output folder. output folder will be in the same parent folder as --data_root",
    )
    ap.add_argument(
        "--output_width",
        default=256,
        help="width of the bird-eye 2D view to scale vertices to",
    )
    ap.add_argument(
        "--output_height",
        default=256,
        help="height of the bird-eye 2D view to scale vertices to",
    )
    ap.add_argument(
        "--verbose", default=False, action="store_true", help="use to enable printout"
    )
    args = ap.parse_args()
    return args


def main(args):
    data_root = args.data_root
    scenes = sorted(os.listdir(data_root))

    ### prepare
    parent = os.path.abspath(os.path.join(data_root, os.pardir))
    out_folder = os.path.join(parent, args.output)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    annotation_out_folder = os.path.join(out_folder, "annotations")
    if not os.path.exists(annotation_out_folder):
        os.mkdir(annotation_out_folder)

    train_img_folder = os.path.join(out_folder, "train")
    val_img_folder = os.path.join(out_folder, "val")
    test_img_folder = os.path.join(out_folder, "test")

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(annotation_out_folder, "train.json")
    coco_val_json_path = os.path.join(annotation_out_folder, "val.json")
    coco_test_json_path = os.path.join(annotation_out_folder, "test.json")

    coco_train_dict = {"images": [], "annotations": [], "categories": []}
    coco_val_dict = {"images": [], "annotations": [], "categories": []}
    coco_test_dict = {"images": [], "annotations": [], "categories": []}

    for key, value in TYPE_ID_MAPPING.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        coco_train_dict["categories"].append(type_dict)
        coco_val_dict["categories"].append(type_dict)
        coco_test_dict["categories"].append(type_dict)

    ### begin processing
    instance_id = 0
    for scene in tqdm(scenes):
        scene_path = os.path.join(data_root, scene)
        scene_id = scene.split("_")[-1]

        if int(scene_id) in INVALID_SCENES:
            print("skip {}".format(scene))
            continue

        # load pre-generated point cloud
        ply_path = os.path.join(scene_path, "point_cloud.ply")
        points = read_scene_pc(ply_path)
        xyz = points[:, :3]

        ### project point cloud to density map
        density, normalization_dict = generate_density(
            xyz, width=args.output_width, height=args.output_height
        )

        ### rescale raw annotations
        normalized_annos = normalize_annotations(scene_path, normalization_dict)

        ### prepare coco dict
        img_id = int(scene_id)
        img_dict = {}
        img_dict["file_name"] = os.path.join(scene, "density.png")
        img_dict["id"] = img_id
        img_dict["width"] = args.output_width
        img_dict["height"] = args.output_height

        ### parse annotations
        polys = parse_floor_plan_polys(normalized_annos)
        polygons_list = generate_coco_dict(
            normalized_annos, polys, instance_id, img_id, ignore_types=["outwall"]
        )

        instance_id += len(polygons_list)

        ### train
        if int(scene_id) < 3000:
            coco_train_dict["images"].append(img_dict)
            coco_train_dict["annotations"] += polygons_list
            density_out_dir = os.path.join(train_img_folder, scene)
            if not os.path.exists(density_out_dir):
                os.mkdir(density_out_dir)
            export_density(density, density_out_dir, "density")
            os.rename(ply_path, os.path.join(density_out_dir, "point_cloud.ply"))

        ### val
        elif int(scene_id) >= 3000 and int(scene_id) < 3250:
            coco_val_dict["images"].append(img_dict)
            coco_val_dict["annotations"] += polygons_list
            density_out_dir = os.path.join(val_img_folder, scene)
            if not os.path.exists(density_out_dir):
                os.mkdir(density_out_dir)
            export_density(density, density_out_dir, "density")
            os.rename(ply_path, os.path.join(density_out_dir, "point_cloud.ply"))

        ### test
        else:
            coco_test_dict["images"].append(img_dict)
            coco_test_dict["annotations"] += polygons_list
            density_out_dir = os.path.join(test_img_folder, scene)
            if not os.path.exists(density_out_dir):
                os.mkdir(density_out_dir)
            export_density(density, density_out_dir, "density")
            os.rename(ply_path, os.path.join(density_out_dir, "point_cloud.ply"))

        if args.verbose:
            print(scene_id)

    with open(coco_train_json_path, "w") as f:
        json.dump(coco_train_dict, f)
    with open(coco_val_json_path, "w") as f:
        json.dump(coco_val_dict, f)
    with open(coco_test_json_path, "w") as f:
        json.dump(coco_test_dict, f)


if __name__ == "__main__":
    main(config())
