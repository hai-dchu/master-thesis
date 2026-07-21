# ---------------------------------------------------------------------------------------------------
# Modified from Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries
# https://github.com/ywyue/RoomFormer/blob/main/data_preprocess/stru3d/generate_point_cloud_stru3d.py
#
# Process raw .ply data and 3d annotations and store them in stru3d_processed in the form
# stru3d_processed
# |-scene_<scene_id>
#   |-point_cloud.ply
#   |-annotation.json
#
# Hai Chu
# ---------------------------------------------------------------------------------------------------

import argparse
import os
from tqdm import tqdm
from PointCloudReaderPanorama import PointCloudReaderPanorama


def config():
    a = argparse.ArgumentParser(description="Generate point cloud for Structured3D")
    a.add_argument(
        "--data_root",
        type=str,
        help="path to raw Structured3D_panorama folder",
    )
    a.add_argument(
        "--downsample",
        default=False,
        action="store_true",
        help="if true, downsample the point cloud",
    )
    args = a.parse_args()
    return args


def main(args):
    print("Creating point cloud from perspective views...")
    data_root = args.data_root
    scenes = os.listdir(data_root)

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_root, scene)
        try:
            reader = PointCloudReaderPanorama(
                scene_path, random_level=0, generate_color=True, downsample=args.downsample
            )
            save_path = os.path.join(scene_path, "point_cloud.ply")
            reader.export_ply(save_path)
        except Exception:
            print(f'{scene_path} is corrupted')


if __name__ == "__main__":
    main(config())
