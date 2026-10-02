# What? Generate cubemap from panorama images from Structured3D
# Why? To use as input for DINOv3
# How? With the help of the almighty py360convert
#
# Note that the output data directory looks like this:
# |- stru3d_cube_pc
#   |- train/test/val
#     |- scene_<scene_id>
#       |- face_<orientation>.png # for orientation in (U, D, R, L, F, B)
#       |- point_cloud.ply # currently subsampled, question to be changed so that matching with faces doesn't cause accident (in)occlusion(?)
#       |- density.png
#   |- annotations
#     |- train/test/val.json
# Another question is that should we generate 512x512x3 patches then later downsample it to 256x256x3 or should we just keep the entire thing 256x256x3 from the start?
import argparse
import os
from pathlib import Path

import numpy as np
from GeneralReader import GeneralReader
from tqdm import tqdm

with open("data_preprocess/stru3d/invalid_scenes.txt", "r") as file:
    INVALID_SCENES = file.read().split(",")

INVALID_SCENES = [int(x) for x in INVALID_SCENES]


def config():
    ap = argparse.ArgumentParser(
        description="Generate point cloud and cubemap 2D projection from panorama"
    )
    ap.add_argument(
        "--data_dir",
        default="data/Structured3D",
        help="Data directory. For Structured3D, it would be data/Structured3D",
    )
    ap.add_argument(
        "-o",
        "--output_dir",
        default="stru3d_processed",
        help="Output directory. Default is data/stru3d_processed",
    )
    ap.add_argument(
        "-w",
        "--width",
        default=256,
        type=int,
        help="Face width. `py360convert.p2c` assumes that we want the output to be squares (which is true)",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Print output if included",
    )  # Actually not quiet
    ap.add_argument(
        "-d", "--dry_run", default=False, action="store_true", help="For testing"
    )
    ap.add_argument(
        "--normal",
        default=False,
        action="store_true",
        help="If true, ONLY generate point cloud with normal (for LitePT)",
    )
    args = ap.parse_args()
    return args


def main(args):
    data_root = Path(args.data_dir)
    assert data_root.exists(), "data directory not found"

    output_dir = data_root.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Structured3D tree
    # | scene_<scene_id>
    #   | 2D_rendering
    #     | <room_id>/panorama
    #       | full
    #         | rgb_coldlight.png
    #         | depth.png
    #       | camera_xyz.txt
    scenes = os.listdir(data_root)
    for scene in tqdm(sorted(scenes)):
        if int(scene.split("_")[-1]) in INVALID_SCENES:
            if args.dry_run or args.verbose:
                print(f"skip {scene}")
            continue
        try:
            # if args.verbose or args.dry_run:
            #     print(f"processing {scene}")
            id = int(scene.split("_")[1])
            target = None
            if id < 3000:
                target = "train"
            elif id >= 3000 and id < 3250:
                target = "val"
            else:
                target = "test"
            scene_path = data_root / scene
            out_path = data_root / scene
            npy_out_path = output_dir / target / scene
            if not out_path.exists():
                out_path.mkdir(parents=True, exist_ok=True)
            reader = GeneralReader(
                scene_dir=scene_path,
                out_dir=out_path,
                face_w=args.width,
                generate_color=True,
                generate_normal=args.normal,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            # if not args.dry_run:
            if not args.normal:
                reader.export_point_cloud_and_cubemap()
            else:
                point_cloud = reader.export_point_cloud_normal()

                xyz = point_cloud['coords']
                colors = point_cloud['colors']
                normals = point_cloud['normals']

                # TODO: Downsample point cloud (1/10, 1/20 etc.)
                idxs = [i for i in range(0, len(xyz), 10)]
                xyz = xyz[idxs]
                colors = colors[idxs]
                normals = normals[idxs]

                merge = np.concat([xyz, colors, normals], axis=1)
                if args.dry_run or args.verbose:
                    print(npy_out_path / "point_cloud.npy", merge.shape)
                else:
                    np.save(npy_out_path / "point_cloud.npy", merge)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main(config())
