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

from tqdm import tqdm

from GeneralReader import GeneralReader


def config():
    ap = argparse.ArgumentParser(
        description="Generate point cloud and cubemap 2D projection from panorama"
    )
    ap.add_argument(
        "--data_dir",
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
        try:
            if args.verbose:
                print(f"processing {scene}")
            id = int(scene.split("_")[1])
            target = ""
            if id < 3000:
                target = "train"
            elif id >= 3000 and id < 3250:
                target = "val"
            else:
                target = "test"
            scene_path = data_root / scene
            out_path = output_dir / target / scene
            if not out_path.exists():
                out_path.mkdir(parents=True, exist_ok=True)
            reader = GeneralReader(
                scene_dir=scene_path,
                out_dir=out_path,
                face_w=args.width,
                generate_color=True,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            reader.export_point_cloud_and_cubemap()
        except Exception as e:
            tb = e.__traceback__
            print(f"{type(e).__name__} - {e}")
            print(f"File Name: {tb.tb_frame.f_code.co_filename}")
            print(f"Line Number: {tb.tb_lineno}")


if __name__ == "__main__":
    main(config())
