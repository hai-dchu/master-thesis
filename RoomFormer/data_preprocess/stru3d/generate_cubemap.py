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

from CubeMapReader import CubeMapReader


def config():
    ap = argparse.ArgumentParser(
        description="Generate cubemap 2D projection from panorama"
    )
    ap.add_argument(
        "-d",
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
    scenes = ["scene_00400"] + os.listdir(data_root)
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
            reader = CubeMapReader(
                scene_dir=scene_path, out_dir=out_path, verbose=args.verbose
            )
            reader.export_masks()
        except Exception as e:
            print(f"{type(e).__name__} - {e}")


if __name__ == "__main__":
    main(config())
