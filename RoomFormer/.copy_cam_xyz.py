import argparse
import os
from pathlib import Path
import shutil


def config():
    a = argparse.ArgumentParser(
        description="Copy camera xyz to matching scene in stru3d_processed"
    )
    a.add_argument(
        "--data_root",
        type=str,
        help="path to raw Structured3D_panorama folder",
    )
    a.add_argument("--target", default="stru3d_processed", help="target folder")
    a.add_argument(
        "-d", "--dry_run", default=False, action="store_true", help="test run"
    )
    args = a.parse_args()
    return args


def main(args):
    modes = ["train", "test", "val"]
    for mode in modes:
        data_root = Path(args.data_root)
        target_root = data_root.parent / args.target / mode
        scenes = os.listdir(target_root)
        for scene in sorted(scenes):
            room_paths = target_root / scene
            rooms = os.listdir(room_paths)

            for room in rooms:
                target = target_root / scene / room
                if not target.is_dir():
                    continue
                try:
                    # print(target)
                    xyz_path = data_root / scene / "2D_rendering" / room / "panorama" / "camera_xyz.txt"

                    assert xyz_path.exists(), f"{xyz_path} not found"
                    assert target.exists(), f"{target} not found"

                    if args.dry_run:
                        print(f"camera xyz: {str(xyz_path)}")
                        print(f"target    : {str(target)}")
                    else:
                        shutil.copy(xyz_path, target)
                except AssertionError as e:
                    print(f"{e}")
                    continue


if __name__ == "__main__":
    main(config())
