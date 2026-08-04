import argparse
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData
from tqdm import tqdm

def config():
    a = argparse.ArgumentParser(
        description="Reduce point cloud for scene in stru3d_processed"
    )
    a.add_argument(
        "--data_root",
        type=str,
        help="path to stru3d_processed folder",
    )
    a.add_argument(
        "--reduction_rate",
        type=int,
        default=10,
        help="rate of reduction to be applied, default=10, i.e. 1/10 of the original data is kept",
    )
    a.add_argument(
        "-d", "--dry_run", default=False, action="store_true", help="test run"
    )
    args = a.parse_args()
    return args


def main(args):
    modes = ["train", "test", "val"]
    for mode in modes:
        data_root = Path(args.data_root)
        target_root = data_root / mode
        scenes = os.listdir(target_root)
        for scene in tqdm(sorted(scenes)):
            target = target_root / scene
            try:
                ply_path = target / "point_cloud.ply"
                assert target.exists(), f"{target} not found"
                assert ply_path.exists(), f"{ply_path} not found"

                plydata = PlyData.read(ply_path)
                vertex = plydata["vertex"]

                xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
                colors = np.stack(
                    [vertex["red"], vertex["green"], vertex["blue"]], axis=-1
                )

                original_pc_size = xyz.shape[0]

                rate = args.reduction_rate

                if original_pc_size > 1000000: # more than 1m points
                    rate *= 2
                elif original_pc_size < 100000: # less than 100k points
                    rate = 1

                idx = [i for i in range(0, len(xyz), rate)]
                xyz = xyz[idx]
                colors = colors[idx]

                points = {"coords": xyz, "colors": colors}

                if args.dry_run:
                    print(
                        f"original point cloud size: {original_pc_size} \treduced point cloud size: {points['coords'].shape[0]}"
                    )
                else:
                    export_ply(ply_path, points)

            except AssertionError as e:
                print(f"{e}")
                continue


def export_ply(path, point_cloud: dict[str, np.array], generate_color: bool = True):
    """
    ply
    format ascii 1.0
    comment Mars model by Paul Bourke
    element vertex 259200
    property float x
    property float y
    property float z
    property uchar r
    property uchar g
    property uchar b
    property float nx
    property float ny
    property float nz
    end_header
    """
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % point_cloud["coords"].shape[0])
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if generate_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(point_cloud["coords"].shape[0]):
            color = []
            coord = point_cloud["coords"][i].tolist()
            if generate_color:
                color = list(map(int, (point_cloud["colors"][i] * 255).tolist()))
            data = coord + color
            f.write(" ".join(list(map(str, data))) + "\n")


if __name__ == "__main__":
    main(config())
