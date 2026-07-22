import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import py360convert

# import matplotlib.pyplot as plt  # just for the sake of visual debug
from plyfile import PlyData

FACES = ["U", "F", "R", "L", "B", "D"]

# Define Rotation matrices for each face relative to standard camera coordinate frame
# OpenCV Frame: +X Right, +Y Down, +Z Forward
ROTATIONS = {
    "F": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),  # Yaw   0°
    "R": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),  # Yaw +90°
    "B": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),  # Yaw 180°
    "L": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),  # Yaw -90°
    "D": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),  # Pitch +90°
    "U": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),  # Pitch -90°
}


class CubeMapReader:
    def __init__(
        self, scene_dir: Path, out_dir: Path, face_w: int = 256, verbose: bool = False
    ):
        assert scene_dir.exists(), f"{scene_dir} not found"
        self.scene_dir = scene_dir

        assert out_dir.exists(), f"{out_dir} not found"
        self.out_dir = out_dir

        self.rooms = os.listdir(scene_dir / "2D_rendering")
        self.rgb_paths = [
            scene_dir
            / "2D_rendering"
            / room
            / "panorama"
            / "full"
            / "rgb_coldlight.png"
            for room in self.rooms
        ]
        self.depth_paths = [
            scene_dir / "2D_rendering" / room / "panorama" / "full" / "depth.png"
            for room in self.rooms
        ]
        self.cam_xyz_paths = [
            scene_dir / "2D_rendering" / room / "panorama" / "camera_xyz.txt"
            for room in self.rooms
        ]

        self.ply_path = scene_dir / "point_cloud.ply"

        for p in self.rgb_paths:
            assert p.exists(), f"{p} not found"

        for p in self.depth_paths:
            assert p.exists(), f"{p} not found"

        for p in self.cam_xyz_paths:
            assert p.exists(), f"{p} not found"

        assert self.ply_path.exists(), f"{self.ply_path} not found"

        self.face_w = face_w
        self.verbose = verbose

        # Generate faces and pre-compute which 3D points lie in each face
        self.K_face = self._get_cubemap_intrinsics(self.face_w)
        self.R_base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        self.Ts_world = self._load_camera_centers(self.cam_xyz_paths)

        self._generate_faces()

    def _load_camera_centers(self, cam_xyz_paths: List[str | Path]) -> np.array:
        Ts = []
        for path in cam_xyz_paths:
            with open(path, "r") as f:
                Ts.append([float(v) for v in f.readline().strip().split()])
        return np.array(Ts, dtype=np.float32)

    def _generate_faces(self):
        for room, rgb_path, depth_path, cam_xyz_path in zip(
            self.rooms, self.rgb_paths, self.depth_paths, self.cam_xyz_paths
        ):
            out_path = self.out_dir / room
            out_path.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            img_dict = py360convert.e2c(img, face_w=self.face_w, cube_format="dict")
            depth_dict = py360convert.e2c(
                depth[:, :, None], face_w=self.face_w, cube_format="dict"
            )
            # [:, :, None] is used to expand [width, height] to [width, height, channel]

            for face in FACES:
                img_name = out_path / f"img_{face}.png"
                dep_name = out_path / f"dep_{face}.png"

                if self.verbose:
                    print(img_name, dep_name)

                cv2.imwrite(img_name, img_dict[face])
                cv2.imwrite(dep_name, depth_dict[face])

    def _get_cubemap_intrinsics(self, width) -> np.array:
        """
        Computes Intrinsic Matrix K for a 90-degree FOV square cubemap face.
        Focal length f = W / (2 * tan(90/2)) = W / 2
        """
        f = width / 2.0
        cx, cy = width / 2.0, width / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        return K

    def _get_points_in_face(
        self, pts_3d, colors, K, R_face, R_base, T_cam, img_w, img_h
    ):
        """
        Transforms 3D points into camera space and determines which points project
        inside the image boundaries [0, img_w] x [0, img_h].
        """
        # 1. Combined World-to-Camera Rotation Matrix
        R_total = R_face @ R_base

        # 2. Transform 3D World Points -> 3D Camera Points
        pts_cam = (R_total @ (pts_3d - T_cam).T).T  # (N, 3)

        # 3. Filter points in front of the camera (Z > 0)
        valid_z = pts_cam[:, 2] > 0.1

        # 4. Project onto Image Plane: [u, v, 1] = K * [X/Z, Y/Z, 1]
        pts_proj = (K @ (pts_cam / pts_cam[:, 2:3]).T).T
        u = pts_proj[:, 0]
        v = pts_proj[:, 1]

        # 5. Filter points strictly inside pixel bounds
        in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & valid_z

        return in_bounds, u[in_bounds], v[in_bounds]

    def export_masks(self):
        plydata = PlyData.read(self.ply_path)
        vertex = plydata["vertex"]

        # Extract 3D coordinates (X, Y, Z)
        pts_3d_world = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
            np.float32
        )

        # Extract RGB Colors (handles 0-255 uint8 or 0.0-1.0 float format)
        if "red" in vertex.data.dtype.names:
            colors_r = vertex["red"]
            colors_g = vertex["green"]
            colors_b = vertex["blue"]

            # Normalize uint8 (0-255) to float (0-1) for matplotlib
            if colors_r.max() > 1.0:
                colors_world = (
                    np.stack([colors_r, colors_g, colors_b], axis=1).astype(np.float32)
                    / 255.0
                )
            else:
                colors_world = np.stack([colors_r, colors_g, colors_b], axis=1).astype(
                    np.float32
                )
        else:
            # Default fallback color (gray) if PLY lacks RGB channels
            print("no rgb")
            colors_world = np.ones_like(pts_3d_world) * 0.5

        if self.verbose:
            print(f"Loaded Point Cloud: {pts_3d_world.shape[0]} points")

        for room, xyz in zip(self.rooms, self.cam_xyz_paths):
            with open(xyz, "r") as f:
                Ts = np.array(
                    [float(v) for v in f.readline().strip().split()], dtype=np.float32
                )

            out_path = self.out_dir / room
            for face in FACES:
                R_face = ROTATIONS[face]
                mask, _, _ = self._get_points_in_face(
                    pts_3d_world,
                    colors_world,
                    self.K_face,
                    R_face,
                    self.R_base,
                    Ts,
                    self.face_w,
                    self.face_w,
                )

                mask_name = out_path / f"mask_{face}"
                if self.verbose:
                    print(f"save mask to {mask_name}.npy")
                np.save(mask_name, mask)
