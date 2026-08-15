"""
What: general cube map and point cloud reader
Why : mainly because mapping each room's point cloud
      to their corresponding visible faces is hard
      to do separately, and reading the point cloud
      again is time-consuming and inefficient

Author: Hai Chu
"""

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import py360convert

"""
Overall idea: parsing through the dataset:
- save point cloud to data source
- save cube map faces and mask to data target
"""

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

REJECT_THRESH_DEPTH = 500.0


class GeneralReader:
    def __init__(
        self,
        scene_dir: Path,
        out_dir: Path,
        face_w: int = 256,
        generate_color: bool = False,
        verbose: bool = False,
        dry_run: bool = False,
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

        # assert self.ply_path.exists(), f"{self.ply_path} not found"

        self.face_w = face_w
        self.generate_color = generate_color

        self.verbose = verbose
        self.dry_run = dry_run

        # Generate faces and pre-compute which 3D points lie in each face
        # self.K_face = self._get_cubemap_intrinsics(self.face_w)
        self.R_base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        self.camera_centers = self._load_camera_centers(self.cam_xyz_paths)

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

            self.img_dict = py360convert.e2c(
                img, face_w=self.face_w, cube_format="dict"
            )
            self.depth_dict = py360convert.e2c(
                depth[:, :, None], face_w=self.face_w, cube_format="dict"
            )
            # [:, :, None] is used to expand [width, height] to [width, height, channel]

            for face in FACES:
                img_name = out_path / f"img_{face}.png"
                dep_name = out_path / f"dep_{face}.png"

                if self.verbose or self.dry_run:
                    print(img_name, dep_name)

                if not self.dry_run:
                    cv2.imwrite(img_name, self.img_dict[face])
                    cv2.imwrite(dep_name, self.depth_dict[face])

    def _radial_to_planar_depth(self, depth_radial, K):
        """Converts 360 radial depth map into pinhole planar Z-depth map."""
        h, w = depth_radial.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_dir = (x - cx) / fx
        y_dir = (y - cy) / fy
        ray_length = np.sqrt(x_dir**2 + y_dir**2 + 1.0)

        return (depth_radial.squeeze() / ray_length).astype(np.float32)

    def _generate_point_cloud(
        self,
        rgb_path: str,
        depth_path: str,
        camera_center: np.array,
        random_level: int = 0,
    ):
        """
        Function to generate point cloud from a single panorama + depth map
        """
        # Assuming assertion is done in `self.__init__`
        try:
            depth_img = cv2.imread(
                depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
            )
            assert depth_img is not None, f"img {depth_path} not available"

            coords, colors = [], []

            width, height = depth_img.shape[:2]
            x_tick = 180.0 / width
            y_tick = 360.0 / height

            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, code=cv2.COLOR_BGR2RGB)

            for x in range(0, width):
                for y in range(0, height):
                    # alpha in range (90, -90)
                    alpha = 90 - (x * x_tick)

                    # beta in range (-180, 180)
                    beta = y * y_tick - 180

                    alpha = np.deg2rad(alpha)
                    beta = np.deg2rad(beta)

                    depth = depth_img[x, y] + np.random.random() * random_level

                    if depth > REJECT_THRESH_DEPTH:
                        # This is basically converting from spherical coordinate to cartesian coordinate
                        z_offset = depth * np.sin(alpha)

                        xy_offset = depth * np.cos(alpha)

                        x_offset = xy_offset * np.sin(beta)
                        y_offset = xy_offset * np.cos(beta)

                        point = np.asarray([x_offset, y_offset, z_offset])

                        coords.append(point + camera_center)
                        colors.append(rgb_img[x, y])
            return np.asarray(coords), np.asarray(colors)

        except AssertionError as e:
            print(e)

    def _get_points_in_face(
        self,
        pts_3d: np.array,  # Room-specific point cloud, i.e. a subset of the scene
        K: np.array,
        R_face: np.array,
        R_base: np.array,
        T_cam: np.array,
        depth: np.array,
        img_width: int = 256,
        img_height: int = 256,
        depth_tol: int = 10.0,
    ) -> tuple[np.array, np.array, np.array]:
        """
        Transforms 3D points into camera space and determines which points project
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

        # map to (0,255)
        u = (u - u.min()) / (u.max() - u.min()) * (img_width - 1)
        v = (v - v.min()) / (v.max() - v.min()) * (img_height - 1)

        # Keep points in front of the camera since every point is in frame
        in_bounds = valid_z

        # z_point = pts_cam[in_bounds, 2]

        # 6. DEPTH VERIFICATION (Occlusion / Distance Agreement)
        # Query corresponding planar depth from face depth map
        # Since our data already contains a depth cube map, it is simpler to account for occlusion
        # as depth-mismatch point (up to depth_tol) are considered occluded and should be hidden

        # u_valid = u[in_bounds].astype(int)
        # v_valid = v[in_bounds].astype(int)

        return in_bounds, u[in_bounds], v[in_bounds]

        # z_map_expected = depth[v_valid, u_valid]

        # # Keep points whose 3D depth matches the face depth map within tolerance
        # depth_match = (np.abs(z_point - z_map_expected) <= depth_tol) & (
        #     z_map_expected > 0
        # )

        # # Combine masks
        # final_mask = np.zeros_like(in_bounds, dtype=bool)
        # final_indices = np.where(in_bounds)[0][depth_match]
        # final_mask[final_indices] = True

        # return final_mask, u[final_mask], v[final_mask]

    def export_point_cloud_and_cubemap(self):
        all_coords, all_colors = [], []
        for i, (room, rgb_path, depth_path, cam_xyz_path) in enumerate(
            zip(self.rooms, self.rgb_paths, self.depth_paths, self.cam_xyz_paths)
        ):
            camera_center = self.camera_centers[i]
            coords, colors = self._generate_point_cloud(
                rgb_path, depth_path, camera_center
            )
            colors = colors / 255.0  # normalize to [0,1]

            # RoomFormer subsampling
            coords[:, :2] = np.round(coords[:, :2] / 10) * 10.0
            coords[:, 2] = np.round(coords[:, 2] / 100) * 100.0
            unique_coords, unique_ind = np.unique(coords, return_index=True, axis=0)

            coords = coords[unique_ind]
            colors = colors[unique_ind]

            all_coords.append(coords)
            all_colors.append(colors)

        K_face = np.array(
            [
                [self.face_w / 2.0, 0, self.face_w / 2.0],
                [0, self.face_w / 2.0, self.face_w / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Since each room has their own point cloud, we need to keep track the mask
        # w.r.t. the scene point cloud
        mask_temp = np.zeros(sum([i.shape[0] for i in all_coords]))

        # Proposed subsampling
        original_size = mask_temp.shape[0]
        rate = 10

        if original_size > 1e6:
            rate *= 2
        elif original_size < 1e5:
            rate = 1

        room_idx = []
        all_idx = []
        start = 0
        for c in all_coords:
            idx = np.array([j for j in range(0, c.shape[0], rate)])
            room_idx.append(idx)
            all_idx.append(start + idx)
            start += c.shape[0]

        start = 0
        for i, room in enumerate(self.rooms):
            coords = all_coords[i]
            colors = all_colors[i]

            #
            idx_room = room_idx[i]
            coords = coords[idx_room]
            colors = colors[idx_room]

            depth_planar_dict = {
                key: self._radial_to_planar_depth(self.depth_dict[key], K_face)
                for key in self.img_dict.keys()
            }

            out_path = self.out_dir / room

            for face in FACES:
                mask = mask_temp.copy()
                R_face = ROTATIONS[face]
                depth = depth_planar_dict[face]
                mask_room, u_valid, v_valid = self._get_points_in_face(
                    coords,
                    K_face,
                    R_face,
                    self.R_base,
                    camera_center,
                    depth,
                )

                mask[start : start + coords.shape[0]] = mask_room

                prj_points = np.stack([u_valid, v_valid], axis=1)

                mask_name = out_path / f"mask_{face}"
                prj_points_name = out_path / f"prj_points_{face}"

                if self.verbose or self.dry_run:
                    print(f"save mask to {mask_name}.npy")
                    print(f"save projected points to {prj_points_name}.npy")

                if not self.dry_run:
                    np.save(mask_name, mask)
                    np.save(prj_points_name, prj_points)

            start += coords.shape[0]

        all_coords = np.concat(all_coords, axis=0)
        all_colors = np.concat(all_colors, axis=0)

        all_idx = np.concat(all_idx, axis=0)
        all_coords = all_coords[all_idx]
        all_colors = all_colors[all_idx]

        point_cloud = {}
        point_cloud["coords"] = all_coords
        point_cloud["colors"] = all_colors

        if self.verbose or self.dry_run:
            print(f"point cloud size: {all_coords.shape}")
            print(f"saved to {self.ply_path}")

        if not self.dry_run:
            export_ply(self.ply_path, point_cloud, self.generate_color)


def export_ply(path: str, point_cloud: dict[str, np.array], generate_color=False):
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
    assert "coords" in point_cloud.keys(), "empty point cloud"
    assert ~(generate_color and ("colors" in point_cloud.keys())), (
        "given generate_color=True, color for each point should be provided"
    )

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
