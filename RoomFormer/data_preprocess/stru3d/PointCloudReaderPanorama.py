import cv2
import os

import numpy as np

NUM_SECTIONS = -1
REJECT_THRESH_DEPTH = 500.0  # mm, eliminate near-camera noise


class PointCloudReaderPanorama:
    def __init__(
        self,
        path,
        resolution="full",
        random_level=0,
        generate_color=False,
        verbose=False,
        downsample=False,
    ):
        self.path = path
        self.random_level = random_level
        self.resolution = resolution
        self.generate_color = generate_color
        self.verbose = verbose
        self.downsample = downsample

        sections = [p for p in os.listdir(os.path.join(path, "2D_rendering"))]
        self.depth_paths = [
            os.path.join(
                *[path, "2D_rendering", p, "panorama", self.resolution, "depth.png"]
            )
            for p in sections
        ]
        self.rgb_paths = [
            os.path.join(
                *[
                    path,
                    "2D_rendering",
                    p,
                    "panorama",
                    self.resolution,
                    "rgb_coldlight.png",
                ]
            )
            for p in sections
        ]
        self.camera_paths = [
            os.path.join(*[path, "2D_rendering", p, "panorama", "camera_xyz.txt"])
            for p in sections
        ]
        self.camera_centers = self.read_camera_center()
        self.point_cloud = self.generate_point_cloud(
            self.random_level, color=self.generate_color
        )

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], "r") as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
        return camera_centers

    def generate_point_cloud(self, random_level=0, color=False):
        coords = []
        colors = []

        # Getting Coordinates
        for i in range(len(self.depth_paths)):
            try:
                depth_img = cv2.imread(
                    self.depth_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
                )
                assert depth_img is not None, f"img {self.depth_paths[i]} not available"

                # This is the script to convert panorama image coordinate to cartesian coordinate
                """
                The image (rgb_img) is wrapped around a sphere, then the point coordinate is calculated
                from the depth its angle in space.
                """
                width, height = depth_img.shape[:2]
                x_tick = 180.0 / width
                y_tick = 360.0 / height

                rgb_img = cv2.imread(self.rgb_paths[i])
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
                            # This is basically converting from cartesian coordinate to spherical coordinate
                            z_offset = depth * np.sin(alpha)

                            xy_offset = depth * np.cos(alpha)

                            x_offset = xy_offset * np.sin(beta)
                            y_offset = xy_offset * np.cos(beta)

                            point = np.asarray([x_offset, y_offset, z_offset])

                            coords.append(point + self.camera_centers[i])
                            colors.append(rgb_img[x, y])
            except AssertionError:
                print(f"img {self.depth_paths[i]} not available")
                continue

        coords = np.asarray(coords)
        colors = np.asarray(colors) / 255.0  # normalize to 0, 1

        if self.downsample:
            # Downsampling/voxelizing the point cloud by removing points that are
            # - less than 10mm from others in x and y axes and
            # - less than 100mm from others in z axis
            coords[:, :2] = np.round(coords[:, :2] / 10) * 10.0
            coords[:, 2] = np.round(coords[:, 2] / 100) * 100.0
            unique_coords, unique_ind = np.unique(coords, return_index=True, axis=0)

            coords = coords[unique_ind]
            colors = colors[unique_ind]

        points = {}
        points["coords"] = coords
        points["colors"] = colors

        if self.verbose:
            print("Pointcloud size:", points["coords"].shape[0])

        return points

    def get_point_cloud(self):
        return self.point_cloud

    def export_ply(self, path):
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
            f.write("element vertex %d\n" % self.point_cloud["coords"].shape[0])
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if self.generate_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(self.point_cloud["coords"].shape[0]):
                color = []
                coord = self.point_cloud["coords"][i].tolist()
                if self.generate_color:
                    color = list(
                        map(int, (self.point_cloud["colors"][i] * 255).tolist())
                    )
                data = coord + color
                f.write(" ".join(list(map(str, data))) + "\n")
