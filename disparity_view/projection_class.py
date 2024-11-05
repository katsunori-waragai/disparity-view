from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d

from .util import dummy_camera_matrix
from .util import safer_imsave
from .o3d_project import generate_point_cloud
from .o3d_project import DEPTH_MAX, DEPTH_SCALE
from .zed_info import CameraParameter


@dataclass
class StereoCamera:
    baseline: float = field(default=120.0)  # [mm]
    left_camera_matrix: np.ndarray = field(default=None)
    right_camera_matrix: np.ndarray = field(default=None)
    extrinsics: np.ndarray = field(default=None)
    depth_scale: float = DEPTH_SCALE
    depth_max: float = DEPTH_MAX
    pcd: o3d.t.geometry.PointCloud = field(default=None)
    rgbd: o3d.t.geometry.RGBDImage = field(default=None)
    shape: Tuple[float] = field(default=None)

    def load_camera_parameter(self, json: Path):
        """ """
        self.left_camera_matrix = CameraParameter.load_json(json).to_matrix()
        self.right_camera_matrix = self.left_camera_matrix

    def set_camera_matrix(self, shape: np.ndarray, focal_length: float = 1070.0, baseline=120):
        self.shape = shape
        self.left_camera_matrix = o3d.core.Tensor(dummy_camera_matrix(shape, focal_length=focal_length))
        self.right_camera_matrix = self.left_camera_matrix
        self.baseline = baseline

    def generate_point_cloud(self, disparity_map: np.ndarray, left_image: np.ndarray):
        if disparity_map.shape[:2] != left_image.shape[:2]:
            print(f"{disparity_map.shape=} {left_image.shape[:2]=}")
        assert disparity_map.shape[:2] == left_image.shape[:2]
        return generate_point_cloud(disparity_map, left_image, self.left_camera_matrix, self.baseline)

    def project_to_rgbd_image(self, extrinsics=o3d.core.Tensor(np.eye(4, dtype=np.float32))):
        height, width = self.shape[:2]
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(self.left_camera_matrix, o3d.core.Tensor)
        assert isinstance(self.right_camera_matrix, o3d.core.Tensor)
        assert isinstance(extrinsics, o3d.core.Tensor) or isinstance(extrinsics, np.ndarray)
        return self.pcd.project_to_rgbd_image(
            width,
            height,
            intrinsics=self.left_camera_matrix,
            extrinsics=extrinsics,
            depth_scale=DEPTH_SCALE,
            depth_max=DEPTH_MAX,
        )

    def scaled_baseline(self):
        return self.baseline / DEPTH_SCALE
