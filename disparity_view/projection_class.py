from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from .animation_gif import AnimationGif
from .util import dummy_camera_matrix, safer_imsave
from .util import safer_imsave
from .o3d_project import generate_point_cloud, gen_tvec, as_extrinsics
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

    def set_camera_matrix(self, shape: np.ndarray, focal_length: float = 1070.0):
        self.shape = shape
        self.left_camera_matrix = o3d.core.Tensor(dummy_camera_matrix(shape, focal_length=focal_length))
        self.right_camera_matrix = self.left_camera_matrix

    def set_baseline(self, baseline=120):
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


def gen_right_image(disparity, left_image, outdir, left_name, axis=0):
    stereo_camera = StereoCamera(baseline=120)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    scaled_baseline = stereo_camera.scaled_baseline()  # [mm]
    tvec = gen_tvec(scaled_shift=scaled_baseline, axis=axis)
    extrinsics = as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = outdir / f"color_{left_name.stem}.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    depth_legacy = np.asarray(projected.depth.to_legacy())
    depth_file = outdir / f"depth_{left_name.stem}.png"
    depth_file.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(depth_file), depth_legacy)
    print(f"saved {outfile}")
    print(f"saved {depth_file}")

    assert outfile.lstat().st_size > 0


def make_animation_gif(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path, axis=0):
    """
    save animation gif file

    Args:
        disparity: disparity image
        left_image:left camera image
        outdir: destination directory
        left_name: file name of the left camera image
    Returns：
        None
    """
    assert axis in (0, 1, 2)

    stereo_camera = StereoCamera(baseline=120)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    scaled_baseline = stereo_camera.scaled_baseline()  # [mm]
    tvec = gen_tvec(scaled_shift=scaled_baseline, axis=axis)
    extrinsics = as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = outdir / f"color_{left_name.stem}.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    depth_legacy = np.asarray(projected.depth.to_legacy())
    depth_file = outdir / f"depth_{left_name.stem}.png"
    depth_file.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(depth_file), depth_legacy)
    print(f"saved {outfile}")
    print(f"saved {depth_file}")


    maker = AnimationGif()
    n = 16
    for i in tqdm(range(n + 1)):
        scaled_baseline = stereo_camera.scaled_baseline()
        tvec = gen_tvec(scaled_baseline * i / n, axis)
        extrinsics = as_extrinsics(tvec)
        projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
        color_img = np.asarray(projected.color.to_legacy())
        color_img = (color_img * 255).astype(np.uint8)
        maker.append(color_img)

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    maker.save(gifname)
