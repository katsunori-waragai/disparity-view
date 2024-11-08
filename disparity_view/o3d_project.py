from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from tqdm import tqdm

from .animation_gif import AnimationGif
from .util import create_camera_matrix, safer_imsave
from .cam_param import CameraParameter


DEPTH_SCALE = 1000.0
DEPTH_MAX = 10.0


def as_extrinsics(tvec: np.ndarray, rot_mat=np.eye(3, dtype=float)) -> np.ndarray:
    if len(tvec.shape) == 1:
        tvec = np.ndarray([tvec])
    return np.vstack((np.hstack((rot_mat, tvec.T)), [0, 0, 0, 1]))


def gen_tvec(scaled_shift: float, axis: int) -> np.ndarray:
    assert axis in (0, 1, 2)
    if axis == 0:
        tvec = np.array([[-scaled_shift, 0.0, 0.0]])
    elif axis == 1:
        tvec = np.array([[0.0, scaled_shift, 0.0]])
    elif axis == 2:
        tvec = np.array([[0.0, 0.0, scaled_shift]])
    return tvec


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
        camera_param = CameraParameter.load_json(json)
        self.left_camera_matrix = camera_param.to_matrix()
        self.right_camera_matrix = self.left_camera_matrix
        self.baseline = camera_param.baseline

    def set_camera_matrix(self, shape: np.ndarray, focal_length: float):
        self.shape = shape
        self.left_camera_matrix = o3d.core.Tensor(create_camera_matrix(shape, focal_length=focal_length))
        self.right_camera_matrix = self.left_camera_matrix

    def generate_point_cloud(self, disparity_map: np.ndarray, left_image: np.ndarray):
        def get_fx(intrinsics):
            return intrinsics.numpy()[0, 0] if not isinstance(intrinsics, np.ndarray) else intrinsics[0, 0]

        if disparity_map.shape[:2] != left_image.shape[:2]:
            print(f"{disparity_map.shape=} {left_image.shape[:2]=}")
        assert disparity_map.shape[:2] == left_image.shape[:2]
        self._fix_camera_param_if_needed(disparity_map)

        focal_length = get_fx(self.left_camera_matrix)
        depth = self.baseline * float(focal_length) / (disparity_map + 1e-8)
        rgbd = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(left_image), o3d.t.geometry.Image(depth))
        return o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics=self.left_camera_matrix, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
        )

    def _fix_camera_param_if_needed(self, disparity_map):
        height, width = disparity_map.shape[:2]
        cx = self.left_camera_matrix[0, 2].numpy()
        cy = self.left_camera_matrix[1, 2].numpy()
        if abs(width / 2.0 - cx) > 1.0:
            print(f"Warn: mismatched image width and fx: {width=} {cx}=")
            self.left_camera_matrix[0, 2] = width / 2.0
        if abs(height / 2.0 - cy) > 1.0:
            print(f"Warn: mismatched image height and fy: {height=} {cy=}")
            self.left_camera_matrix[1, 2] = height / 2.0

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


def gen_right_image(
    disparity: np.ndarray, left_image: np.ndarray, cam_param: CameraParameter, outdir: Path, left_name: Path, axis=0
):
    stereo_camera = StereoCamera(baseline=cam_param.baseline)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=cam_param.fx)
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


def make_animation_gif(
    disparity: np.ndarray, left_image: np.ndarray, cam_param: CameraParameter, outdir: Path, left_name: Path, axis=0
):
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

    stereo_camera = StereoCamera(baseline=cam_param.baseline)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=cam_param.fx)
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
