from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import skimage.io
import cv2

from tqdm import tqdm

from . import CameraParameter
from .animation_gif import AnimationGif
from .util import dummy_camera_matrix, safer_imsave


DEPTH_SCALE = 1000.0
DEPTH_MAX = 10.0


def shape_of(image) -> Tuple[float, float]:
    if isinstance(image, np.ndarray):
        return image.shape
    else:
        return (image.rows, image.columns)


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


def depth_from_disparity(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    depth = baseline * float(focal_length) / (disparity + 1e-8)
    return depth


def depth_by_disparity_and_intrinsics(disparity: np.ndarray, baseline: float, intrinsics: np.ndarray) -> np.ndarray:
    if not isinstance(intrinsics, np.ndarray):
        focal_length = intrinsics.numpy()[0, 0]
    else:
        focal_length = np.asarray(intrinsics)[0, 0]
    if not isinstance(focal_length, float):
        print(f"{focal_length}")
    assert isinstance(focal_length, float)
    return depth_from_disparity(disparity, baseline, focal_length)


def generate_point_cloud(
    disparity: np.ndarray, left_image: np.ndarray, intrinsics: np.ndarray, baseline: float
) -> o3d.t.geometry.PointCloud:
    depth = depth_by_disparity_and_intrinsics(disparity, baseline, intrinsics)
    rgbd = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(left_image), o3d.t.geometry.Image(depth))
    return o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics=intrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )


def project_point_cloud(
    pcd: o3d.t.geometry.PointCloud, intrinsics: np.ndarray, tvec: np.ndarray
) -> o3d.t.geometry.RGBDImage:
    extrinsics = as_extrinsics(tvec)
    img_w, img_h = int(2 * intrinsics[0][2]), int(2 * intrinsics[1][2])
    shape = [img_h, img_w]

    return pcd.project_to_rgbd_image(
        shape[1], shape[0], intrinsics=intrinsics, extrinsics=extrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )


def project_from_left_and_disparity(
    left_image: np.ndarray, disparity: np.ndarray, intrinsics: np.ndarray, baseline=120.0, tvec=np.array((0, 0, 0))
) -> Tuple[np.ndarray, np.ndarray]:  ## replace by class
    shape = left_image.shape

    pcd = generate_point_cloud(disparity, left_image, intrinsics, baseline)
    rgbd_reproj = project_point_cloud(pcd, intrinsics, tvec=tvec)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    return color_legacy, depth_legacy


def gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir, left_name, axis):  ## replace by class
    left_name = Path(left_name)
    shape = left_image.shape

    intrinsics = dummy_camera_matrix(shape, focal_length=535.4)
    baseline = 120  # カメラ間の距離[mm] 基線長

    scaled_baseline = baseline / DEPTH_SCALE

    tvec = gen_tvec(scaled_baseline, axis)
    color_legacy, depth_legacy = project_from_left_and_disparity(
        left_image, disparity, intrinsics, baseline=baseline, tvec=tvec
    )

    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / f"depth_{left_name.stem}.png"
    color_out = outdir / f"color_{left_name.stem}.png"

    safer_imsave(str(color_out), color_legacy)
    print(f"saved {color_out}")
    safer_imsave(str(depth_out), depth_legacy)
    print(f"saved {depth_out}")


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
    camera_matrix = dummy_camera_matrix(left_image.shape)
    baseline = 120.0  # [mm] same to zed2i

    pcd = generate_point_cloud(disparity, left_image, camera_matrix, baseline)

    maker = AnimationGif()
    n = 16
    for i in tqdm(range(n + 1)):
        scaled_baseline = baseline / DEPTH_SCALE
        tvec = gen_tvec(scaled_baseline * i / n, axis)
        projected_rgbdimage = project_point_cloud(pcd, camera_matrix, tvec=tvec)
        color_img = np.asarray(projected_rgbdimage.color.to_legacy())
        color_img = (color_img * 255).astype(np.uint8)
        maker.append(color_img)

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    maker.save(gifname)


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
