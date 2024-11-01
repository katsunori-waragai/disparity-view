from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import skimage.io
import cv2

from tqdm import tqdm

from .animation_gif import AnimationGif
from .util import dummy_camera_matrix


DEPTH_SCALE = 1000.0
DEPTH_MAX = 10.0


def shape_of(image) -> Tuple[float, float]:
    if isinstance(image, np.ndarray):
        return image.shape
    else:
        return (image.rows, image.columns)


def depth_from_disparity(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    depth = baseline * float(focal_length) / (disparity + 1e-8)
    return depth


def depth_by_disparity_and_intrinsics(disparity: np.ndarray, baseline: float, intrinsics: np.ndarray) -> np.ndarray:
    focal_length = np.asarray(intrinsics)[0, 0]
    return depth_from_disparity(disparity, baseline, focal_length)


def as_extrinsics(tvec: np.ndarray, rot_mat=np.eye(3, dtype=float)) -> np.ndarray:
    if len(tvec.shape) == 1:
        tvec = np.ndarray([tvec])
    return np.vstack((np.hstack((rot_mat, tvec.T)), [0, 0, 0, 1]))


def generate_point_cloud(
    disparity: np.ndarray, left_image: np.ndarray, intrinsics: np.ndarray, baseline: float
) -> o3d.t.geometry.PointCloud:
    depth = depth_by_disparity_and_intrinsics(disparity, baseline, intrinsics)
    rgbd = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(left_image), o3d.t.geometry.Image(depth))
    return o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics=intrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )


def reproject_point_cloud(
    pcd: o3d.t.geometry.PointCloud, intrinsics: np.ndarray, tvec: np.ndarray
) -> o3d.t.geometry.RGBDImage:
    extrinsics = as_extrinsics(tvec)
    img_w, img_h = int(2 * intrinsics[0][2]), int(2 * intrinsics[1][2])
    shape = [img_h, img_w]

    return pcd.project_to_rgbd_image(
        shape[1], shape[0], intrinsics=intrinsics, extrinsics=extrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )


def reproject_from_left_and_disparity(
    left_image: np.ndarray, disparity: np.ndarray, intrinsics: np.ndarray, baseline=120.0, tvec=np.array((0, 0, 0))
) -> Tuple[np.ndarray, np.ndarray]:
    shape = left_image.shape

    pcd = generate_point_cloud(disparity, left_image, intrinsics, baseline)
    rgbd_reproj = reproject_point_cloud(pcd, intrinsics, tvec=tvec)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    assert isinstance(color_legacy, np.ndarray)
    assert isinstance(depth_legacy, np.ndarray)

    assert color_legacy.shape[:2] == depth_legacy.shape

    assert np.max(color_legacy.flatten()) > 0
    assert np.max(depth_legacy.flatten()) > 0

    return color_legacy, depth_legacy


def gen_tvec(scaled_shift: float, axis: int) -> np.ndarray:
    assert axis in (0, 1, 2)
    if axis == 0:
        tvec = np.array([[-scaled_shift, 0.0, 0.0]])
    elif axis == 1:
        tvec = np.array([[0.0, scaled_shift, 0.0]])
    elif axis == 2:
        tvec = np.array([[0.0, 0.0, scaled_shift]])
    return tvec


def gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir, left_name, axis):
    left_name = Path(left_name)
    shape = left_image.shape

    intrinsics = dummy_camera_matrix(shape, focal_length=535.4)
    baseline = 120  # カメラ間の距離[mm] 基線長

    scaled_baseline = baseline / DEPTH_SCALE

    tvec = gen_tvec(scaled_baseline, axis)
    color_legacy, depth_legacy = reproject_from_left_and_disparity(
        left_image, disparity, intrinsics, baseline=baseline, tvec=tvec
    )
    assert isinstance(color_legacy, np.ndarray)
    assert isinstance(depth_legacy, np.ndarray)
    assert color_legacy.shape[:2] == depth_legacy.shape
    assert np.max(color_legacy.flatten()) > 0
    assert np.max(depth_legacy.flatten()) > 0

    print(f"{color_legacy.dtype=}")
    print(f"{depth_legacy.dtype=}")

    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / f"depth_{left_name.stem}.png"
    color_out = outdir / f"color_{left_name.stem}.png"

    skimage.io.imsave(str(color_out), color_legacy)
    print(f"saved {color_out}")
    skimage.io.imsave(str(depth_out), depth_legacy)
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
        reprojected_rgbdimage = reproject_point_cloud(pcd, camera_matrix, tvec=tvec)
        color_img = np.asarray(reprojected_rgbdimage.color.to_legacy())
        color_img = (color_img * 255).astype(np.uint8)
        maker.append(color_img)

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    maker.save(gifname)
