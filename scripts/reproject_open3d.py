"""
open3d.t.geometry.PointCloud
https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.create_from_rgbd_image
depth_scale (float, optional, default=1000.0) – The depth is scaled by 1 / depth_scale.
    - mm 単位のものを m 単位に変換する効果を持つ。

depth_max (float, optional, default=3.0) – Truncated at depth_max distance.
    - それより遠方の点を除外する効果を持つ（らしい）。

"""

from typing import Tuple

import open3d as o3d
import numpy as np
import skimage.io


import inspect

from disparity_view.util import dummy_pinhole_camera_intrincic


def shape_of(image) -> Tuple[float, float]:
    if isinstance(image, np.ndarray):
        return image.shape
    else:
        return (image.rows, image.columns)


def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    depth = baseline * focal_length / (disparity + 1e-8)
    return depth


def o3d_gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir, left_name, axis):

    DEPTH_SCALE = 1000.0
    DEPTH_MAX = 10.0
    left_name = Path(left_name)
    shape = left_image.shape

    # disparityからdepth にする関数を抜き出すこと
    intrinsics = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    # 基線長の設定
    baseline = 120  # カメラ間の距離[mm]

    right_camera_intrinsics = intrinsics

    focal_length = 535.4

    depth = disparity_to_depth(disparity, baseline, focal_length)
    depth = np.array(depth, dtype=np.uint16)

    open3d_img = o3d.t.geometry.Image(left_image)
    open3d_depth = o3d.t.geometry.Image(depth)

    rgbd = o3d.t.geometry.RGBDImage(open3d_img, open3d_depth)

    assert isinstance(rgbd, o3d.t.geometry.RGBDImage)
    assert isinstance(intrinsics, o3d.cpu.pybind.core.Tensor)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics=intrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )

    assert isinstance(pcd, o3d.geometry.PointCloud) or isinstance(pcd, o3d.t.geometry.PointCloud)

    scaled_baseline = baseline / DEPTH_SCALE

    if axis == 0:
        extrinsics = [[1, 0, 0, -scaled_baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    elif axis == 1:
        extrinsics = [[1, 0, 0, 0], [0, 1, 0, scaled_baseline], [0, 0, 1, 0], [0, 0, 0, 1]]
    elif axis == 2:
        extrinsics = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, scaled_baseline], [0, 0, 0, 1]]

    rgbd_reproj = pcd.project_to_rgbd_image(
        shape[1], shape[0], intrinsics=intrinsics, extrinsics=extrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / f"depth_{left_name.stem}.png"
    color_out = outdir / f"color_{left_name.stem}.png"

    skimage.io.imsave(color_out, color_legacy)
    print(f"saved {color_out}")
    skimage.io.imsave(depth_out, depth_legacy)
    print(f"saved {depth_out}")


if __name__ == "__main__":
    from pathlib import Path

    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    left_name = "../test/test-imgs/left/left_motorcycle.png"
    left_image = skimage.io.imread(left_name)
    outdir = Path("reprojected_open3d")
    axis = 1
    o3d_gen_right_image(disparity, left_image, outdir, left_name, axis)
