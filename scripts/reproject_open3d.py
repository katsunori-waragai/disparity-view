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


if __name__ == "__main__":
    from pathlib import Path

    device = o3d.core.Device("CPU:0")
    imfile1 = "../test/test-imgs/left/left_motorcycle.png"
    left_image = o3d.t.io.read_image(str(imfile1)).to(device)

    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    shape = [left_image.rows, left_image.columns]

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    # 基線長の設定
    baseline = 120  # カメラ間の距離[mm]

    right_camera_intrinsics = intrinsic

    focal_length = 535.4
    depth = baseline * focal_length / (disparity + 1e-8)

    print(f"{np.max(depth.flatten())=}")

    depth = np.array(depth, dtype=np.uint16)

    open3d_img = o3d.t.geometry.Image(left_image)
    open3d_depth = o3d.t.geometry.Image(depth)

    o3d.t.io.write_image("depth_my.png", open3d_depth)

    rgbd = o3d.t.geometry.RGBDImage(open3d_img, open3d_depth)

    assert isinstance(rgbd, o3d.t.geometry.RGBDImage)
    assert isinstance(intrinsic, o3d.cpu.pybind.core.Tensor)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics=intrinsic, depth_scale=1000.0, depth_max=10.0
    )

    assert isinstance(pcd, o3d.geometry.PointCloud) or isinstance(pcd, o3d.t.geometry.PointCloud)

    pcd.project_to_rgbd_image

    device = o3d.core.Device("CPU:0")
    baseline = 120.0 / 1000.0
    pcd.transform([[1, 0, 0, -baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    open3d_right_intrinsic = right_camera_intrinsics

    print(f"{open3d_right_intrinsic=}")

    shape = [left_image.rows, left_image.columns]
    rgbd_reproj = pcd.project_to_rgbd_image(shape[1], shape[0], intrinsic, depth_scale=1000.0, depth_max=10.0)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    print(f"{color_legacy.dtype=}")
    print(f"{depth_legacy.dtype=}")
    print(f"{np.max(depth_legacy.flatten())=}")
    print(f"{np.max(color_legacy.flatten())=}")
    print(f"{np.min(depth_legacy.flatten())=}")
    print(f"{np.min(color_legacy.flatten())=}")
    outdir = Path("reprojected_open3d")
    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / "depth.png"
    color_out = outdir / "color.png"

    skimage.io.imsave(color_out, color_legacy)
    print(f"saved {color_out}")
    skimage.io.imsave(depth_out, depth_legacy)
    print(f"saved {depth_out}")
