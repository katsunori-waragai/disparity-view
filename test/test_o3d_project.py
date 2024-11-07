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

import disparity_view
from disparity_view.o3d_project import gen_right_image, make_animation_gif
import inspect

from disparity_view.util import safer_imsave


def shape_of(image) -> Tuple[float, float]:
    if isinstance(image, np.ndarray):
        return image.shape
    else:
        return (image.rows, image.columns)


def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float):
    depth = baseline * focal_length / (disparity + 1e-8)
    return depth


def test_o3d_reproject():
    from pathlib import Path

    device = o3d.core.Device("CPU:0")
    imfile1 = "../test/test-imgs/left/left_motorcycle.png"
    left_image = o3d.t.io.read_image(str(imfile1)).to(device)

    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    shape = (left_image.rows, left_image.columns)

    # disparityからdepth にする関数を抜き出すこと
    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    # 基線長の設定
    baseline = 120  # カメラ間の距離[mm]

    right_camera_intrinsics = intrinsic

    focal_length = 535.4

    depth = disparity_to_depth(disparity, baseline, focal_length)

    print(f"{np.max(depth.flatten())=}")

    depth = np.array(depth, dtype=np.uint16)

    open3d_img = o3d.t.geometry.Image(left_image) if isinstance(left_image, np.ndarray) else left_image
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

    rgbd_reproj = pcd.project_to_rgbd_image(shape[1], shape[0], intrinsic, depth_scale=1000.0, depth_max=10.0)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())

    assert len(color_legacy.shape) == 3
    assert color_legacy.shape[2] == 3
    assert len(depth_legacy.shape) == 2
    assert depth_legacy.shape == shape

    assert np.max(depth_legacy.flatten()) > 0
    assert np.max(color_legacy.flatten()) > 0

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

    safer_imsave(str(color_out), color_legacy)
    print(f"saved {color_out}")
    safer_imsave(str(depth_out), depth_legacy)
    print(f"saved {depth_out}")


def test_gen_right_image():
    from pathlib import Path

    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    disparity_name = Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    assert left_name.is_file()
    assert left_name.lstat().st_size > 0
    assert disparity_name.is_file()
    assert disparity_name.lstat().st_size > 0

    axis = 0
    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))

    height, width = disparity.shape[:2]

    assert len(left_image.shape) == 3
    assert len(disparity.shape) == 2

    cx = width / 2.0
    cy = height / 2.0
    focal_length = 1070
    baseline = 120  # [mm]

    cam_param = disparity_view.CameraParameter(width=width, height=height, cx=cx, cy=cy, fx=focal_length, fy=focal_length, baseline=baseline)
    gen_right_image(disparity, left_image, cam_param, Path("out"), left_name, axis=axis)
    outfile = Path("out") / "color_left_motorcycle.png"
    assert outfile.lstat().st_size > 0


if __name__ == "__main__":
    test_o3d_reproject()
    test_gen_right_image()
