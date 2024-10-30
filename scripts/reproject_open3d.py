"""
open3d.t.geometry.PointCloud
https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.create_from_rgbd_image
depth_scale (float, optional, default=1000.0) – The depth is scaled by 1 / depth_scale.
    - mm 単位のものを m 単位に変換する効果を持つ。

depth_max (float, optional, default=3.0) – Truncated at depth_max distance.
    - それより遠方の点を除外する効果を持つ（らしい）。

"""

import open3d as o3d
import numpy as np
import skimage.io

from disparity_view.o3d_reprojection import disparity_to_depth, dummy_o3d_camera_matrix, as_extrinsics

DEPTH_SCALE = 1000.0
DEPTH_MAX = 10.0


def o3d_reproject_from_left_and_disparity(left_image, disparity, intrinsics, baseline=120.0, tvec=np.array((0, 0, 0))):
    shape = left_image.shape
    focal_length = np.asarray(intrinsics)[0, 0]
    depth = disparity_to_depth(disparity, baseline, focal_length)

    open3d_img = o3d.t.geometry.Image(left_image)
    open3d_depth = o3d.t.geometry.Image(depth)

    rgbd = o3d.t.geometry.RGBDImage(open3d_img, open3d_depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics=intrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )

    extrinsics = as_extrinsics(tvec)
    rgbd_reproj = pcd.project_to_rgbd_image(
        shape[1], shape[0], intrinsics=intrinsics, extrinsics=extrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())

    return color_legacy, depth_legacy


def o3d_gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir, left_name, axis):
    left_name = Path(left_name)
    shape = left_image.shape

    intrinsics = dummy_o3d_camera_matrix(shape, focal_length=535.4)
    baseline = 120  # カメラ間の距離[mm] 基線長

    scaled_baseline = baseline / DEPTH_SCALE
    if axis == 0:
        tvec = np.array([[-scaled_baseline, 0.0, 0.0]])
    elif axis == 1:
        tvec = np.array([[0.0, scaled_baseline, 0.0]])
    elif axis == 2:
        tvec = np.array([[0.0, 0.0, scaled_baseline]])

    color_legacy, depth_legacy = o3d_reproject_from_left_and_disparity(
        left_image, disparity, intrinsics, baseline=baseline, tvec=tvec
    )
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
    axis = 2
    o3d_gen_right_image(disparity, left_image, outdir, left_name, axis)
