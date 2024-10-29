from typing import Tuple

import open3d as o3d
import numpy as np
import cv2
import skimage.io


import inspect

from disparity_view.util import dummy_pinhole_camera_intrincic


def shape_of(image) -> Tuple[float, float]:
    if isinstance(image, np.ndarray):
        return image.shape
    else:
        return (image.rows, image.columns)


def o3d_generate_point_cloud(
    disparity_map, left_image, intrinsic: o3d.camera.PinholeCameraIntrinsic, baseline: float
) -> o3d.geometry.PointCloud:
    """
    視差マップと左カメラのRGB画像から点群データを生成する

    Args:
        disparity_map: 視差マップ (HxW)
        left_image: 左カメラのRGB画像 (HxWx3)
        camera_matrix: カメラの内部パラメータ
        baseline: 基線長

    Returns:
        pcd: Open3DのPointCloudオブジェクト
    """

    # 視差から深度を計算
    focal_length, _ = intrinsic.get_focal_length()
    depth = baseline * focal_length / (disparity_map + 1e-8)

    open3d_img = o3d.geometry.Image(left_image)
    open3d_depth = o3d.geometry.Image(depth)
    # 深度マップとカラー画像から点群を作成
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(open3d_img, open3d_depth)

    intrinsic = dummy_pinhole_camera_intrincic(shape(left_image))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsic)
    return pcd


def reproject_point_cloud(
    pcd: o3d.geometry.PointCloud, right_camera_intrinsics: o3d.camera.PinholeCameraIntrinsic, baseline: float
):
    """
    点群データを右カメラ視点に再投影する

    Args:
        pcd: Open3DのPointCloudオブジェクト
        right_camera_intrinsics: 右カメラの内部パラメータ
        baseline: 基線長

    Returns:
        reprojected_image: 再投影画像
    """

    # 視点変換（平行移動）
    device = o3d.core.Device("CPU:0")
    pcd.transform([[1, 0, 0, baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    open3d_right_intrinsic = right_camera_intrinsics

    print(f"{open3d_right_intrinsic=}")

    for k, v in inspect.getmembers(pcd):
        if inspect.ismethod(v):
            print(k, v)

    shape = [left_image.rows, left_image.columns]

    # AttributeError: 'open3d.cpu.pybind.geometry.PointCloud' object has no attribute
    rgbd_reproj = pcd.project_to_rgbd_image(shape[1], shape[0], intrinsic, depth_scale=5000.0, depth_max=10.0)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    print(f"{color_legacy.dtype=}")
    print(f"{depth_legacy.dtype=}")
    color_img = skimage.img_as_ubyte(color_legacy)

    return color_img


if __name__ == "__main__":
    from pathlib import Path

    import inspect

    device = o3d.core.Device("CPU:0")
    imfile1 = "../test/test-imgs/left/left_motorcycle.png"
    left_image = o3d.t.io.read_image(str(imfile1)).to(device)

    if 0:
        for k, v in inspect.getmembers(left_image):
            print(k, v)

    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    shape = [left_image.rows, left_image.columns]

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    # 基線長の設定
    baseline = 120  # カメラ間の距離[m]

    right_camera_intrinsics = intrinsic

    focal_length = 535.4
    depth = baseline * focal_length / (disparity + 1e-8)

    print(f"{np.max(depth.flatten())=}")

    depth = np.array(depth, dtype=np.uint16)

    open3d_img = o3d.t.geometry.Image(left_image)
    open3d_depth = o3d.t.geometry.Image(depth)

    o3d.t.io.write_image("depth_my.png", open3d_depth)

    # 深度マップとカラー画像から点群を作成
    rgbd = o3d.t.geometry.RGBDImage(open3d_img, open3d_depth)

    assert isinstance(rgbd, o3d.t.geometry.RGBDImage)
    assert isinstance(intrinsic, o3d.cpu.pybind.core.Tensor)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics=intrinsic, depth_scale=5000.0, depth_max=10.0)

    assert isinstance(pcd, o3d.geometry.PointCloud) or isinstance(pcd, o3d.t.geometry.PointCloud)

    pcd.project_to_rgbd_image

    # 再投影
    device = o3d.core.Device("CPU:0")
    pcd.transform([[1, 0, 0, baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    open3d_right_intrinsic = right_camera_intrinsics

    print(f"{open3d_right_intrinsic=}")

    shape = [left_image.rows, left_image.columns]
    rgbd_reproj = pcd.project_to_rgbd_image(shape[1], shape[0], intrinsic, depth_scale=5000.0, depth_max=10.0)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    print(f"{color_legacy.dtype=}")
    print(f"{depth_legacy.dtype=}")
    outdir = Path("reprojected_open3d")
    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / "depth.png"
    color_out = outdir / "color.png"

    skimage.io.imsave(color_out, color_legacy)
    print(f"saved {color_out}")
    skimage.io.imsave(depth_out, depth_legacy)
    print(f"saved {depth_out}")
