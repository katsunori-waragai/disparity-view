import open3d as o3d
import numpy as np
import cv2

import inspect

from disparity_view.util import dummy_pihhole_camera_intrincic


def o3d_generate_point_cloud(disparity_map, left_image, intrinsic, baseline):
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
    intrinsic = dummy_pihhole_camera_intrincic(left_image.shape)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsic)
    return pcd


def reproject_point_cloud(pcd: o3d.geometry.PointCloud, right_camera_intrinsics: o3d.camera.PinholeCameraIntrinsic, baseline):
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
    pcd.transform([[1, 0, 0, baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # 投影行列の作成

    open3d_right_intrinsic = right_camera_intrinsics

    print(f"{open3d_right_intrinsic=}")

    # 点群を投影
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.get_render_option().point_size = 2
    # # ctr = vis.get_view_control()
    # # ctr.convert_from_pinhole_camera_parameters(parameter=open3d_right_intrinsic)
    # vis.update_geometry()
    # vis.poll_events()
    # vis.capture_screen_image("reprojected_image.png")
    # vis.destroy_window()

    # 画像を読み込み
    # reprojected_image = cv2.imread("reprojected_image.png")

    return reprojected_image


if __name__ == "__main__":
    from pathlib import Path

    imfile1 = "../test/test-imgs/left/left_motorcycle.png"
    bgr1 = cv2.imread(str(imfile1))
    left_image = bgr1

    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    intrinsic = dummy_pihhole_camera_intrincic(left_image.shape)
    # 基線長の設定
    baseline = 120  # カメラ間の距離[m]

    right_camera_intrinsics = intrinsic

    assert isinstance(right_camera_intrinsics, o3d.camera.PinholeCameraIntrinsic)

    # 点群データの生成
    point_cloud = o3d_generate_point_cloud(disparity, left_image, intrinsic, baseline)

    assert isinstance(point_cloud, o3d.geometry.PointCloud)

    # 再投影
    reprojected_image = reproject_point_cloud(point_cloud, right_camera_intrinsics, baseline)
    if isinstance(reprojected_image, np.ndarray):
        cv2.imwrite("reprojected_open3d.png", reprojected_image)
