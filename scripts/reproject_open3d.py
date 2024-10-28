import open3d as o3d
import numpy as np
import cv2
import skimage

import inspect

from disparity_view.util import dummy_pihhole_camera_intrincic


def shape(left_image):
    if isinstance(left_image, np.ndarray):
        return left_image.shape
    else:
        return (left_image.rows, left_image.columns)


if __name__ == "__main__":
    from pathlib import Path

    import inspect

    device = o3d.core.Device("CPU:0")
    imfile1 = "../test/test-imgs/left/left_motorcycle.png"
    left_image = o3d.t.io.read_image(str(imfile1)).to(device)

    if 0:
        for k, v in inspect.getmembers(left_image):
            print(k, v)

    # disparity = o3d.geometry.Image(np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy"))
    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    shape = [left_image.rows, left_image.columns]
    intrinsic = dummy_pihhole_camera_intrincic(shape)
    # 基線長の設定
    baseline = 120  # カメラ間の距離[m]

    right_camera_intrinsics = intrinsic

    assert isinstance(right_camera_intrinsics, o3d.camera.PinholeCameraIntrinsic)

    # 点群データの生成
    # point_cloud = o3d_generate_point_cloud(disparity, left_image, intrinsic, baseline)

    focal_length, _ = intrinsic.get_focal_length()
    depth = baseline * focal_length / (disparity + 1e-8)

    print(f"{np.max(depth.flatten())=}")

    depth = np.array(depth, dtype=np.uint16)

    open3d_img = o3d.geometry.Image(left_image)
    open3d_depth = o3d.geometry.Image(depth)

    o3d.io.write_image("depth_my.png", open3d_depth)

    # 深度マップとカラー画像から点群を作成
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(open3d_img, open3d_depth)

    def shape(left_image):
        if isinstance(left_image, np.ndarray):
            return left_image.shape
        else:
            return (left_image.rows, left_image.columns)

    intrinsic = dummy_pihhole_camera_intrincic(shape(left_image))
    extrinsic = np.array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsic, extrinsic=extrinsic)  # passed without "t."

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
    reprojected_image = skimage.img_as_ubyte(color_legacy)

    if isinstance(reprojected_image, np.ndarray):
        cv2.imwrite("reprojected_open3d.png", reprojected_image)
