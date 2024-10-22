import numpy as np
import cv2
import open3d as o3d


def generate_point_cloud(disparity_map, left_image, camera_matrix, baseline):
    """
    視差マップと左カメラのRGB画像から点群データを生成する関数

    Args:
        disparity_map: 視差マップ (HxW)
        left_image: 左カメラのRGB画像 (HxWx3)
        camera_matrix: カメラの内部パラメータ
        baseline: 基線長

    Returns:
        point_cloud: 点群データ (Nx3)
        color: 点の色情報 (Nx3)
    """

    height, width = disparity_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 視差から深度を計算
    depth = baseline * camera_matrix[0, 0] / (disparity_map + 1e-8)

    # カメラ座標系での3D座標を計算
    X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
    Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
    Z = depth

    # 点群データと色情報を生成
    point_cloud = np.stack((X, Y, Z), axis=2).reshape(-1, 3)
    color = left_image.reshape(-1, 3)

    return point_cloud, color


def reproject_point_cloud(point_cloud, color, right_camera_intrinsics, baseline):
    """
    点群データを右カメラ視点に再投影する関数

    Args:
        point_cloud: 点群データ (Nx3 numpy array)
        color: 点の色情報 (Nx3 numpy array)
        right_camera_intrinsics: 右カメラの内部パラメータ
        baseline: 基線長

    Returns:
        reprojected_image: 再投影画像
    """

    # 点群データをOpen3DのPointCloudに変換
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color)


    # カメラ座標系から画像座標系に変換 (投影)
    points_2d, _ = cv2.projectPoints(point_cloud, np.zeros(3), np.zeros(3), right_camera_intrinsics, np.zeros(5))
    points_2d = np.int32(points_2d).reshape(-1, 2)

    point_cloud[:, 0] += baseline

    # 再投影画像の作成
    print(f"{right_camera_intrinsics=}")
    # img_h, img_w = right_camera_intrinsics[2][2], right_camera_intrinsics[2][2]
    img_w, img_h = 2 * right_camera_intrinsics[0][2], 2 * right_camera_intrinsics[1][2]
    reprojected_image = np.zeros((int(img_h), int(img_w), 3), dtype=np.uint8)

    # 点を画像に描画
    for pt, c in zip(points_2d, color):
        # print(f"{pt=}")
        x, y = pt[0], pt[1]  # points_2dの形状に合わせて修正
        if 0 <= x < img_w and 0 <= y < img_h:
            reprojected_image[y, x] = (c * 255).astype(np.uint8)

    return reprojected_image


if __name__ == "__main__":
    from pathlib import Path
    imfile1 = "test/test-imgs/left/left_motorcycle.png"
    imfile2 = "test/test-imgs/right/right_motorcycle.png"
    bgr1 = cv2.imread(str(imfile1))
    bgr2 = cv2.imread(str(imfile2))
    left_image = bgr1

    disparity = np.load("test/test-imgs/disparity-IGEV/left_motorcycle.npy")

    # 近似値
    cx = left_image.shape[1] / 2.0
    cy = left_image.shape[0] / 2.0

    # ダミー
    fx = 1070  # [mm]
    fy = fx

    # カメラパラメータの設定
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # 基線長の設定
    baseline = 100  # カメラ間の距離[m]

    right_camera_intrinsics = camera_matrix

    # 点群データの生成
    point_cloud, color = generate_point_cloud(disparity, left_image, camera_matrix, baseline)

    print(f"{point_cloud.shape=}")
    print(f"{color.shape=}")
    # 再投影
    reprojected_image = reproject_point_cloud(point_cloud, color, right_camera_intrinsics, baseline)

    cv2.imwrite("reprojected.png", reprojected_image)
