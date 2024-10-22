import numpy as np
import cv2
import open3d as o3d
import torch

import stereoigev


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
    points_2d, _ = cv2.projectPoints(pcd.points, np.zeros(3), np.zeros(3), right_camera_intrinsics, np.zeros(5))
    points_2d = np.int32(points_2d).reshape(-1, 2)

    # 再投影画像の作成
    img_h, img_w = right_camera_intrinsics[2][2], right_camera_intrinsics[2][2]
    reprojected_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    reprojected_image = cv2.cvtColor(reprojected_image, cv2.COLOR_BGR2RGB)

    # 点を画像に描画
    for pt, c in zip(points_2d, color):
        x, y = pt[0][0], pt[0][1]  # points_2dの形状に合わせて修正
        if 0 <= x < img_w and 0 <= y < img_h:
            reprojected_image[y, x] = (c * 255).astype(np.uint8)

    return reprojected_image


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    REPO_ROOT = Path(__file__).resolve().parent
    DEFAULT_MODEL = REPO_ROOT / "stereoigev/models/sceneflow.pth"
    print(f"{DEFAULT_MODEL=}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_ckpt",
        help="restore checkpoint",
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--save_numpy", default=True, help="save output as numpy arrays")

    parser.add_argument("-l", "--left_imgs", help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument("-r", "--right_imgs", help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    parser.add_argument("--output_directory", help="directory to save output", default="./demo-output/")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--valid_iters", type=int, default=32, help="number of flow-field updates during forward pass")

    # Architecture choices
    parser.add_argument(
        "--hidden_dims", nargs="+", type=int, default=[128] * 3, help="hidden state and context dimensions"
    )
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
        help="correlation volume implementation",
    )
    parser.add_argument(
        "--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders"
    )
    parser.add_argument("--corr_levels", type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument("--n_downsample", type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument("--max_disp", type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    disparity_calculator = stereoigev.DisparityCalculator(args=args)

    with torch.no_grad():
        imfile1 = "test/test-imgs/left/left_motorcycle.png"
        imfile2 = "test/test-imgs/right/right_motorcycle.png"
        bgr1 = cv2.imread(str(imfile1))
        bgr2 = cv2.imread(str(imfile2))
        left_image = bgr1

        torch_image1 = stereoigev.as_torch_img(bgr1, is_BGR_order=True)
        torch_image2 = stereoigev.as_torch_img(bgr2, is_BGR_order=True)
        disparity = disparity_calculator.predict(torch_image1, torch_image2)

        # 近似値
        cx = left_image.shape[1] / 2.0
        cy = left_image.shape[0] / 2.0

        # ダミー
        fx = 1070  # [mm]
        fy = fx

        # カメラパラメータの設定
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # 基線長の設定
        baseline = 0.1  # カメラ間の距離

        right_camera_intrinsics = camera_matrix

        # 点群データの生成
        point_cloud, color = generate_point_cloud(disparity, left_image, camera_matrix, baseline)

        # 再投影
        reprojected_image = reproject_point_cloud(point_cloud, color, right_camera_intrinsics, baseline)

        cv2.imwrite("reprojected.png", reprojected_image)
