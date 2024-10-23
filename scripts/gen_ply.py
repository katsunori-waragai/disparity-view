from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def dummy_camera_matrix(image_shape) -> np.ndarray:
    # 近似値
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    # ダミー
    fx = 1070  # [pixel]
    fy = fx

    # カメラパラメータの設定
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return camera_matrix


def gen_ply(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path, baseline=100.0 ):

    height, width = left_image.shape[:2]

    camera_matrix = dummy_camera_matrix(left_image.shape)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    left_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    focal_length = fx
    depth = baseline * focal_length / disparity

    rgb = o3d.io.read_image(str(left_name))
    open3d_depth = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, open3d_depth)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, left_cam_intrinsic)
    outdir.mkdir(exist_ok=True, parents=True)
    plyname = outdir / f"{left_name.stem}.ply"
    o3d.io.write_point_cloud(str(plyname), pcd)
    print(f"saved {plyname}")


if __name__ == "__main__":
    """
    python3 gen_ply.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="genarate ply file")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    gen_ply(disparity, left_image, Path(args.outdir), left_name)
