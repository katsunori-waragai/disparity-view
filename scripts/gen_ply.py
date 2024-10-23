from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

def dummy_camera_matrix(image_shape) -> np.ndarray:
    # 近似値
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    # ダミー
    fx = 1070  # [mm]
    fy = fx

    # カメラパラメータの設定
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return camera_matrix

def gen_ply(disparity, left_image, outdir, left_name):
    camera_parameter = CameraParameter.load_json(json_file)

    width = camera_parameter.width
    height = camera_parameter.height
    fx = camera_parameter.fx
    fy = camera_parameter.fy
    cx = camera_parameter.cx
    cy = camera_parameter.cy

    left_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    plyname = disparity_name.with_suffix(".ply")
    disparity = np.load(str(disparity_name))
    baseline = camera_parameter.baseline
    focal_length = camera_parameter.fx
    depth = baseline * focal_length / disparity

    rgb = o3d.io.read_image(str(left_name))
    open3d_depth = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, open3d_depth)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, left_cam_intrinsic)
    if args.save:
        o3d.io.write_point_cloud(str(plyname), pcd)


if __name__ == "__main__":
    """
    python3 reproject.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="reprojector")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    gen_ply(disparity, left_image, Path(args.outdir), left_name)
