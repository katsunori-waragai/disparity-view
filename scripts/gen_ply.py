from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

import disparity_view


def gen_ply(disparity: np.ndarray, left_image: np.ndarray, cam_param, outdir: Path, left_name: Path):
    """
    generate point cloud and save
    """

    stereo_camera = disparity_view.StereoCamera(baseline=cam_param.baseline)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=cam_param.fx)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    assert isinstance(stereo_camera.pcd, o3d.t.geometry.PointCloud)
    print(f"{stereo_camera.pcd=}")
    outdir.mkdir(exist_ok=True, parents=True)
    plyname = outdir / f"{left_name.stem}.ply"
    print(f"{plyname=}")
    pcd = stereo_camera.pcd.to_legacy()
    o3d.io.write_point_cloud(str(plyname), pcd, write_ascii=False, compressed=False, print_progress=True)
    print(f"saved {plyname}")


if __name__ == "__main__":
    """
    python3 gen_ply.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png ../test/zed-imgs/camera_param.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="generate ply file")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("json", help="json file for camera parameter")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    cam_param = disparity_view.CameraParameter.load_json(args.json)
    gen_ply(disparity, left_image, cam_param, Path(args.outdir), left_name)
