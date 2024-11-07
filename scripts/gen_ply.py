from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from disparity_view.util import dummy_pinhole_camera_intrincic
from disparity_view.o3d_project import StereoCamera

def gen_ply(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path, baseline=120.0):
    """
    generate point cloud and save
    """

    stereo_camera = StereoCamera(baseline=120)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    assert isinstance(stereo_camera.pcd, o3d.t.geometry.PointCloud)
    print(f"{stereo_camera.pcd=}")
    outdir.mkdir(exist_ok=True, parents=True)
    plyname = outdir / f"{left_name.stem}_remake.ply"
    print(f"{plyname=}")
    pcd = stereo_camera.pcd.to_legacy()
    o3d.io.write_point_cloud(str(plyname), pcd, format='auto', write_ascii=False, compressed=False, print_progress=True)
    print(f"saved {plyname}")


if __name__ == "__main__":
    """
    python3 gen_ply.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="generate ply file")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    gen_ply(disparity, left_image, Path(args.outdir), left_name)
