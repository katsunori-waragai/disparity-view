from pathlib import Path
import numpy as np

from disparity_view.o3d_project import gen_tvec, DEPTH_SCALE
from disparity_view.o3d_project import as_extrinsics
from disparity_view.projection_class import StereoCamera
from disparity_view.util import  safer_imsave

import skimage.io

if __name__ == "__main__":
    """
    python3 project.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="reprojector")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--axis", default=0, help="axis to shift(0: to right, 1: to upper, 2: to far)")
    parser.add_argument("--gif", action="store_true", help="git animation")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()

    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    axis = int(args.axis)
    outdir = Path(args.outdir)

    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))

    height, width = disparity[:2]

    stereo_camera = StereoCamera()
    shape = disparity.shape
    stereo_camera.set_camera_matrix(shape=shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    scaled_baseline = 120 / DEPTH_SCALE # [mm]
    tvec=gen_tvec(scaled_shift=scaled_baseline, axis=0)
    extrinsics = as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = outdir / "color_left_motorcycle.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    assert outfile.lstat().st_size > 0
