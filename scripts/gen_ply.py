from pathlib import Path

import numpy as np
import skimage.io

import disparity_view
from disparity_view.o3d_project import gen_ply

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
    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    cam_param = disparity_view.CameraParameter.load_json(args.json)
    gen_ply(disparity, left_image, cam_param, Path(args.outdir), left_name)
