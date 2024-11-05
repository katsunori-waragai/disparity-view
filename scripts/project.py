"""
open3d.t.geometry.PointCloud
https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.create_from_rgbd_image
depth_scale (float, optional, default=1000.0) – The depth is scaled by 1 / depth_scale.
    - mm 単位のものを m 単位に変換する効果を持つ。

depth_max (float, optional, default=3.0) – Truncated at depth_max distance.
    - それより遠方の点を除外する効果を持つ（らしい）。

"""

from pathlib import Path

import numpy as np
import skimage.io

import disparity_view

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
    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    if args.gif:
        disparity_view.make_animation_gif(disparity, left_image, Path(args.outdir), left_name, axis=axis)
    else:
        disparity_view.gen_right_image(disparity, left_image, Path(args.outdir), left_name, axis=axis)
