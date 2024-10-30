"""
open3d.t.geometry.PointCloud
https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.create_from_rgbd_image
depth_scale (float, optional, default=1000.0) – The depth is scaled by 1 / depth_scale.
    - mm 単位のものを m 単位に変換する効果を持つ。

depth_max (float, optional, default=3.0) – Truncated at depth_max distance.
    - それより遠方の点を除外する効果を持つ（らしい）。

"""

import numpy as np
import skimage.io

from disparity_view.o3d_reprojection import o3d_gen_right_image

if __name__ == "__main__":
    from pathlib import Path

    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    left_name = "../test/test-imgs/left/left_motorcycle.png"
    left_image = skimage.io.imread(left_name)
    outdir = Path("reprojected_open3d")
    axis = 2
    o3d_gen_right_image(disparity, left_image, outdir, left_name, axis)
