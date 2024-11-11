from pathlib import Path

import numpy as np
import skimage.io
import open3d as o3d

import disparity_view
from disparity_view.o3d_project import gen_ply

if __name__ == "__main__":
    """
    python3 gen_ply.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png ../test/zed-imgs/camera_param.json
    """

def test_gen_ply():
    disparity_name = Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    cam_param = disparity_view.CameraParameter.load_json("../test/zed-imgs/zed_camera_param.json")
    gen_ply(disparity, left_image, cam_param, Path("output"), left_name)
    plyname = Path("output") / f"{left_name.stem}.ply"
    assert plyname.is_file()
    assert plyname.lstat().st_size > 0
    pcd = o3d.io.read_point_cloud(str(plyname))


if __name__ == "__main__":
    test_gen_ply()
