from pathlib import Path

import numpy as np
import skimage.io

import disparity_view
from disparity_view.util import safer_imsave


def test_stereo_camera_class():
    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    disparity_name = Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    json_file = Path("../test/zed-imgs/camera_param.json")

    axis = 0
    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))

    camera_param = disparity_view.CameraParameter.load_json(json_file)
    stereo_camera = disparity_view.StereoCamera.create_from_camera_param(camera_param)
    shape = disparity.shape
    stereo_camera.set_camera_matrix(shape=shape, focal_length=1070)
    scaled_baseline = stereo_camera.scaled_baseline()  # [mm] to [m]
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    tvec = disparity_view.gen_tvec(scaled_shift=scaled_baseline, axis=axis)
    extrinsics = disparity_view.as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = Path("out_class") / "color_left_motorcycle.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    assert outfile.lstat().st_size > 0
    assert np.max(color_legacy.flatten()) > 0


if __name__ == "__main__":
    test_stereo_camera_class()
