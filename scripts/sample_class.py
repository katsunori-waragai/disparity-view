from pathlib import Path
import numpy as np

from disparity_view.projection_class import StereoCamera
from disparity_view.util import  safer_imsave

if __name__ == "__main__":
    import skimage.io
    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    disparity_name = Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    outdir = Path("out")

    assert left_name.is_file()
    assert left_name.lstat().st_size > 0
    assert disparity_name.is_file()
    assert disparity_name.lstat().st_size > 0

    axis = 0
    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))

    assert len(left_image.shape) == 3
    assert len(disparity.shape) == 2
    height, width = disparity[:2]

    stereo_camera = StereoCamera()
    shape = disparity.shape
    stereo_camera.set_camera_matrix(shape=shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    projected = stereo_camera.project_to_rgbd_image(width, height)
    color_legacy = np.asarray(projected.color.to_legacy())
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = Path("out_class") / "color_left_motorcycle.png"
    safer_imsave(str(outfile), color_legacy)
    assert outfile.lstat().st_size > 0
