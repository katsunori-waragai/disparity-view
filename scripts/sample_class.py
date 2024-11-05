from pathlib import Path
import numpy as np

from disparity_view.o3d_project import gen_tvec, DEPTH_SCALE
from disparity_view.o3d_project import as_extrinsics
from disparity_view.projection_class import StereoCamera
from disparity_view.util import  safer_imsave

if __name__ == "__main__":
    import skimage.io
    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    disparity_name = Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    outdir = Path("out_class")

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
    scaled_baseline = 120 / DEPTH_SCALE # [mm]
    tvec=gen_tvec(scaled_shift=scaled_baseline, axis=0)
    extrinsics = as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = Path("out_class") / "color_left_motorcycle.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    assert outfile.lstat().st_size > 0
