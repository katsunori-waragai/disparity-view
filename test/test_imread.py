import numpy as np
import open3d as o3d
import skimage.io


def test_imread():
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth_path = tum_data.depth_path
    color_path = tum_data.color_path
    cvdepth = skimage.io.imread(depth_path)
    cvcolor = skimage.io.imread(color_path)
    assert cvdepth.dtype == np.uint16
    assert len(cvdepth.shape) == 2

    assert cvcolor.dtype == np.uint8
    assert len(cvcolor.shape) == 3
    print(f"{cvcolor.shape=}")
    assert cvcolor.shape[2] in (3, 4)
