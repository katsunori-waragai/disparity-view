import numpy as np
import open3d as o3d
import skimage.io


def test_imread():
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth_path = tum_data.depth_path
    color_path = tum_data.color_path
    cvdepth = skimage.io.imread(depth_path)
    cvcolor = skimage.io.imread(color_path)
    print(f"{cvdepth.dtype=}")
#    assert cvdepth.dtype in (np.uint16, np.uint32)
    assert len(cvdepth.shape) == 2
#    assert cvcolor.dtype == np.uint8
    assert len(cvcolor.shape) == 3
    assert cvcolor.shape[2] in (3, 4)

if __name__ == "__main__":
    test_imread()
