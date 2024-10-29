import open3d as o3d
import numpy as np

import inspect


def test_t_read_image():
    device = o3d.core.Device("CPU:0")

    tum_data = o3d.data.SampleTUMRGBDImage()
    depth_path = tum_data.depth_path
    color_path = tum_data.color_path

    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    print(f"{color.rows=} {color.columns=}")
    print(f"{color.channels=}")
    print(f"{color.dtype=}")
    assert depth.rows == color.rows
    assert depth.columns == color.columns
    assert color.channels == 3
    assert color.dtype == o3d.core.Dtype.UInt8
    # print(f"{color.size=}")

    if 0:
        for k, v in inspect.getmembers(color):
            print(k, v)

def test_read_image():
    device = o3d.core.Device("CPU:0")

    tum_data = o3d.data.SampleTUMRGBDImage()
    depth_path = tum_data.depth_path
    color_path = tum_data.color_path

    depth = o3d.io.read_image(depth_path)
    color = o3d.io.read_image(color_path)
    print(f"{color.dimension()=}")
    print(f"{color.get_geometry_type()=}")
    assert hasattr(color, "rows") == False
    assert hasattr(color, "columns") == False
    assert hasattr(color, "channels") == False
    assert hasattr(color, "dtype") == False
    assert np.asarray(depth).shape[:2] == np.asarray(color).shape[:2]
    assert np.asarray(color).shape[2] == 3
    assert np.asarray(color).dtype == np.uint8

