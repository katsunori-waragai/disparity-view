import open3d as o3d

import inspect

def test_read_image():
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
        for k, v in  inspect.getmembers(color):
            print(k, v)

if __name__ == "__main__":
    test_read_image()
