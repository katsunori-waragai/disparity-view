from pathlib import Path

import open3d as o3d
import numpy as np
import skimage.io


def read_and_reproject(depth_path: str, color_path: str):
    """
    read depth, color from files, and reproject from constructed point cloud
    """
    print(f"{depth_path=}")
    print(f"{color_path=}")
    device = o3d.core.Device("CPU:0")
    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    assert depth.rows == color.rows
    assert depth.columns == color.columns
    print(f"{color.rows=} {color.columns=}")

    width = color.columns
    height = color.rows

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, depth_scale=5000.0, depth_max=10.0)
    rgbd_reproj = pcd.project_to_rgbd_image(width, height, intrinsic, depth_scale=5000.0, depth_max=10.0)

    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    print(f"{color_legacy.dtype=}")
    print(f"{depth_legacy.dtype=}")
    outdir = Path("reprojected")
    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / "depth.png"
    color_out = outdir / "color.png"
    skimage.io.imsave(color_out, color_legacy)
    skimage.io.imsave(depth_out, depth_legacy)

    print(f"saved {color_out} {depth_out}")

if __name__ == "__main__":
    """
    pcd.project_to_rgbd_imageの使い方を確認するためのスクリプト
    """
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth_path = tum_data.depth_path
    color_path = tum_data.color_path

    read_and_reproject(depth_path, color_path)
