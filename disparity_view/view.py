"""
library to view disparity npy files.

依存性：
    3dの可視化にはOpen3Dを用いている。
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


from .depth_to_normal import DepthToNormalMap
from .cam_param import CameraParameter


def finitemax(depth: np.ndarray) -> float:
    return np.nanmax(depth[np.isfinite(depth)])


def finitemin(depth: np.ndarray) -> float:
    return np.nanmin(depth[np.isfinite(depth)])


def normalize_image(depth_raw: np.ndarray, vmax=None, vmin=None) -> np.ndarray:
    vmin = finitemin(depth_raw) if vmin is None else vmin
    vmax = finitemax(depth_raw) if vmax is None else vmax
    depth_raw = (depth_raw - vmin) / (vmax - vmin) * 255.0
    depth_raw = depth_raw.astype(np.uint8)  # depth_raw might have NaN, PosInf, NegInf.
    return depth_raw


def as_colorimage(depth_raw: np.ndarray, vmin=None, vmax=None, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    apply color mapping with vmin, vmax
    """
    depth_raw = normalize_image(depth_raw, vmax, vmin)
    return cv2.applyColorMap(depth_raw, colormap)


def as_gray(depth_raw: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    apply color mapping with vmin, vmax
    """
    gray = normalize_image(depth_raw, vmax, vmin)
    return cv2.merge((gray, gray, gray))


def resize_image(image: np.ndarray, rate: float) -> np.ndarray:
    H, W = image.shape[:2]
    return cv2.resize(image, (int(W * rate), int(H * rate)))


def as_matrix(chw_array: np.ndarray) -> np.ndarray:
    H_, W_ = chw_array.shape[-2:]
    return np.reshape(chw_array, (H_, W_))


def get_dirs(captured_dir: Path) -> Tuple[Path, Path, Path]:
    leftdir = captured_dir / "left"
    rightdir = captured_dir / "right"
    disparity_dir = captured_dir / "zed-disparity"
    for p in (leftdir, rightdir, disparity_dir):
        p.mkdir(exist_ok=True, parents=True)
    return leftdir, rightdir, disparity_dir


def depth_overlay(gray: np.ndarray, color_depth: np.ndarray) -> np.ndarray:
    """
    overlay color_depth to gray scale image
    """
    assert len(gray.shape) == 2
    assert len(color_depth.shape) == 3
    gray2 = cv2.merge((gray, gray, gray)).astype(np.uint16)
    color_depth2 = color_depth.astype(np.uint16)
    return ((gray2 + color_depth2) / 2).astype(np.uint8)


def view_by_colormap(args):
    captured_dir = Path(args.captured_dir)
    leftdir, rightdir, disparity_dir = get_dirs(captured_dir)
    sec = args.sec
    vmax = args.vmax
    vmin = args.vmin

    left_images = sorted(leftdir.glob("**/*.png"))
    disparity_npys = sorted(disparity_dir.glob("**/*.npy"))
    cv2.namedWindow("left depth", cv2.WINDOW_NORMAL)
    for leftname, disparity_name in tqdm(list(zip(left_images, disparity_npys))):
        print(leftname, disparity_name)
        image = cv2.imread(str(leftname))
        disparity = np.load(str(disparity_name))
        colored_name = disparity_name.with_suffix(".png")

        if args.gray:
            colored = as_gray(disparity)
        elif args.jet:
            colored = as_colorimage(disparity, vmax=vmax, vmin=vmin, colormap=cv2.COLORMAP_JET)
        elif args.inferno:
            colored = as_colorimage(disparity, vmax=vmax, vmin=vmin, colormap=cv2.COLORMAP_INFERNO)
        else:
            colored = as_colorimage(disparity, vmax=vmax, vmin=vmin, colormap=cv2.COLORMAP_JET)

        assert image.shape == colored.shape
        assert image.dtype == colored.dtype
        if args.save:
            cv2.imwrite(str(colored_name), colored)
        results = np.concatenate((image, colored), axis=1)
        results = resize_image(results, rate=0.5)
        cv2.imshow("left depth", results)
        cv2.waitKey(10)
        time.sleep(sec)


def view3d(args):
    captured_dir = Path(args.captured_dir)
    leftdir, _, disparity_dir = get_dirs(captured_dir)
    sec = args.sec

    left_images = sorted(leftdir.glob("**/*.png"))
    disparity_npys = sorted(disparity_dir.glob("**/*.npy"))

    json_file = list(captured_dir / "*.json")[0]
    camera_parameter = CameraParameter.load_json(json_file)

    width = camera_parameter.width
    height = camera_parameter.height
    fx = camera_parameter.fx
    fy = camera_parameter.fy
    cx = camera_parameter.cx
    cy = camera_parameter.cy

    left_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for leftname, disparity_name in tqdm(list(zip(left_images, disparity_npys))):
        print(leftname, disparity_name)
        plyname = disparity_name.with_suffix(".ply")
        disparity = np.load(str(disparity_name))
        baseline = camera_parameter.baseline
        focal_length = camera_parameter.fx
        depth = baseline * focal_length / disparity

        rgb = o3d.io.read_image(str(leftname))
        open3d_depth = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, open3d_depth)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, left_cam_intrinsic)
        if args.save:
            o3d.io.write_point_cloud(str(plyname), pcd)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(sec)

    vis.destroy_window()


def view_npy(disparity: np.ndarray, args, npy=None):
    vmin = args.vmin
    vmax = args.vmax
    if args.gray:
        colored = as_gray(disparity)
    elif args.jet:
        colored = as_colorimage(disparity, vmax=None, vmin=None, colormap=cv2.COLORMAP_JET)
    elif args.inferno:
        colored = as_colorimage(disparity, vmax=None, vmin=None, colormap=cv2.COLORMAP_INFERNO)
    else:
        colored = as_colorimage(disparity, vmax=None, vmin=None, colormap=cv2.COLORMAP_JET)

    if args.outdir:
        outname = Path(args.outdir) / f"colormap_{npy.stem}.png"
    else:
        outname = Path(".") / f"colormap_{npy.stem}.png"
    outname.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(outname), colored)
    print(f"saved as {outname}")
    cv2.imshow("img", colored)
    cv2.waitKey(-1)


def disparity_viewer_main():
    """
    A tool to view depth(as npy file) and left image.
    In --disp3d case, use open3d to show 3D point cloud.
    """
    import argparse

    parser = argparse.ArgumentParser(description="disparity npy file viewer")
    parser.add_argument("captured_dir", help="captured directory by capture.py")
    parser.add_argument("--sec", type=int, default=1, help="wait sec")
    parser.add_argument("--vmax", type=float, default=500, help="max disparity [pixel]")
    parser.add_argument("--vmin", type=float, default=0, help="min disparity [pixel]")
    parser.add_argument("--disp3d", action="store_true", help="display 3D")
    parser.add_argument("--save", action="store_true", help="save colored or ply")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", action="store_true", help="gray colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    group.add_argument("--inferno", action="store_true", help="inferno colormap")
    args = parser.parse_args()
    if args.disp3d:
        view3d(args)
    else:
        view_by_colormap(args)


def view_npy_main():
    parser = argparse.ArgumentParser(description="np file viewer")
    parser.add_argument("npy_file", help="npy_file to view")
    parser.add_argument("--vmax", type=float, default=500, help="max disparity [pixel]")
    parser.add_argument("--vmin", type=float, default=0, help="min disparity [pixel]")
    parser.add_argument("--disp3d", action="store_true", help="display 3D")
    parser.add_argument("--outdir", default="output", help="save colored or ply")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", action="store_true", help="gray colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    group.add_argument("--inferno", action="store_true", help="inferno colormap")
    group.add_argument("--normal", action="store_true", help="normal mapping")

    args = parser.parse_args()
    if not Path(args.npy_file).exists():
        print(f"no such file {args.npy_file}")
        exit()
    if Path(args.npy_file).is_file():
        npys = [Path(args.npy_file)]
    elif Path(args.npy_file).is_dir():
        npys = sorted(Path(args.npy_file).glob("*.npy"))
    baseline = 120.0  # [mm] baseline value for ZED2i case
    for npy in tqdm(npys):
        disparity = np.load(npy)

        minval = np.nanmin(disparity.flatten())
        if args.normal:
            converter = DepthToNormalMap()
            depth_map = baseline / disparity
            # padded to remove non-finite values
            depth_map[np.logical_not(np.isfinite(depth_map))] = baseline / minval
            normal_bgr = converter.convert(depth_map)
            oname = Path(args.outdir) / f"normal_{npy.stem}.png"
            oname.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(oname), normal_bgr)
            print(f"saved {oname}")
        else:
            view_npy(disparity, args, npy)
