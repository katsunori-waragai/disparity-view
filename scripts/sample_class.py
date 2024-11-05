from pathlib import Path

import numpy as np
import skimage.io
from tqdm import tqdm

from disparity_view.o3d_project import gen_tvec, DEPTH_SCALE
from disparity_view.o3d_project import as_extrinsics
from disparity_view.projection_class import StereoCamera
from disparity_view.util import safer_imsave
from disparity_view.animation_gif import AnimationGif


def gen_right_image(disparity, left_image, outdir, left_name, axis=0):
    stereo_camera = StereoCamera(baseline=120)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    scaled_baseline = stereo_camera.scaled_baseline()  # [mm]
    tvec = gen_tvec(scaled_shift=scaled_baseline, axis=axis)
    extrinsics = as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = outdir / f"color_{left_name.stem}.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    depth_legacy = np.asarray(projected.depth.to_legacy())
    depth_file = outdir / f"depth_{left_name.stem}.png"
    depth_file.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(depth_file), depth_legacy)
    print(f"saved {outfile}")
    print(f"saved {depth_file}")

    assert outfile.lstat().st_size > 0

def make_animation_gif(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path, axis=0):
    """
    save animation gif file

    Args:
        disparity: disparity image
        left_image:left camera image
        outdir: destination directory
        left_name: file name of the left camera image
    Returns：
        None
    """
    assert axis in (0, 1, 2)

    stereo_camera = StereoCamera(baseline=120)
    stereo_camera.set_camera_matrix(shape=disparity.shape, focal_length=1070)
    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left_image)
    scaled_baseline = stereo_camera.scaled_baseline()  # [mm]
    tvec = gen_tvec(scaled_shift=scaled_baseline, axis=axis)
    extrinsics = as_extrinsics(tvec)
    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
    color_legacy = np.asarray(projected.color.to_legacy())
    outfile = outdir / f"color_{left_name.stem}.png"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(outfile), color_legacy)
    depth_legacy = np.asarray(projected.depth.to_legacy())
    depth_file = outdir / f"depth_{left_name.stem}.png"
    depth_file.parent.mkdir(exist_ok=True, parents=True)
    safer_imsave(str(depth_file), depth_legacy)
    print(f"saved {outfile}")
    print(f"saved {depth_file}")


    maker = AnimationGif()
    n = 16
    for i in tqdm(range(n + 1)):
        scaled_baseline = stereo_camera.scaled_baseline()
        tvec = gen_tvec(scaled_baseline * i / n, axis)
        extrinsics = as_extrinsics(tvec)
        projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
        color_img = np.asarray(projected.color.to_legacy())
        color_img = (color_img * 255).astype(np.uint8)
        maker.append(color_img)

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    maker.save(gifname)


if __name__ == "__main__":
    """
    python3 project.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="reprojector")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--axis", default=0, help="axis to shift(0: to right, 1: to upper, 2: to far)")
    parser.add_argument("--gif", action="store_true", help="git animation")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()

    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    axis = int(args.axis)
    outdir = Path(args.outdir)

    left_image = skimage.io.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    if args.gif:
        make_animation_gif(disparity, left_image, outdir, left_name, axis=axis)
    else:
        gen_right_image(disparity, left_image, outdir, left_name, axis=axis)
