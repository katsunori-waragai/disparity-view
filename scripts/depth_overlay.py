from pathlib import Path

import cv2
import numpy as np

import disparity_view

if __name__ == "__main__":
    """
    python3 depth_overlay.py --jet ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="overlay depth image to left image")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--outdir", default="output", help="output folder")
    group = parser.add_argument_group("colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    group.add_argument("--inferno", action="store_true", help="inferno colormap")

    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    outdir = Path(args.outdir)

    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)

    colormap = cv2.COLORMAP_INFERNO if args.inferno else cv2.COLORMAP_JET
    color_depth = disparity_view.as_colorimage(disparity, vmax=None, vmin=None, colormap=colormap)

    overlayed = disparity_view.depth_overlay(gray, color_depth)
    outname = outdir / f"overlay_{left_name.stem}.png"
    outname.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(outname), overlayed)
