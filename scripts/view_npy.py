import argparse

import cv2
import numpy as np

from disparity_view.view import as_colorimage, as_gray

def view_npy(disparity, args):
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

    outname = "tmp.png"
    cv2.imwrite(outname, colored)
    print(f"saved as {outname}")
    cv2.imshow("img", colored)
    cv2.waitKey(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="np file viewer")
    parser.add_argument("npy_file", help="npy_file to view")
    parser.add_argument("--vmax", type=float, default=500, help="max disparity [pixel]")
    parser.add_argument("--vmin", type=float, default=0, help="min disparity [pixel]")
    parser.add_argument("--disp3d", action="store_true", help="display 3D")
    parser.add_argument("--save", action="store_true", help="save colored or ply")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", action="store_true", help="gray colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    group.add_argument("--inferno", action="store_true", help="inferno colormap")

    args = parser.parse_args()
    print(args)
    disparity = np.load(args.npy_file)
    view_npy(disparity, args)
