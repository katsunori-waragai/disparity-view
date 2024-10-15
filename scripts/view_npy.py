import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from disparity_view.view import view_npy
import disparity_view

if __name__ == "__main__":
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
    group.add_argument("--normal", action="store_true", help="normal colormap")

    args = parser.parse_args()
    print(args)
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
            converter = disparity_view.DepthToNormalMap()
            depth_map = baseline / disparity
            # padded to remove non-finite values
            depth_map[np.logical_not(np.isfinite(depth_map))] = baseline / minval
            normal_bgr = converter.convert(depth_map)
            oname = Path(args.outdir) / f"normal_{npy.stem}.png"
            oname.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(oname), normal_bgr)
            print(f"saved {oname}")
        else:
            view_npy(disparity, args)
