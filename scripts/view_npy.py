import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from disparity_view.view import view_npy

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
    if Path(args.npy_file).is_file():
        disparity = np.load(args.npy_file)
        view_npy(disparity, args)
    elif Path(args.npy_file).is_dir():
        npys = sorted(Path(args.npy_file).glob("*.npy"))
        for npy in tqdm(npys):
            disparity = np.load(npy)
            view_npy(disparity, args)
    else:
        print(f"no such file {args.npy_file}")
