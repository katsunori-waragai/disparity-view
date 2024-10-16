from pathlib import Path

import cv2

import disparity_view

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert depth map to normal map")
    parser.add_argument("input", type=str, help="Path to depth map gray image")
    parser.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Output directory for normal map image (default: output)",
    )
    args = parser.parse_args()

    converter = disparity_view.DepthToNormalMap()
    inputname = Path(args.input)
    depth_map = cv2.imread(str(inputname), cv2.IMREAD_GRAYSCALE)
    normal_bgr = converter.convert(depth_map)
    outname = Path(args.outdir) / f"normal_{inputname.stem}.png"
    outname.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(outname), normal_bgr)
    print(f"saved {outname}")
