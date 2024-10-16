import cv2

import disparity_view

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert depth map to normal map")
    parser.add_argument("input", type=str, help="Path to depth map gray image")
    parser.add_argument(
        "--outdir",
        type=str,
        default="normal_map.png",
        help="Output path for normal map image (default: normal_map.png)",
    )
    args = parser.parse_args()

    converter = disparity_view.DepthToNormalMap()
    depth_map = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    normal_bgr = converter.convert(depth_map)
    cv2.imwrite(args.outdir, normal_bgr)
    print(f"saved {args.outdir}")
