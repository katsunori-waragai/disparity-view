import cv2

import disparity_view

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert depth map to normal map")
    parser.add_argument("--input", type=str, help="Path to depth map image")
    parser.add_argument("--max_depth", type=int, default=255, help="Maximum depth value (default: 255)")
    parser.add_argument(
        "--output_path",
        type=str,
        default="normal_map.png",
        help="Output path for normal map image (default: normal_map.png)",
    )
    args = parser.parse_args()

    converter = disparity_view.DepthToNormalMap(max_depth=args.max_depth)
    depth_map = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    normal_bgr = converter.convert(depth_map)
    cv2.imwrite(args.output_path, normal_bgr)
