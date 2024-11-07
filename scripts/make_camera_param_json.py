from pathlib import Path

from disparity_view.zed_info import CameraParameter

if __name__ == "__main__":
    """
    カメラパラメータのjsonファイルを作成するためのスクリプト
    """
    import argparse

    parser = argparse.ArgumentParser("create camera param json file")
    parser.add_argument("--width", type=int, required=True, help="width")
    parser.add_argument("--height", type=int, required=True, help="height")
    parser.add_argument("--focal_length", type=float, required=True, help="focal_length[pixel]")
    parser.add_argument("--baseline", type=float, required=True, help="baseline[mm]")
    parser.add_argument("--output", type=str, default="output", help="output dir")
    args = parser.parse_args()
    print(args)
    width = args.width
    height = args.height
    focal_length = args.focal_length
    baseline = args.baseline
    camera_parameter = CameraParameter(width=width, height=height, fx=focal_length, fy=focal_length, baseline=baseline)
    json = Path(args.output) / f"dummy_{width}_{height}.json"
    camera_parameter.save_json(json)
