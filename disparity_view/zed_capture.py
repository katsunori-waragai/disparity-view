"""
capture script using zed2i (StereoLabs Camera)

requirement:
    ZED2i camera
        https://www.stereolabs.com/en-jp
    ZED SDK
        https://github.com/stereolabs/zed-sdk
"""

import pyzed.sl as sl

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from disparity_view.view import as_colorimage, get_dirs, resize_image
from disparity_view.cam_param import CameraParameter


def parse_args_to_params(args, init_params):
    if len(args.input_svo_file) > 0 and args.input_svo_file.endswith(".svo"):
        init_params.set_from_svo_file(args.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(args.input_svo_file))
    elif len(args.ip_address) > 0:
        ip_str = args.ip_address
        if _is_ip_address(ip_str):
            init_params.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif _is_numerical_ip_address(ip_str):
            init_params.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")
    if "HD2K" in args.resolution:
        init_params.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in args.resolution:
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in args.resolution:
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in args.resolution:
        init_params.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in args.resolution:
        init_params.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in args.resolution:
        init_params.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(args.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def _is_numerical_ip_address(ip_str: str) -> bool:
    return ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4


def _is_ip_address(ip_str: str) -> bool:
    return (
        ip_str.replace(":", "").replace(".", "").isdigit()
        and len(ip_str.split(".")) == 4
        and len(ip_str.split(":")) == 2
    )


def capture_main(args):
    outdir = Path(args.outdir)
    leftdir, rightdir, disparity_dir = get_dirs(outdir)

    zed = sl.Camera()
    init_params = sl.InitParameters()

    parse_args_to_params(args, init_params)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        sys.exit(1)

    left_image = sl.Mat()
    right_image = sl.Mat()
    depth = sl.Mat()
    depth_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_parameters.confidence_threshold = args.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")

    title = f"Depth {init_params.depth_mode=}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    cam_info = zed.get_camera_information()
    camera_parameter = CameraParameter.create(cam_info)
    json_name = outdir / "zed_camera_param.json"
    camera_parameter.save_json(json_name)

    counter = 0
    while True:
        if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT, sl.MEM.CPU)
        zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        cv_left_image = left_image.get_data()
        assert cv_left_image.shape[2] == 4  # ZED SDK dependent.
        cv_left_image = cv_left_image[:, :, :3].copy()
        cv_left_image = np.ascontiguousarray(cv_left_image)
        cv_right_image = right_image.get_data()
        assert cv_right_image.shape[2] == 4  # ZED SDK dependent.
        cv_right_image = cv_right_image[:, :, :3].copy()
        cv_right_image = np.ascontiguousarray(cv_right_image)
        cv_depth_img = depth_image.get_data()[:, :, 0]
        depth_data = depth.get_data()
        leftname = leftdir / f"left_{counter:05d}.png"
        rightname = rightdir / f"right_{counter:05d}.png"
        cv2.imwrite(str(leftname), cv_left_image)
        cv2.imwrite(str(rightname), cv_right_image)
        print(f"saved {leftname} {rightname}")

        assert cv_left_image.shape[2] == 3
        assert cv_left_image.dtype == np.uint8
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed_depth = depth.get_data()
        baseline = camera_parameter.baseline
        focal_length = camera_parameter.fx
        disparity = baseline * focal_length / zed_depth
        disparity_npyname = disparity_dir / f"zed_disparity_{counter:05d}.npy"
        np.save(disparity_npyname, disparity)
        colored_depth_image = as_colorimage(zed_depth)
        results = np.concatenate((cv_left_image, colored_depth_image), axis=1)
        results = resize_image(results, rate=0.5)
        cv2.imshow(title, results)
        key = cv2.waitKey(1)
        counter += 1
        if key == ord("q"):
            sys.exit()

    if "zed" in locals():
        zed.close()


def main():
    parser = argparse.ArgumentParser(description="capture stereo pairs")
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default="",
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup",
        default="",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution, can be either HD2K, HD1080, HD720, SVGA or VGA",
        default="HD1080",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="depth confidence_threshold(0 ~ 100)",
        default=80,
    )
    parser.add_argument(
        "--outdir",
        help="image pair output",
        default="outdir",
    )

    args = parser.parse_args()
    if len(args.input_svo_file) > 0 and len(args.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        sys.exit()
    capture_main(args)
