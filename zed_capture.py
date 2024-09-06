"""
capture script using zed2i

requirement:
    ZED2i camera
    ZED SDK
"""

from pathlib import Path


import pyzed.sl as sl
import argparse

import cv2
import numpy as np

from util_depth_view import depth_as_colorimage

MAX_ABS_DEPTH, MIN_ABS_DEPTH = 0.0, 2.0  # [m]


def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def main(opt):
    outdir = Path(opt.outdir)
    leftdir = outdir / "left"
    rightdir = outdir / "right"
    zeddepthdir = outdir / "zed-depth"
    leftdir.mkdir(exist_ok=True, parents=True)
    rightdir.mkdir(exist_ok=True, parents=True)
    zeddepthdir.mkdir(exist_ok=True, parents=True)

    zed = sl.Camera()
    init_params = sl.InitParameters()

    parse_args(init_params)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD2K

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    left_image = sl.Mat()
    right_image = sl.Mat()
    depth = sl.Mat()
    depth_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_parameters.confidence_threshold = opt.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")

    title = f"Depth {init_params.depth_mode=}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    counter = 0
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
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
            print("done left_image.get_data()")
            cv_depth_img = depth_image.get_data()[:, :, 0]
            print(f"{cv_depth_img.shape=} {cv_depth_img.dtype=}")
            depth_data = depth.get_data()
            leftname = leftdir / f"left_{counter:05d}.png"
            rightname = rightdir / f"right_{counter:05d}.png"
            depthname = zeddepthdir / f"zeddepth_{counter:05d}.png"
            depthnpyname = zeddepthdir / f"zeddepth_{counter:05d}.npy"
            cv2.imwrite(str(leftname), cv_left_image)
            cv2.imwrite(str(rightname), cv_right_image)
            cv2.imwrite(str(depthname), cv2.applyColorMap(cv_depth_img, cv2.COLORMAP_JET))
            np.save(depthnpyname, depth_data)
            print(f"saved {leftname} {rightname}")
        else:
            continue
        assert cv_left_image.shape[2] == 3
        assert cv_left_image.dtype == np.uint8
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
        zed_depth = depth.get_data()  # np.ndarray 型
        print("done depth.get_data()")
        colored_depth_image = depth_as_colorimage(zed_depth)
        results = np.concatenate((cv_left_image, colored_depth_image), axis=1)

        cv2.imshow(title, results)
        key = cv2.waitKey(1)
        counter += 1
        if key == ord("q"):
            exit()

    if "zed" in locals():
        zed.close()


if __name__ == "__main__":
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
        help="Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default="",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="depth confidence_threshold(0 ~ 100)",
        default=100,
    )
    parser.add_argument(
        "--outdir",
        help="image pair output",
        default="outdir",
    )
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
