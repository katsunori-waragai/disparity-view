import inspect
import pyzed.sl as sl


def get_fx_fy_cx_cy(left_cam_params):
    """
    Note:
        left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    """
    return left_cam_params.fx, left_cam_params.fy, left_cam_params.cx, left_cam_params.cy


def get_baseline(cam_info) -> float:
    """
    Note:
        cam_info = zed.get_camera_information()
    """
    return cam_info.camera_configuration.calibration_parameters.get_camera_baseline()


if __name__ == "__main__":
    # Create a ZED camera object
    zed = sl.Camera()

    # Set up initial parameters for the camera
    init_params = sl.InitParameters()

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        exit(1)

    # Retrieve camera information
    cam_info = zed.get_camera_information()

    for k, v in inspect.getmembers(cam_info):
        print(k, v)

    print(f"{cam_info.camera_configuration=}")
    print(f"{cam_info.sensors_configuration=}")

    for k, v in inspect.getmembers(cam_info.camera_configuration):
        print(k, v)

    for k, v in inspect.getmembers(cam_info.sensors_configuration):
        print(k, v)

    for k, v in inspect.getmembers(cam_info.camera_configuration.calibration_parameters):
        print(k, v)

    # Access left and right camera parameters
    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    right_cam_params = cam_info.camera_configuration.calibration_parameters.right_cam

    for k, v in inspect.getmembers(left_cam_params):
        print(k, v)

    for k, v in inspect.getmembers(right_cam_params):
        print(k, v)

    # Print some of the camera parameters
    print("Left Camera Parameters:")
    print(f"image_size: {left_cam_params.image_size.width} x {left_cam_params.image_size.height}")
    print(f"Focal Length (fx, fy): {left_cam_params.fx}, {left_cam_params.fy}")
    print(f"Principal Point (cx, cy): {left_cam_params.cx}, {left_cam_params.cy}")
    print(f"Distortion Coefficients: {left_cam_params.disto}")

    print("\nRight Camera Parameters:")
    print(f"image_size: {right_cam_params.image_size.width} x {right_cam_params.image_size.height}")
    print(f"Focal Length (fx, fy): {right_cam_params.fx}, {right_cam_params.fy}")
    print(f"Principal Point (cx, cy): {right_cam_params.cx}, {right_cam_params.cy}")
    print(f"Distortion Coefficients: {right_cam_params.disto}")
    print("\n")
    print(f"{cam_info.camera_configuration.calibration_parameters.get_camera_baseline()=}")

    print(f"{get_fx_fy_cx_cy(left_cam_params)=}")
    print(f"{get_baseline(cam_info)}")
    # Close the camera
    zed.close()
