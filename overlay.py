import cv2
import numpy as np

def depth_overlay(gray: np.ndarray, color_depth: np.ndarray) -> np.ndarray:
    """
    overlay color_depth to gray scale image
    """
    assert len(gray.shape) == 2
    assert len(color_depth.shape) == 3
    gray2 = cv2.merge((gray, gray, gray)).astype(np.uint16)
    color_depth2 = color_depth.astype(np.uint16)
    return ((gray2 + color_depth2) / 2).astype(np.uint8)


if __name__ == "__main__":
    grayname = "test/test-imgs/left/left_motorcycle.png"
    color_depth_name = "test/test-imgs/disparity-IGEV/left_motorcycle.png"
    gray = cv2.imread(grayname, cv2.IMREAD_GRAYSCALE)
    color_depth = cv2.imread(color_depth_name)
    overlayed = depth_overlay(gray, color_depth)
    cv2.imwrite("overlayed.png", overlayed)
