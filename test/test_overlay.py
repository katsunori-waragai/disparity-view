import cv2

from disparity_view.view import depth_overlay

def test_overlay():
    grayname = "../test/test-imgs/left/left_motorcycle.png"
    color_depth_name = "../test/test-imgs/disparity-IGEV/left_motorcycle.png"
    gray = cv2.imread(grayname, cv2.IMREAD_GRAYSCALE)
    color_depth = cv2.imread(color_depth_name)
    overlayed = depth_overlay(gray, color_depth)
    assert len(overlayed.shape) == 3
    assert overlayed.shape[2] == 3
    assert overlayed.shape[:2] == color_depth.shape[:2]
    cv2.imwrite("overlayed.png", overlayed)
