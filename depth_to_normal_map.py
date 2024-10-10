import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DepthToNormalMap:
    """A class for converting a depth map image to a normal map image.


    Attributes:
        depth_map (ndarray): A numpy array representing the depth map image.
        max_depth (int): The maximum depth value in the depth map image.
    """

    depth_map: np.ndarray = field(default=None)
    max_depth: int = 255


    def convert(self, depth_map: np.ndarray) -> np.ndarray:
        """Converts the depth map image to a normal map image.

        Args:
            output_path (str): The path to save the normal map image file.

        """
        self.depth_map = depth_map
        rows, cols = self.depth_map.shape[:2]

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(self.depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(self.depth_map, cv2.CV_32F, 0, 1)

        # Compute the normal vector for each pixel
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)

        # Save the normal map to a file
        return cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)


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

    converter = DepthToNormalMap(max_depth=args.max_depth)
    depth_map = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    normal_bgr = converter.convert(depth_map)
    cv2.imwrite(args.output_path, normal_bgr)