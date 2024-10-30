"""
簡易的な実装である。
そのため、Open3dのライブラリで今後置き換える可能性あり。
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DepthToNormalMap:
    """A class for converting a depth map image to a normal map image."""

    def convert(self, depth_map: np.ndarray) -> np.ndarray:
        """Converts the depth map image to a normal map image."""
        rows, cols = depth_map.shape[:2]

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)

        # Compute the normal vector for each pixel
        z_value = 0.1 * np.percentile(np.sqrt(dx**2 + dy**2), 50)  # median
        normal = np.dstack((-dx, -dy, np.full((rows, cols), z_value)))
        # sqrt(dx**2 + dy**2 + z_value**2) を計算している
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))

        # それぞれの画素についてRGBのベクトルの大きさが１になるようにしている。
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)

        return cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
