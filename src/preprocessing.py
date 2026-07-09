"""
preprocessing.py

Frame-level preprocessing used before landmark/feature extraction.

The base paper (Dipu et al., 2021) explicitly reports that its CNN/SSD classifier's
"performance ... was not fully satisfied" in low-light conditions or when a flashlight
is pointed at the camera, because it classifies on raw pixel appearance.

Here we apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the luminance
channel before running face-mesh landmark detection. Because our downstream features are
GEOMETRIC RATIOS (EAR/MAR) and ANGLES (head pose), not raw-pixel classifications, they are
far less sensitive to absolute illumination as long as the landmarks can be located --
and CLAHE substantially improves landmark localization in dim / unevenly lit frames.
"""

import cv2
import numpy as np


class FramePreprocessor:
    def __init__(self, clip_limit: float = 2.0, tile_grid=(8, 8), target_size=(480, 480)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
        self.target_size = target_size

    def normalize_illumination(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of LAB color space to boost contrast in dark regions."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def resize(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_AREA)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_bgr = self.resize(frame_bgr)
        frame_bgr = self.normalize_illumination(frame_bgr)
        return frame_bgr
