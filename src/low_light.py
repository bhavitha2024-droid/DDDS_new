"""
low_light.py

Standalone low-light frame enhancement module.

Ported from the second reviewed project (drowsiness-detection / "driveai") as an
ADDITIVE option for this repository. This file does not modify or replace anything
in `preprocessing.py` -- it is a separate, opt-in module you can wire into
`feature_extraction.py` or `realtime_infer.py` yourself if you want a stronger
low-light pipeline (gamma correction + CLAHE + histogram equalization, stacked)
instead of (or in addition to) the CLAHE-only step already used elsewhere in
this project.

Usage (manual, optional -- nothing in this repo calls this automatically):

    from low_light import enhance_low_light
    enhanced_frame = enhance_low_light(frame)

or use the individual steps directly:

    from low_light import apply_gamma, apply_clahe, histogram_equalization
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_gamma(frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Brighten (gamma > 1) or darken (gamma < 1) a frame via a gamma lookup table."""
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(frame, table)


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel (LAB space)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def histogram_equalization(frame: np.ndarray) -> np.ndarray:
    """Global histogram equalization on the luma channel (YCrCb space)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    equalized = cv2.equalizeHist(y_channel)
    merged = cv2.merge((equalized, cr_channel, cb_channel))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """Full stacked pipeline: gamma correction -> CLAHE -> histogram equalization.

    This is the more aggressive 3-stage enhancement from the second project. It is
    provided here as an alternative to this repo's existing single-stage CLAHE step
    in `preprocessing.py`, for cases with very poor ambient lighting.
    """
    enhanced = apply_gamma(frame)
    enhanced = apply_clahe(enhanced)
    enhanced = histogram_equalization(enhanced)
    return enhanced
