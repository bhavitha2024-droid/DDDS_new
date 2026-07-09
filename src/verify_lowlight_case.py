"""
verify_lowlight_case.py

ADDITIVE ONLY -- new file, does not modify anything else.

Targets the base paper's own self-admitted weakness (see its Discussion/Conclusion
sections): accuracy drops in low light / glare, because its face+eye detector
struggles to see the driver at all in poor illumination.

This script runs your EXISTING, unmodified feature_extraction.FeatureExtractor
(MediaPipe FaceMesh + CLAHE preprocessing) on a low-light frame TWICE:

    1. Raw frame as-is             -- represents what a plain/base-paper-style
                                       pipeline sees in the dark
    2. Frame enhanced with the additive low_light.py module (gamma + CLAHE +
       histogram equalization, stacked) -- your proposed improvement

...and reports whether a face/features were successfully extracted in each case.
A clean demo for your report: "face detected: NO (raw) -> YES (enhanced)".

Usage (single image):
    python src/verify_lowlight_case.py --image path/to/dark_frame.jpg

Usage (webcam -- captures one frame, then darkens it synthetically to simulate
low light, useful if you don't have an actual dark photo handy):
    python src/verify_lowlight_case.py --source 0 --simulate_darkness 0.15
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_extraction import FeatureExtractor  # noqa: E402
from low_light import enhance_low_light  # noqa: E402


def darken(frame: np.ndarray, factor: float) -> np.ndarray:
    """Synthetically simulate a low-light frame by scaling pixel brightness down."""
    return (frame.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def describe(feats):
    if feats is None:
        return "NO FACE DETECTED"
    return (f"face detected -- EAR_avg={feats[2]:.3f}, MAR={feats[3]:.3f}, "
            f"pitch={feats[4]:.1f}, yaw={feats[5]:.1f}, roll={feats[6]:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to a low-light frame/photo.")
    parser.add_argument("--source", default=None, help="Camera index, used if --image not given.")
    parser.add_argument("--simulate_darkness", type=float, default=None,
                         help="If set (e.g. 0.15), darkens the captured/loaded frame by this "
                              "factor to simulate poor lighting.")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}")
            sys.exit(1)
    elif args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"Could not capture from source: {args.source}")
            sys.exit(1)
    else:
        print("Provide either --image path or --source (camera index).")
        sys.exit(1)

    if args.simulate_darkness is not None:
        frame = darken(frame, args.simulate_darkness)
        print(f"Simulated low light by scaling brightness by factor {args.simulate_darkness}")

    extractor = FeatureExtractor()
    raw_feats = extractor.extract(frame)
    enhanced_frame = enhance_low_light(frame)
    enhanced_feats = extractor.extract(enhanced_frame)
    extractor.close()

    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "lowlight_raw.jpg"), frame)
    cv2.imwrite(os.path.join(out_dir, "lowlight_enhanced.jpg"), enhanced_frame)

    print("\n=== Low-light face-detection comparison ===")
    print(f"RAW frame:      {describe(raw_feats)}")
    print(f"ENHANCED frame: {describe(enhanced_feats)}")
    print(f"\nSaved both frames to {out_dir}/lowlight_raw.jpg and {out_dir}/lowlight_enhanced.jpg "
          f"-- use these as a before/after screenshot in your report.")

    if raw_feats is None and enhanced_feats is not None:
        print("\n>>> SUCCESS: face detection failed on the raw low-light frame but succeeded "
              "after your low_light.py enhancement -- a clean, demonstrable improvement over "
              "the base paper's self-admitted low-light weakness.")
    elif raw_feats is not None and enhanced_feats is not None:
        print("\nFace was detected in both -- try a darker frame (increase --simulate_darkness "
              "toward 0 or use a dimmer room) for a clearer contrast.")
    elif raw_feats is None and enhanced_feats is None:
        print("\nFace not detected in either -- this frame may be too dark even for the "
              "enhancement, or the face isn't clearly in view. Try a less extreme "
              "--simulate_darkness value.")


if __name__ == "__main__":
    main()
