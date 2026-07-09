"""
calibrate_driver.py

CLI entry point for the personalization step: run this once per driver before their
first drive (or once per session, if you want to be robust to glasses on/off, etc.)
to build their DriverProfile baseline.

Usage:
    python src/calibrate_driver.py --driver_id bhavitha --duration 12
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from feature_extraction import FeatureExtractor  # noqa: E402
from calibration import CalibrationManager  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Calibrate a per-driver baseline profile.")
    parser.add_argument("--driver_id", required=True, help="Unique identifier, e.g. 'bhavitha'.")
    parser.add_argument("--duration", type=int, default=12, help="Seconds of baseline recording.")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--profile_dir", default="models/driver_profiles")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    manager = CalibrationManager(extractor)
    profile = manager.calibrate_from_camera(
        args.driver_id, duration=args.duration, camera_index=args.camera_index
    )
    path = profile.save(args.profile_dir)
    extractor.close()

    print(f"Saved calibration profile for '{args.driver_id}' to {path}")
    print(f"Baseline EAR_avg={profile.mean[2]:.3f}  MAR={profile.mean[3]:.3f}  "
          f"pitch={profile.mean[4]:.1f} yaw={profile.mean[5]:.1f} roll={profile.mean[6]:.1f}")


if __name__ == "__main__":
    main()
