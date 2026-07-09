"""
calibration.py

PERSONALIZATION MODULE.

Note on scope: your original abstract text (as you provided it to me) does not use the
word "personalization" -- it specifies eye closure / yawning / head movement, LSTM
temporal modeling, and an intelligent alert mechanism. I'm adding this as a genuine,
defensible EXTENSION on top of that, not pretending it was already written in your
abstract. If your actual submitted abstract/synopsis document (the one your guide
signed) does mention personalization, tell me and I'll align the wording; otherwise,
frame this in your report as an added enhancement, which is a perfectly normal thing
for an M.Tech project to include.

Why personalization is a real gap in the base paper: it uses ONE fixed global rule
(alarm at exactly N consecutive closed-eye frames) applied identically to every driver.
But resting eye-openness varies a lot person to person (eye shape, eyelid hooding,
glasses, camera angle) -- a threshold tuned for one face is routinely wrong for another.
This module removes that dependency on a single global constant by calibrating each
driver's own "alert baseline" for EAR/MAR/head-pose before detection begins, and
expressing all downstream features RELATIVE TO that driver's own baseline.

Workflow:
    1. `calibrate_driver.py` (or the CalibrationManager class below) records a short
       (~10-15s) baseline clip of the driver looking naturally alert at the camera.
    2. Per-driver mean/std of [EAR_left, EAR_right, EAR_avg, MAR, pitch, yaw, roll]
       is computed and stored in `models/driver_profiles/<driver_id>.json`.
    3. At both training-window construction time (optional) and real-time inference
       time, raw features can be converted to PERSONALIZED features:
           personalized_EAR = raw_EAR / driver_baseline_EAR_mean
       i.e. "what fraction of this driver's own normal eye-openness are they showing
       right now" -- which is comparable across drivers even though absolute EAR is not.
    4. The rule-based baseline's threshold is also personalized: instead of a single
       global `ear_threshold`, it becomes `driver_baseline_EAR_mean * relative_factor`.

This directly targets the base paper's implicit assumption of a single, universal
threshold, and is something the base paper does not do at all.
"""

import json
import os
import time

import cv2
import numpy as np

FEATURE_NAMES = ["EAR_left", "EAR_right", "EAR_avg", "MAR", "pitch", "yaw", "roll"]
EAR_AVG_IDX = 2
MAR_IDX = 3


def personalize_vector(raw_feats: np.ndarray, baseline_mean: np.ndarray) -> np.ndarray:
    """
    Shared personalization transform used BOTH when building personalized training
    data (dataset.py, subject-relative normalization) AND at real-time inference
    (DriverProfile.personalize below), so the LSTM is trained on exactly the same
    kind of feature it will see at deployment.

    EAR/MAR (indices 0-3): expressed as a ratio to the baseline ("what fraction of
    this person's normal eye/mouth openness is showing right now").
    Head pose (indices 4-6): expressed as a delta from the baseline resting pose.
    """
    out = raw_feats.copy()
    for idx in range(4):
        b = baseline_mean[idx] if baseline_mean[idx] != 0 else 1e-6
        out[idx] = raw_feats[idx] / b
    for idx in range(4, 7):
        out[idx] = raw_feats[idx] - baseline_mean[idx]
    return out


class DriverProfile:
    """Holds one driver's calibration baseline (mean/std per feature channel)."""

    def __init__(self, driver_id: str, mean: np.ndarray, std: np.ndarray, num_samples: int):
        self.driver_id = driver_id
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.num_samples = num_samples

    def to_dict(self):
        return {
            "driver_id": self.driver_id,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "num_samples": self.num_samples,
            "feature_order": FEATURE_NAMES,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["driver_id"], np.array(d["mean"]), np.array(d["std"]), d["num_samples"])

    def save(self, profile_dir="models/driver_profiles"):
        os.makedirs(profile_dir, exist_ok=True)
        path = os.path.join(profile_dir, f"{self.driver_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, driver_id: str, profile_dir="models/driver_profiles"):
        path = os.path.join(profile_dir, f"{driver_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No calibration profile found for driver '{driver_id}' at {path}. "
                f"Run calibrate_driver.py first."
            )
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def personalize(self, raw_feats: np.ndarray) -> np.ndarray:
        """
        Convert raw [EAR_left, EAR_right, EAR_avg, MAR, pitch, yaw, roll] into
        driver-relative features, using the SAME transform (personalize_vector) that
        was applied to subject data at training time, so the trained LSTM sees a
        consistent feature distribution at inference time.
        """
        return personalize_vector(raw_feats, self.mean)

    def personalized_ear_threshold(self, relative_factor: float = 0.75) -> float:
        """
        Personalized replacement for the base paper's single global EAR threshold:
        instead of one constant for everyone, use a fraction of THIS driver's own
        baseline eye-openness.
        """
        return self.mean[EAR_AVG_IDX] * relative_factor


class CalibrationManager:
    """Records a short baseline clip for a driver and computes their DriverProfile."""

    def __init__(self, feature_extractor):
        """feature_extractor: an instance of feature_extraction.FeatureExtractor"""
        self.extractor = feature_extractor

    def calibrate_from_camera(self, driver_id: str, duration: int = 12, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        print(f"Calibrating driver '{driver_id}'. Please look naturally at the camera, "
              f"eyes open, relaxed, for {duration} seconds.")
        for i in (3, 2, 1):
            print(i)
            time.sleep(1)

        samples = []
        start = time.time()
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            feats = self.extractor.extract(frame)
            if feats is not None:
                samples.append(feats)

            remaining = duration - (time.time() - start)
            disp = frame.copy()
            cv2.putText(disp, f"CALIBRATING {remaining:0.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow("Calibration - press q to stop early", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(samples) < 10:
            raise RuntimeError(
                f"Only captured {len(samples)} valid frames; need at least 10. "
                f"Ensure your face is clearly visible and try again."
            )

        samples = np.stack(samples, axis=0)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0) + 1e-6
        profile = DriverProfile(driver_id, mean, std, num_samples=len(samples))
        return profile


def personalize_feature_matrix(X_raw: np.ndarray, profile: DriverProfile) -> np.ndarray:
    """Vectorized personalization for a (num_frames_or_windows, ..., 7) raw feature array."""
    orig_shape = X_raw.shape
    flat = X_raw.reshape(-1, orig_shape[-1])
    personalized = np.stack([profile.personalize(row) for row in flat], axis=0)
    return personalized.reshape(orig_shape)
