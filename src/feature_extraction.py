"""
feature_extraction.py

Extracts the per-frame behavioral feature vector used as LSTM input:
    [EAR_left, EAR_right, EAR_avg, MAR, pitch, yaw, roll]

Unlike the base paper's approach (MobileNet-SSD classifying an image crop into
{eye-open, eye-close, yawn, no-yawn}), this module extracts continuous, interpretable
GEOMETRIC signals from 468 MediaPipe FaceMesh landmarks:

  - EAR (Eye Aspect Ratio): standard Soukupova & Cech formulation.
  - MAR (Mouth Aspect Ratio): analogous ratio for yawning.
  - Head pose (pitch/yaw/roll): solvePnP against a generic 3D face model,
    which the base paper does not compute at all (its "head movement" cue is
    listed as a future improvement, not implemented).

These continuous values let the downstream LSTM learn genuine temporal *trends*
(e.g. EAR slowly decreasing over 1-2 seconds) rather than relying on the base
paper's fixed "10 consecutive closed frames" rule.

Can be run standalone to batch-extract features from a directory of labeled videos:

    python feature_extraction.py --video_dir data/raw --out data/processed/features.csv

Expected video naming convention (used to assign weak labels for supervised training):
    <anything>_alert_*.mp4        -> label 0
    <anything>_drowsy_*.mp4       -> label 1
    <anything>_highdrowsy_*.mp4   -> label 2
Adjust `label_from_filename` if your dataset (e.g. NTHU-DDD, YawDD) uses a different scheme.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError:
    mp = None

sys.path.append(os.path.dirname(__file__))
from preprocessing import FramePreprocessor  # noqa: E402


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# Generic 3D model points (mm) for solvePnP head-pose estimation, indexed to
# a small stable subset of FaceMesh landmarks.
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip        (landmark 1)
    (0.0, -63.6, -12.5),      # Chin            (landmark 152)
    (-43.3, 32.7, -26.0),     # Left eye corner (landmark 33)
    (43.3, 32.7, -26.0),      # Right eye corner(landmark 263)
    (-28.9, -28.9, -24.1),    # Left mouth      (landmark 61)
    (28.9, -28.9, -24.1),     # Right mouth     (landmark 291)
], dtype=np.float64)
POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, idxs):
    p = [landmarks[i] for i in idxs]
    vertical1 = _euclidean(p[1], p[5])
    vertical2 = _euclidean(p[2], p[4])
    horizontal = _euclidean(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, idxs):
    p = [landmarks[i] for i in idxs]
    vertical1 = _euclidean(p[2], p[3])
    vertical2 = _euclidean(p[4], p[5])
    horizontal = _euclidean(p[0], p[1])
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def estimate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array(
        [(landmarks[i][0] * w, landmarks[i][1] * h) for i in POSE_LANDMARK_IDS],
        dtype=np.float64,
    )
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(a) for a in euler_angles.flatten()]
    return pitch, yaw, roll


class FeatureExtractor:
    """Wraps MediaPipe FaceMesh to turn a BGR frame into the 7-d behavioral feature vector."""

    def __init__(self, clahe_clip=2.0, clahe_tile=(8, 8)):
        if mp is None:
            raise ImportError("mediapipe is required: pip install mediapipe")
        self.preprocessor = FramePreprocessor(clip_limit=clahe_clip, tile_grid=clahe_tile)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray):
        """Returns a 7-d np.array or None if no face was detected."""
        frame_bgr = self.preprocessor.process(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        points = [(p.x, p.y) for p in lm]

        ear_l = eye_aspect_ratio(points, LEFT_EYE)
        ear_r = eye_aspect_ratio(points, RIGHT_EYE)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = mouth_aspect_ratio(points, MOUTH)
        pitch, yaw, roll = estimate_head_pose(points, frame_bgr.shape)

        return np.array([ear_l, ear_r, ear_avg, mar, pitch, yaw, roll], dtype=np.float32)

    def close(self):
        self.face_mesh.close()


def label_from_filename(path: str) -> int:
    name = os.path.basename(path).lower()
    if "highdrowsy" in name or "high_drowsy" in name:
        return 2
    if "drowsy" in name:
        return 1
    return 0


def subject_from_filename(path: str) -> str:
    """Assumes record_session.py naming convention: <subject>_<label>_<session>.ext"""
    return os.path.basename(path).split("_")[0]


def extract_video(video_path: str, extractor: FeatureExtractor, label: int):
    cap = cv2.VideoCapture(video_path)
    rows = []
    frame_idx = 0
    subject = subject_from_filename(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feats = extractor.extract(frame)
        if feats is not None:
            row = {
                "video": os.path.basename(video_path),
                "subject": subject,
                "frame_idx": frame_idx,
                "label": label,
            }
            for i, name in enumerate(
                ["EAR_left", "EAR_right", "EAR_avg", "MAR", "pitch", "yaw", "roll"]
            ):
                row[name] = feats[i]
            rows.append(row)
        frame_idx += 1
    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Batch-extract behavioral features from videos.")
    parser.add_argument("--video_dir", required=True, help="Directory of input videos.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--ext", default="mp4", help="Video extension to search for.")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.video_dir, "**", f"*.{args.ext}"), recursive=True))
    if not videos:
        print(f"No videos found in {args.video_dir} with extension .{args.ext}")
        sys.exit(1)

    extractor = FeatureExtractor()
    all_rows = []
    for vid in videos:
        label = label_from_filename(vid)
        print(f"Processing {vid} (label={label}) ...")
        rows = extract_video(vid, extractor, label)
        all_rows.extend(rows)
    extractor.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
