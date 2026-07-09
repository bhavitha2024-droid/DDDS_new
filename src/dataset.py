"""
dataset.py

Turns the per-frame feature CSV produced by feature_extraction.py into sliding-window
sequences suitable for LSTM training. This is the key structural difference from the
base paper: instead of one classification decision per single frame, each training
example here is a WINDOW of `sequence_length` consecutive frames, and the label is the
drowsiness state that window represents.

CSV is expected to have columns:
    video, frame_idx, label, EAR_left, EAR_right, EAR_avg, MAR, pitch, yaw, roll
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
import os
sys.path.append(os.path.dirname(__file__))
from calibration import personalize_vector  # noqa: E402

FEATURE_COLUMNS = ["EAR_left", "EAR_right", "EAR_avg", "MAR", "pitch", "yaw", "roll"]


def subject_from_video(video_name: str) -> str:
    """Assumes the record_session.py naming convention: <subject>_<label>_<session>.ext"""
    return video_name.split("_")[0]


def personalize_dataframe(df: pd.DataFrame, feature_columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """
    PERSONALIZATION (training-time half). For each subject in the dataset, compute
    their own baseline (mean of their "Alert"/label==0 frames -- falling back to all
    of that subject's frames if no Alert-labeled clip exists) and express all of that
    subject's frames RELATIVE to their own baseline using the same transform
    (personalize_vector) that calibrate_driver.py / DriverProfile use at real-time
    inference. This is what lets a single shared LSTM generalize across different
    people's resting eye/mouth openness and head orientation, instead of learning
    absolute EAR/MAR/pose values that are only valid for whoever was recorded.

    Without this step, the base paper's own weakness re-appears in a learned form:
    a single set of parameters implicitly tuned to whichever face(s) dominate the
    training data.
    """
    df = df.copy()
    if "subject" not in df.columns:
        df["subject"] = df["video"].apply(subject_from_video)

    for subject, group in df.groupby("subject"):
        baseline_rows = group[group["label"] == 0]
        if len(baseline_rows) == 0:
            baseline_rows = group  # fallback: no Alert clip for this subject
        baseline_mean = baseline_rows[feature_columns].mean().values

        idx = group.index
        raw_vals = df.loc[idx, feature_columns].values.astype(np.float32)
        personalized_vals = np.stack(
            [personalize_vector(row, baseline_mean) for row in raw_vals], axis=0
        )
        df.loc[idx, feature_columns] = personalized_vals

    return df


def build_windows(df: pd.DataFrame, sequence_length: int = 30, stride: int = 5):
    """
    Slide a window over each video's frames independently (never across video boundaries).
    Window label = majority label of frames within the window.
    Returns X: (num_windows, sequence_length, num_features), y: (num_windows,)
    """
    X, y = [], []
    for video_name, group in df.groupby("video"):
        group = group.sort_values("frame_idx")
        feats = group[FEATURE_COLUMNS].values.astype(np.float32)
        labels = group["label"].values.astype(np.int64)

        n = len(group)
        if n < sequence_length:
            continue

        for start in range(0, n - sequence_length + 1, stride):
            end = start + sequence_length
            window_feats = feats[start:end]
            window_labels = labels[start:end]
            majority_label = np.bincount(window_labels).argmax()
            X.append(window_feats)
            y.append(majority_label)

    if not X:
        raise ValueError(
            "No windows could be built. Check that videos have >= sequence_length frames."
        )
    return np.stack(X), np.array(y)


def normalize_features(X: np.ndarray, mean=None, std=None):
    """Z-score normalize each feature channel across the whole dataset."""
    if mean is None or std is None:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X_norm = (X - mean) / std
    return X_norm, mean, std


class DrowsinessSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_prepare(csv_path: str, sequence_length: int, stride: int, personalize: bool = True):
    df = pd.read_csv(csv_path)
    if personalize:
        df = personalize_dataframe(df)
    X, y = build_windows(df, sequence_length, stride)
    X, mean, std = normalize_features(X)
    return X, y, mean, std
