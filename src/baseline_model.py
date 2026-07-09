"""
baseline_model.py

Implements the "EXISTING MODEL" side of the comparison — i.e. what the base paper
(Dipu et al., 2021) effectively does, minus the parts we cannot ethically fabricate
(we are not re-training a full MobileNet-SSD object detector; that needs a large
labeled image-detection dataset the base paper itself had to build separately).

To make the comparison fair and meaningful ON YOUR OWN DATASET, both the baseline
and the proposed LSTM are given exactly the same input information (the same 7-d
EAR/MAR/head-pose features extracted by feature_extraction.py) over the same
sliding windows built by dataset.py. The ONLY thing that differs is:

    Baseline  -> decides from a SINGLE frame / simple fixed rule (no temporal order)
    Proposed  -> LSTM sees the FULL ORDERED SEQUENCE of frames in the window

This isolates exactly what the abstract claims as the contribution: "By learning
sequential behavior rather than relying on isolated frames, the system provides more
stable and accurate detection." Any accuracy gap you observe when you run
compare_models.py on your own recorded data is therefore attributable specifically
to temporal modeling, not to a different feature set — which is a clean, defensible
ablation for a viva.

Two baselines are provided:

1. RuleBasedBaseline
   A direct re-implementation of the base paper's actual decision algorithm
   (Fig. 5 in the base paper): declare drowsy if eyes are found closed for N
   consecutive frames, escalate further if that persists longer / yawning is
   also detected. No learning involved, exactly like the base paper's post-SSD logic.

2. NonTemporalMLBaseline
   A learned classifier (scikit-learn RandomForest or MLP) that only ever sees
   AGGREGATED statistics of a window (mean/std/min/max per channel) — i.e. it has
   access to the same time span as the LSTM but with the ORDER destroyed. This is
   the standard way to isolate "does sequence order matter" in an ablation study,
   and represents the class of "single shot / frame-level deep learning classifier"
   architecture (à la the base paper's MobileNet-SSD) without needing to fabricate
   image-level object-detection training.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


EAR_IDX, MAR_IDX = 2, 3  # indices into the 7-d feature vector: EAR_avg, MAR


class RuleBasedBaseline:
    """Re-implements the base paper's fixed-threshold, consecutive-frame alarm rule."""

    def __init__(self, ear_threshold=0.21, mar_threshold=0.6,
                 highly_drowsy_consecutive=10, drowsy_consecutive=5):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.highly_drowsy_consecutive = highly_drowsy_consecutive
        self.drowsy_consecutive = drowsy_consecutive

    def predict_window(self, window: np.ndarray) -> int:
        """
        window: (sequence_length, 7) RAW (non-normalized) features.
        Mirrors the base paper: count consecutive closed-eye frames within the window.
        """
        ear = window[:, EAR_IDX]
        mar = window[:, MAR_IDX]
        closed = ear < self.ear_threshold
        yawning = np.any(mar > self.mar_threshold)

        max_consecutive = 0
        current = 0
        for c in closed:
            current = current + 1 if c else 0
            max_consecutive = max(max_consecutive, current)

        if max_consecutive >= self.highly_drowsy_consecutive:
            return 2
        if max_consecutive >= self.drowsy_consecutive or yawning:
            return 1
        return 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (num_windows, sequence_length, 7) RAW features -> (num_windows,) predictions."""
        return np.array([self.predict_window(w) for w in X])


def windows_to_aggregate_features(X: np.ndarray) -> np.ndarray:
    """
    Collapse each (sequence_length, 7) window into a fixed-size ORDER-FREE feature
    vector: [mean_0..6, std_0..6, min_0..6, max_0..6] (28-d). This is what the
    non-temporal ML baseline sees — same information content over the same time
    span as the LSTM, but no notion of "what happened first / last".
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([mean, std, mn, mx], axis=1)  # (num_windows, 28)


class NonTemporalMLBaseline:
    """Learned but order-blind baseline: RandomForest or MLP on aggregated window stats."""

    def __init__(self, kind="random_forest", random_state=42):
        if kind == "random_forest":
            self.clf = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=random_state, class_weight="balanced"
            )
        elif kind == "mlp":
            self.clf = MLPClassifier(
                hidden_layer_sizes=(32, 16), max_iter=500, random_state=random_state
            )
        else:
            raise ValueError("kind must be 'random_forest' or 'mlp'")
        self.kind = kind

    def fit(self, X_windows: np.ndarray, y: np.ndarray):
        X_agg = windows_to_aggregate_features(X_windows)
        self.clf.fit(X_agg, y)
        return self

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        X_agg = windows_to_aggregate_features(X_windows)
        return self.clf.predict(X_agg)
