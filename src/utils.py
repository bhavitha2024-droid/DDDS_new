"""
utils.py

Small helpers shared across modules: temporal smoothing of predictions and a
rolling feature buffer used at real-time inference time.
"""

from collections import deque, Counter
import numpy as np


class RollingFeatureBuffer:
    """Keeps the last `maxlen` per-frame feature vectors for LSTM windowed inference."""

    def __init__(self, maxlen: int, feature_dim: int):
        self.maxlen = maxlen
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=maxlen)

    def push(self, feature_vec: np.ndarray):
        self.buffer.append(feature_vec)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.maxlen

    def as_array(self) -> np.ndarray:
        return np.stack(self.buffer, axis=0)  # (maxlen, feature_dim)

    def clear(self):
        self.buffer.clear()


class PredictionSmoother:
    """
    Smooths raw per-window LSTM predictions over time so a single noisy window
    cannot flip the driver's reported state. Uses majority vote over the last K
    predictions, which directly implements the abstract's goal of "more stable and
    accurate detection" versus the base paper's single fixed-frame-count rule.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, prediction: int) -> int:
        self.history.append(prediction)
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.history.clear()
