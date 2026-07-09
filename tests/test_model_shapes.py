"""
test_model_shapes.py

Basic sanity tests that don't require a webcam, dataset, or GPU. Run with:
    python -m pytest tests/
or:
    python tests/test_model_shapes.py
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import DrowsinessLSTM  # noqa: E402
from utils import RollingFeatureBuffer, PredictionSmoother  # noqa: E402
from dataset import build_windows, normalize_features  # noqa: E402
import pandas as pd  # noqa: E402


def test_model_forward_shape():
    model = DrowsinessLSTM(input_dim=7, hidden_dim=16, num_layers=1, num_classes=3)
    x = torch.randn(4, 30, 7)
    out = model(x)
    assert out.shape == (4, 3), f"Unexpected output shape: {out.shape}"


def test_rolling_buffer():
    buf = RollingFeatureBuffer(maxlen=5, feature_dim=7)
    assert not buf.is_ready()
    for _ in range(5):
        buf.push(np.random.randn(7).astype(np.float32))
    assert buf.is_ready()
    arr = buf.as_array()
    assert arr.shape == (5, 7)


def test_prediction_smoother_majority_vote():
    smoother = PredictionSmoother(window_size=3)
    assert smoother.update(0) == 0
    assert smoother.update(1) in (0, 1)
    result = smoother.update(1)
    assert result == 1  # majority of [0,1,1] is 1


def test_build_windows_synthetic():
    rows = []
    for f in range(40):
        rows.append({
            "video": "vid1.mp4", "frame_idx": f, "label": 0 if f < 20 else 1,
            "EAR_left": 0.3, "EAR_right": 0.3, "EAR_avg": 0.3, "MAR": 0.2,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        })
    df = pd.DataFrame(rows)
    X, y = build_windows(df, sequence_length=10, stride=5)
    assert X.shape[1:] == (10, 7)
    assert len(X) == len(y)

    X_norm, mean, std = normalize_features(X)
    assert X_norm.shape == X.shape
    assert mean.shape == (7,)


if __name__ == "__main__":
    test_model_forward_shape()
    test_rolling_buffer()
    test_prediction_smoother_majority_vote()
    test_build_windows_synthetic()
    print("All tests passed.")
