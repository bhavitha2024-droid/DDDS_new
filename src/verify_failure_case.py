"""
verify_failure_case.py

ADDITIVE ONLY -- new file, does not modify anything else.

Quick sanity check: feeds a SYNTHETIC rapid-blinking pattern (eyes closed 65-70% of
a 30-frame window, but NEVER 5 consecutive closed frames) to both:

    1. RuleBasedBaseline (src/baseline_model.py, unmodified)   -- the base-paper-style rule
    2. Your trained LSTM (models/best_model.pt, unmodified)    -- the proposed model

and prints both predictions side by side, so you can confirm on your own machine
whether your LSTM already catches this pattern that the rule structurally cannot.

Usage:
    python src/verify_failure_case.py --checkpoint models/best_model.pt
"""

import argparse
import sys
import os

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__)))
from baseline_model import RuleBasedBaseline  # noqa: E402
from model import DrowsinessLSTM  # noqa: E402

CLASS_NAMES = ["Alert", "Drowsy", "Highly Drowsy"]


def make_rapid_blink_window(seq_len=30):
    """EAR oscillates closed/closed/half-open, closed/closed/half-open... -- never
    5 consecutive closed frames, but eyes are closed ~65-70% of the window overall."""
    pattern = [0.10, 0.10, 0.35, 0.10, 0.10, 0.35]
    rows = []
    for i in range(seq_len):
        ear = pattern[i % len(pattern)]
        rows.append([ear, ear, ear, 0.30, 0.0, 0.0, 0.0])  # EAR_l, EAR_r, EAR_avg, MAR, pitch, yaw, roll
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--ear_threshold", type=float, default=0.21)
    parser.add_argument("--mar_threshold", type=float, default=0.6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std = np.array(ckpt["feature_std"], dtype=np.float32)

    window = make_rapid_blink_window(cfg["window"]["sequence_length"])
    pct_closed = (window[:, 2] < args.ear_threshold).mean() * 100

    # 1. Rule-based (existing / base-paper style)
    rule_model = RuleBasedBaseline(ear_threshold=args.ear_threshold, mar_threshold=args.mar_threshold)
    rule_pred = rule_model.predict_window(window)

    # 2. Proposed LSTM
    window_norm = (window - mean) / std
    model = DrowsinessLSTM(
        input_dim=cfg["model"]["input_dim"], hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"], num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"], bidirectional=cfg["model"]["bidirectional"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(window_norm).float().unsqueeze(0).to(device)
        logits = model(x)
        lstm_pred = int(logits.argmax(dim=1).item())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    print(f"\nSynthetic window: eyes closed {pct_closed:.0f}% of the time, "
          f"but never 5+ consecutive closed frames (simulated rapid blinking)\n")
    print(f"Existing (Rule-Based, base-paper style) prediction: {CLASS_NAMES[rule_pred]}")
    print(f"Proposed (LSTM) prediction:                          {CLASS_NAMES[lstm_pred]}")
    print(f"LSTM class probabilities [Alert, Drowsy, Highly Drowsy]: {np.round(probs, 3)}\n")

    if rule_pred == 0 and lstm_pred != 0:
        print(">>> SUCCESS: this is a clean, demonstrable case where the existing "
              "rule-based approach misses drowsiness that your LSTM correctly catches.")
    elif rule_pred == lstm_pred:
        print(">>> Both models currently agree on this synthetic pattern. See "
              "ADDITIONS.md section 5 for how to make your LSTM learn to catch this "
              "pattern by adding a few rapid-blinking training clips.")


if __name__ == "__main__":
    main()
