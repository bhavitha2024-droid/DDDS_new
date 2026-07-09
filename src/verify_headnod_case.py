"""
verify_headnod_case.py

ADDITIVE ONLY -- new file, does not modify anything else.

Targets a SECOND, even more structurally guaranteed weakness than rapid blinking:

    RuleBasedBaseline.predict_window() (src/baseline_model.py, unmodified) only ever
    reads EAR (index 2) and MAR (index 3). It never looks at pitch/yaw/roll
    (indices 4, 5, 6) at all. This mirrors the actual base paper, which never
    computes head pose in the first place (see feature_extraction.py docstring).

    Therefore: a window where the driver's HEAD DROOPS/NODS (large pitch change)
    while eyes stay technically "open" (EAR above threshold) and there's no yawn,
    is GUARANTEED -- by code inspection, not just typical-case behaviour -- to be
    classified "Alert" by the existing rule-based approach, every single time.

This script feeds a synthetic head-nodding window to both:
    1. RuleBasedBaseline (existing / base-paper style)   -- provably says "Alert"
    2. Your trained LSTM (models/best_model.pt)          -- may or may not catch it,
       depending on whether your training data included head-nodding examples

Usage:
    python src/verify_headnod_case.py --checkpoint models/best_model.pt
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


def make_headnod_window(seq_len=30, normal_ear=0.30, normal_mar=0.25):
    """
    Eyes stay OPEN (EAR well above a 0.21 threshold) and mouth stays closed
    (no yawn) throughout -- but pitch drifts from ~0 (head level) to a large
    negative value (head drooping forward) and partially back, simulating a
    driver nodding off while their eyes remain open.
    """
    rows = []
    for i in range(seq_len):
        # smooth nod: pitch dips down in the middle of the window then partially recovers
        t = i / (seq_len - 1)
        pitch = -35.0 * np.sin(np.pi * t)  # degrees; 0 at start/end, -35 at midpoint
        rows.append([normal_ear, normal_ear, normal_ear, normal_mar, pitch, 0.0, 0.0])
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

    window = make_headnod_window(cfg["window"]["sequence_length"])

    # 1. Rule-based (existing / base-paper style) -- guaranteed to ignore pitch entirely
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

    print(f"\nSynthetic window: eyes OPEN throughout (EAR={window[0,2]:.2f}), no yawn, "
          f"but head pitch dips to {window[:,4].min():.0f} degrees (simulated head-nod/microsleep)\n")
    print(f"Existing (Rule-Based, base-paper style) prediction: {CLASS_NAMES[rule_pred]}  "
          f"<- guaranteed 'Alert' by code inspection, since pitch is never read")
    print(f"Proposed (LSTM) prediction:                          {CLASS_NAMES[lstm_pred]}")
    print(f"LSTM class probabilities [Alert, Drowsy, Highly Drowsy]: {np.round(probs, 3)}\n")

    if rule_pred == 0 and lstm_pred != 0:
        print(">>> SUCCESS: a second, even more structurally guaranteed case where the "
              "existing approach cannot possibly detect drowsiness (it never reads head "
              "pose at all), while your LSTM correctly catches it.")
    elif rule_pred == lstm_pred:
        print(">>> The rule-based result (Alert) is guaranteed and will never change. "
              "If your LSTM also says Alert here, your training clips likely didn't "
              "include head-nodding examples -- record 1-2 short clips of yourself "
              "slowly nodding off with eyes open, label them following your existing "
              "convention, and retrain (see ADDITIONS.md section 6).")


if __name__ == "__main__":
    main()
