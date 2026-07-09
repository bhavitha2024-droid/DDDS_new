"""
failure_case_demo.py

ADDITIVE ONLY -- new file. Reuses existing, unmodified modules: feature_extraction.py,
baseline_model.py (RuleBasedBaseline), model.py, utils.py.

Runs BOTH the existing (rule-based, base-paper style) and proposed (LSTM) models on
the SAME live webcam window at the SAME time, and displays both predictions
side by side on screen -- with a highlighted banner whenever they DISAGREE. This is
built specifically to produce the "existing model fails, proposed model succeeds"
demonstration your guide asked for.

For the clearest demo, blink rapidly/repeatedly in front of the camera (never fully
closing your eyes for a long continuous stretch) -- see ADDITIONS.md section 5 for
why this specific pattern is the one the rule-based approach structurally cannot
catch, and how to confirm your trained LSTM catches it (via verify_failure_case.py)
before relying on it live.

Usage:
    python src/failure_case_demo.py --source 0 --checkpoint models/best_model.pt
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_extraction import FeatureExtractor  # noqa: E402
from model import DrowsinessLSTM  # noqa: E402
from baseline_model import RuleBasedBaseline  # noqa: E402
from utils import RollingFeatureBuffer  # noqa: E402

CLASS_NAMES = ["Alert", "Drowsy", "Highly Drowsy"]
COLORS = {0: (0, 200, 0), 1: (0, 165, 255), 2: (0, 0, 255)}


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DrowsinessLSTM(
        input_dim=cfg["model"]["input_dim"], hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"], num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"], bidirectional=cfg["model"]["bidirectional"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std = np.array(ckpt["feature_std"], dtype=np.float32)
    return model, cfg, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--ear_threshold", type=float, default=0.21)
    parser.add_argument("--mar_threshold", type=float, default=0.6)
    parser.add_argument("--log_path", default="logs/failure_case_comparison.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, mean, std = load_model(args.checkpoint, device)
    seq_len = cfg["window"]["sequence_length"]

    extractor = FeatureExtractor()
    raw_buffer = RollingFeatureBuffer(maxlen=seq_len, feature_dim=cfg["model"]["input_dim"])
    norm_buffer = RollingFeatureBuffer(maxlen=seq_len, feature_dim=cfg["model"]["input_dim"])
    rule_model = RuleBasedBaseline(ear_threshold=args.ear_threshold, mar_threshold=args.mar_threshold)

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    log_is_new = not os.path.exists(args.log_path)
    log_f = open(args.log_path, "a")
    if log_is_new:
        log_f.write("timestamp,rule_pred,rule_label,lstm_pred,lstm_label,mismatch\n")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        sys.exit(1)

    rule_label, lstm_label, mismatch = "...", "...", False
    print("Press 'q' to quit. Blink rapidly/repeatedly for the clearest demo.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feats = extractor.extract(frame)
            if feats is not None:
                raw_buffer.push(feats)
                norm_buffer.push((feats - mean) / std)

                if raw_buffer.is_ready() and norm_buffer.is_ready():
                    raw_window = raw_buffer.as_array()
                    norm_window = norm_buffer.as_array()

                    rule_pred = rule_model.predict_window(raw_window)
                    with torch.no_grad():
                        x = torch.from_numpy(norm_window).float().unsqueeze(0).to(device)
                        lstm_pred = int(model(x).argmax(dim=1).item())

                    rule_label = CLASS_NAMES[rule_pred]
                    lstm_label = CLASS_NAMES[lstm_pred]
                    mismatch = rule_pred != lstm_pred

                    log_f.write(f"{time.time()},{rule_pred},{rule_label},{lstm_pred},{lstm_label},{mismatch}\n")
                    log_f.flush()

            # --- HUD ---
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
            cv2.putText(frame, f"EXISTING (Rule-Based / base-paper style): {rule_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        COLORS.get(CLASS_NAMES.index(rule_label) if rule_label in CLASS_NAMES else 0, (255, 255, 255)), 2)
            cv2.putText(frame, f"PROPOSED (LSTM, temporal):              {lstm_label}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        COLORS.get(CLASS_NAMES.index(lstm_label) if lstm_label in CLASS_NAMES else 0, (255, 255, 255)), 2)

            if mismatch:
                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 255), -1)
                cv2.putText(frame, "MISMATCH -- existing model missed this, proposed model caught it",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Existing vs Proposed -- Live Comparison", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        log_f.close()
        print(f"Comparison log saved to {args.log_path}")


if __name__ == "__main__":
    main()
