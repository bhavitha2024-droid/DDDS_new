"""
realtime_infer_live.py

ADDITIVE ONLY -- this is a new file. `src/realtime_infer.py` is untouched and still
works exactly as before if you just want the webcam + HUD + alerts with no logging.

This script does the *same* pipeline as `realtime_infer.py` (it reuses the exact same
existing modules: feature_extraction, model, utils, alert_system, calibration -- no
duplicated/forked logic there), but additionally appends one row per processed frame
to a CSV log file. That log file is what the new `dashboard_streamlit.py` "Live
Monitor" tab tails to draw live-updating graphs in the browser, the same way the
second reviewed project's detector + Streamlit dashboard worked together.

Usage:
    python src/realtime_infer_live.py --source 0 --checkpoint models/best_model.pt

Then, in another terminal:
    streamlit run dashboard_streamlit.py
    -> open the "Live Monitor" tab

Log file written to: logs/live_features.csv (created automatically)
Columns: timestamp, EAR_left, EAR_right, EAR_avg, MAR, pitch, yaw, roll, level, level_name
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_extraction import FeatureExtractor  # noqa: E402
from model import DrowsinessLSTM  # noqa: E402
from utils import RollingFeatureBuffer, PredictionSmoother  # noqa: E402
from alert_system import AlertManager, LEVEL_NAMES  # noqa: E402
from calibration import DriverProfile  # noqa: E402

COLORS = {
    0: (0, 200, 0),      # Alert -> green
    1: (0, 165, 255),    # Drowsy -> amber
    2: (0, 0, 255),      # Highly Drowsy -> red
}

LOG_COLUMNS = [
    "timestamp",
    "EAR_left",
    "EAR_right",
    "EAR_avg",
    "MAR",
    "pitch",
    "yaw",
    "roll",
    "confidence",
    "fps",
    "level",
    "level_name",
]


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DrowsinessLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std = np.array(ckpt["feature_std"], dtype=np.float32)
    personalize = ckpt.get("personalize", True)
    return model, cfg, mean, std, personalize


def draw_hud(frame, level, fps, extra_text=""):
    color = COLORS.get(level, (255, 255, 255))
    label = LEVEL_NAMES.get(level, "UNKNOWN")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
    cv2.putText(frame, f"STATUS: {label}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if extra_text:
        cv2.putText(frame, extra_text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def open_log_writer(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    is_new = not os.path.exists(log_path)
    f = open(log_path, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(LOG_COLUMNS)
        f.flush()
    return f, writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video file path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sound_dir", default=None, help="Optional dir with chime.wav/siren.wav")
    parser.add_argument("--driver_id", default=None,
                         help="Optional: personalize using this driver's calibration profile.")
    parser.add_argument("--profile_dir", default="models/driver_profiles")
    parser.add_argument("--log_path", default="logs/live_features.csv",
                         help="CSV file the Streamlit dashboard's Live Monitor tab tails.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, mean, std, model_expects_personalized = load_model(args.checkpoint, device)

    driver_profile = None
    if args.driver_id:
        try:
            driver_profile = DriverProfile.load(args.driver_id, args.profile_dir)
            print(f"Loaded calibration profile for driver '{args.driver_id}' "
                  f"({driver_profile.num_samples} baseline samples). Personalization ENABLED.")
        except FileNotFoundError as e:
            print(f"{e}\nFalling back to non-personalized (global) thresholds/normalization.")

    if model_expects_personalized and driver_profile is None:
        print(
            "WARNING: this checkpoint was trained WITH personalization enabled, but no "
            "--driver_id / calibration profile was supplied. Predictions will be less "
            "reliable."
        )

    seq_len = cfg["window"]["sequence_length"]
    extractor = FeatureExtractor()
    buffer = RollingFeatureBuffer(maxlen=seq_len, feature_dim=cfg["model"]["input_dim"])
    smoother = PredictionSmoother(window_size=cfg.get("alerts", {}).get("smoothing_window", 10))
    alert_mgr = AlertManager(
        drowsy_cooldown_sec=cfg.get("alerts", {}).get("drowsy_cooldown_sec", 4),
        highly_drowsy_cooldown_sec=cfg.get("alerts", {}).get("highly_drowsy_cooldown_sec", 2),
        sound_dir=args.sound_dir,
    )

    log_file, log_writer = open_log_writer(args.log_path)
    print(f"Logging live features to: {args.log_path} (watched by dashboard_streamlit.py)")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        sys.exit(1)

    prev_time = time.time()
    current_level = 0

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feats = extractor.extract(frame)
            extra_text = "No face detected"
            if feats is not None:
                feats_for_model = driver_profile.personalize(feats) if driver_profile else feats
                feats_norm = (feats_for_model - mean) / std
                buffer.push(feats_norm)
                extra_text = f"EAR={feats[2]:.2f} MAR={feats[3]:.2f}"
                if driver_profile:
                   extra_text += f" | driver={args.driver_id} (personalized)"
                confidence = 0.0
                if buffer.is_ready():
                    window = buffer.as_array()
                    x = torch.from_numpy(window).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = F.softmax(logits, dim=1)
                        confidence = float(probs.max().item())
                        pred = int(probs.argmax(dim=1).item())
                    current_level = smoother.update(pred)
                    action = alert_mgr.process(current_level)
                    if action["fire"]:
                        print(f"[ALERT] {action['level_name']} triggered at {time.strftime('%X')}")

                # log every processed frame with a face, regardless of whether the
                # window/LSTM has produced a fresh prediction yet
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now    
                log_writer.writerow([
                        time.time(),
                        feats[0],
                        feats[1],
                        feats[2],
                        feats[3],
                        feats[4],
                        feats[5],
                        feats[6],
                        confidence if buffer.is_ready() else 0,
                        fps,
                        current_level,
                        LEVEL_NAMES.get(current_level, "UNKNOWN"),
                    ])
                log_file.flush()
                frame = draw_hud(frame, current_level, fps, extra_text)
                cv2.imshow("Intelligent Driver Drowsiness Detection (LSTM) - Live Logging", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        log_file.close()
        print("Session summary:", alert_mgr.summary())


if __name__ == "__main__":
    main()
