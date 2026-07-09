"""
realtime_infer.py

End-to-end real-time demo: webcam/video -> preprocessing -> feature extraction ->
rolling window -> LSTM -> temporal smoothing -> graded alert manager -> on-screen HUD.

This script is the concrete realization of the abstract's final claim: "The proposed
system is efficient for real-time applications and enhances road safety by combining
reliable detection with proactive alerting."

Usage:
    python realtime_infer.py --source 0 --checkpoint models/best_model.pt
    python realtime_infer.py --source path/to/video.mp4 --checkpoint models/best_model.pt
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video file path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sound_dir", default=None, help="Optional dir with chime.wav/siren.wav")
    parser.add_argument("--driver_id", default=None,
                         help="Optional: personalize using this driver's calibration profile "
                              "(see calibrate_driver.py). If omitted, uses global normalization only.")
    parser.add_argument("--profile_dir", default="models/driver_profiles")
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
            "reliable. Run `python src/calibrate_driver.py --driver_id <name>` first, then "
            "pass --driver_id <name> here."
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

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        sys.exit(1)

    prev_time = time.time()
    current_level = 0

    print("Press 'q' to quit.")
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
            print("EAR =", feats[2], "MAR =", feats[3], "Pitch =", feats[4], "Yaw =", feats[5], "Roll =", feats[6])
            if driver_profile:
                extra_text += f" | driver={args.driver_id} (personalized)"

            if buffer.is_ready():
                window = buffer.as_array()  # (seq_len, 7)
                x = torch.from_numpy(window).float().unsqueeze(0).to(device)  # (1, seq_len, 7)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    pred = int(logits.argmax(dim=1).item())
                    print("Prediction =", pred, "Probabilities =", probs.cpu().numpy())
                current_level = smoother.update(pred)
                action = alert_mgr.process(current_level)
                if action["fire"]:
                    print(f"[ALERT] {action['level_name']} triggered at {time.strftime('%X')}")

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_hud(frame, current_level, fps, extra_text)
        cv2.imshow("Intelligent Driver Drowsiness Detection (LSTM)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Session summary:", alert_mgr.summary())


if __name__ == "__main__":
    main()
