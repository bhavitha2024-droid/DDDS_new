# Detailed Comparison: This Project vs. the Base Paper

Base paper: Dipu, M.T.A., Hossain, S.S., Arafat, Y., Rafiq, F.B. (2021). *Real-time Driver
Drowsiness Detection using Deep Learning.* IJACSA, 12(7), 844-850.

## Architecture

| Aspect | Base paper | This project |
|---|---|---|
| Core model | MobileNet backbone + SSD object-detection head | MediaPipe FaceMesh landmark extraction + 2-layer LSTM |
| Task framing | Per-frame 4-class object detection (eye-open, eye-close, yawn, no-yawn) | Sequence classification over a sliding window of behavioral features |
| Temporal reasoning | None — decision based on counting consecutive closed-eye frames (fixed threshold = 10) | Explicit — LSTM hidden state carries information across the whole window (default 30 frames) |
| Head movement | Not modeled | Pitch/yaw/roll computed every frame via solvePnP and fed to the LSTM |
| Yawn fusion into final alarm | Explicitly *not* used for the alarm decision (stated as future work in the paper's conclusion) | MAR (mouth aspect ratio) is one of the 7 LSTM input features, directly influencing the drowsiness class |
| Alert behavior | Single binary alarm | Three-level graded alert (Alert / Drowsy / Highly Drowsy) with per-level cooldowns |
| Low-light robustness | Reported as a weak point (raw-pixel CNN struggles under glare/dim light) | CLAHE-based illumination normalization + geometric (ratio/angle) features, which are inherently less sensitive to absolute brightness |
| Reported accuracy | mAP ≈ 0.98 on PASCAL VOC metric for the 4 detection classes (single-frame task, not fatigue prediction) | Sequence-level accuracy/precision/recall/F1 reported via `evaluate.py`; metric target is the fatigue *state*, not per-frame object detection |
| Inference cost | Full SSD forward pass every frame | Lightweight per-frame landmark extraction + a small LSTM (2 layers, 64 hidden units) evaluated once per window — designed to be cheaper than a full detector re-run each frame |

## Why sequence classification is more appropriate for drowsiness

Fatigue is not a single-frame property — it is a *trend*: eyelids closing more slowly, blinks
lasting longer, yawns recurring, the head drifting down and correcting repeatedly. The base
paper's own related-work review acknowledges PERCLOS (percentage of eye closure over time) as a
standard fatigue metric, yet its own proposed system reduces this temporal idea to a single fixed
rule ("10 consecutive frames"). This project generalizes that idea properly by letting an LSTM
learn arbitrary temporal patterns directly from data, rather than hand-picking one threshold.

## What is intentionally kept from the base paper

- The general behavioral cues used (eye state, yawning) are retained, since the base paper
  correctly identifies these as the most literature-supported drowsiness indicators.
- A lightweight, real-time-first design philosophy (no expensive per-frame detector, must run on
  commodity hardware) is preserved and, if anything, made cheaper.
- Standard evaluation practice (train/val/test split, precision/recall/F1) is kept for
  comparability, adapted to sequence classification.
