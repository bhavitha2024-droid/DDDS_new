# Intelligent Driver Drowsiness Detection using Temporal Analysis (LSTM)

**Author:** N. Bhavitha, M.Tech CSE (24886M601009)
**Base paper:** Dipu, Hossain, Arafat, Rafiq — *"Real-time Driver Drowsiness Detection using Deep
Learning"*, IJACSA Vol. 12, No. 7, 2021 (MobileNet-SSD, frame-level, eye-open/eye-close/yawn/no-yawn).

This repository implements the system described in your abstract: a driver-drowsiness pipeline
that fuses **eye closure, yawning, and head-pose/movement cues**, tracks them **over time with an
LSTM**, and drives a **graded, intelligent alert mechanism** — instead of the base paper's
single-frame / fixed-threshold ("10 consecutive closed frames") CNN-SSD approach.

---

## 1. Why this extends the base paper (limitation → fix)

| # | Limitation in the base paper (Dipu et al., 2021) | How this project overcomes it |
|---|---|---|
| 1 | **Frame-level / fixed-threshold logic.** Drowsy is declared only if eyes are closed for *exactly* "10 consecutive frames" — a hard-coded, FPS-dependent rule with no notion of a trend. | Features are collected into a **sliding temporal window** (default 30 frames ≈ 1 s at 30 FPS, but window length is a config parameter, not baked into the logic) and passed to an **LSTM** that learns the *sequential pattern* of drowsiness (slow eyelid closing, repeated micro-sleeps, gradual head droop), not just a frame count. |
| 2 | **Only eye state is used for the final alarm** (yawn/no-yawn classes exist but are *not* fused into the alarm decision — the paper explicitly lists this as future work: "Further enhancement... using a yawning dataset as we could not use those annotations for detecting drowsiness"). | Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR / yawning) and **head-pose (pitch/yaw/roll + nodding)** are extracted every frame and **all three streams are fused as a single feature vector per timestep** fed to the LSTM, so yawning and head-nod events genuinely influence the drowsiness score. |
| 3 | **Binary/one-shot alarm** (alarm rings or not). No levels, no early warning. | The LSTM outputs a **3-class drowsiness level (Alert / Drowsy / Highly Drowsy)** each timestep, which is smoothed over time and mapped to a **graded alert system** (soft visual cue → audible chime → escalating alarm), giving the driver time to react instead of an abrupt single alarm. |
| 4 | **Object-detection framing (SSD) is heavier and less suited to *behavioral trend* modeling** — SSD only ever answers "what is in this frame", never "how has the driver been behaving". | Landmark-based lightweight geometric features (EAR/MAR/head angles from **MediaPipe Face Mesh**, 468 landmarks, CPU-friendly) replace the SSD object-detection backbone, and the temporal reasoning is delegated entirely to the LSTM — this is both **more accurate for fatigue (a temporal phenomenon)** and **cheaper to run in real time** than re-running a full SSD detector every frame. |
| 5 | **Reported limitation: poor performance in low light / flashlight glare** — the paper's CNN relies on raw pixel appearance. | Because features are **geometric ratios (EAR/MAR) and pose angles** rather than raw-pixel classification, and the pipeline includes a **CLAHE-based illumination-normalization step** before landmark detection (`src/preprocessing.py`), the system degrades more gracefully under uneven/low light than an appearance-only CNN classifier. |
| 6 | **No notion of "different drivers have different resting eye/mouth openness."** A single global EAR threshold is applied to every driver, even though eye shape, eyelid hooding, glasses, and camera angle all shift what a "normal, alert" EAR looks like per person. | A **personalization/calibration module** (`src/calibration.py`, `src/calibrate_driver.py`) records a short per-driver baseline and expresses all downstream EAR/MAR/pose features RELATIVE to that driver's own baseline (ratio for EAR/MAR, delta for head pose) — the same transform is applied consistently at training time (`dataset.py: personalize_dataframe`, subject-relative normalization) and at real-time inference (`DriverProfile.personalize`), so the shared LSTM learns a person-independent notion of "how drowsy relative to your own normal" rather than absolute values tuned to whoever was recorded. |
| 7 | **No confidence/temporal smoothing** — a single misclassified frame can trigger or miss an alarm. | An **Exponential Moving Average + majority-vote smoothing** layer sits between the LSTM output and the alert manager, so isolated blinks or a single bad frame cannot flip the driver state. |

---

## 2. Pipeline overview

```
Camera / Video
     │
     ▼
[Preprocessing]  CLAHE illumination normalization, resize, face detection
     │
     ▼
[Feature Extraction]  MediaPipe FaceMesh → EAR (both eyes), MAR (yawn), head pitch/yaw/roll
     │  (per-frame feature vector, dim = 7)
     ▼
[Sliding Window Buffer]  last N frames → shape (N, 7)
     │
     ▼
[LSTM Temporal Model]  2-layer LSTM + FC head → drowsiness class per window
     │  {0: Alert, 1: Drowsy, 2: Highly Drowsy}
     ▼
[Temporal Smoothing]  EMA + majority vote over last K predictions
     │
     ▼
[Intelligent Alert Manager]  level-based response (visual → chime → siren) + logging
```

---

## 3. Repository layout

```
DriverDrowsinessLSTM/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── record_session.py      # records YOUR OWN labeled webcam sessions (no public dataset)
│   ├── calibrate_driver.py     # PERSONALIZATION: records a per-driver baseline profile
│   ├── calibration.py          # PERSONALIZATION: DriverProfile + subject-relative feature transform
│   ├── preprocessing.py       # illumination normalization / frame prep
│   ├── feature_extraction.py  # EAR, MAR, head-pose via MediaPipe FaceMesh
│   ├── dataset.py             # builds (sequence, label) windows from a features CSV
│   ├── model.py                # LSTM model definition (PyTorch) -- the proposed model
│   ├── baseline_model.py       # EXISTING-style models: rule-based + non-temporal ML
│   ├── compare_models.py       # trains/evaluates baseline(s) vs. LSTM side-by-side
│   ├── train.py                # training loop, checkpointing, metrics
│   ├── evaluate.py             # accuracy / precision / recall / F1 / confusion matrix
│   ├── alert_system.py         # graded alert manager (visual/audio escalation)
│   ├── realtime_infer.py       # webcam / video real-time demo, end-to-end
│   └── utils.py                # smoothing, logging helpers
├── data/
│   ├── raw/                    # your recorded videos land here (record_session.py)
│   └── processed/              # extracted per-frame feature CSVs land here
├── models/                     # saved .pt checkpoints
├── docs/
│   ├── comparison_with_base_paper.md   # architecture-level comparison vs. the base paper
│   ├── novelty_and_review_guide.md     # honest project-level assessment + viva prep
│   └── results/                        # compare_models.py writes its output here
└── tests/
    └── test_model_shapes.py
```

---

## 4. Quick start — using YOUR OWN dataset (no public dataset required)

```bash
pip install -r requirements.txt

# 1) Record your own labeled sessions (repeat for each subject / class / lighting condition)
python src/record_session.py --subject bhavitha --label alert       --duration 30
python src/record_session.py --subject bhavitha --label drowsy      --duration 30
python src/record_session.py --subject bhavitha --label highdrowsy  --duration 30
# -> saved as data/raw/bhavitha_alert_<timestamp>.mp4 etc.
# See src/record_session.py docstring for the recommended recording protocol
# (multiple subjects/sessions/lighting conditions for a defensible dataset).

# 2) Extract behavioral features (EAR, MAR, head pose) from your recorded videos
python src/feature_extraction.py --video_dir data/raw --out data/processed/features.csv

# 3) Train the proposed LSTM on the windowed sequences
#    Personalization (subject-relative EAR/MAR/pose normalization) is ON by default;
#    add --no_personalize to train on raw absolute features instead.
python src/train.py --features data/processed/features.csv --epochs 40 --window 30

# 4) Evaluate the LSTM alone (automatically uses whatever personalization setting
#    the checkpoint was trained with)
python src/evaluate.py --checkpoint models/best_model.pt --features data/processed/features.csv

# 5) Compare EXISTING approaches vs. the PROPOSED LSTM on your own data
python src/compare_models.py --features data/processed/features.csv --window 30 --out docs/results
#    -> prints + saves accuracy/precision/recall/F1/confusion-matrix for:
#         (a) Rule-based baseline (re-implements the base paper's fixed 10-frame rule)
#         (b) Non-temporal ML baseline (RandomForest on aggregated window stats)
#         (c) Proposed LSTM (sees the full ordered sequence)
#    All three see the SAME features from the SAME windows, so the comparison is fair —
#    see docs/comparison_with_base_paper.md and docs/novelty_and_review_guide.md.

# 6) PERSONALIZATION: calibrate a specific driver before deploying in real time
#    (records ~12s of that driver looking normally alert, stores their own baseline)
python src/calibrate_driver.py --driver_id bhavitha --duration 12

# 7) Run real-time detection on a webcam (0) or a video file, personalized to that driver
python src/realtime_infer.py --source 0 --checkpoint models/best_model.pt --driver_id bhavitha
```

`config.yaml` centralizes window size, EAR/MAR thresholds (used only as auxiliary sanity
checks / by the rule-based baseline — the LSTM does the real classification), alert
cooldowns, and model hyperparameters.

### A note on accuracy
The comparison script reports **real, reproducible metrics computed on whatever data you
give it** — this README and the code deliberately do not hard-code or promise a specific
accuracy number. Your result depends on dataset size/quality (see the recording protocol in
`src/record_session.py`), class balance, and hyperparameters. To push accuracy up in a
legitimate way: record more sessions per class, vary lighting/pose, keep classes balanced,
try `sequence_length` 20-45 and `hidden_dim` 32-128 in `config.yaml`, and check
`docs/results/comparison_summary.csv` after each run.

---

## 5. Feature vector definition (per frame)

| Index | Feature | Description |
|---|---|---|
| 0 | `EAR_left` | Eye Aspect Ratio, left eye |
| 1 | `EAR_right` | Eye Aspect Ratio, right eye |
| 2 | `EAR_avg` | Mean of both eyes |
| 3 | `MAR` | Mouth Aspect Ratio (yawning indicator) |
| 4 | `pitch` | Head pitch (nodding, degrees) |
| 5 | `yaw` | Head yaw (left/right turn, degrees) |
| 6 | `roll` | Head roll (tilt, degrees) |

These 7 values per frame form the input at every LSTM timestep — this is the concrete
implementation of "eye closure, yawning, and head movement" fused into a single temporal model,
as described in your abstract.

---

## 6. Alert levels

| LSTM class | Meaning | Alert action |
|---|---|---|
| 0 – Alert | Normal driving | No alert, HUD shows green status |
| 1 – Drowsy | Sustained partial eye closure / occasional yawns / slow head droop trend | Visual amber warning + soft chime, logged with timestamp |
| 2 – Highly Drowsy | Sustained eye closure / repeated yawns / nodding trend over the window | Escalating audible alarm + on-screen red warning, suggests pulling over |

Alert escalation includes a cooldown so it doesn't spam once the state changes back to Alert.

---

## 7. Notes on training data

This project ships **code and architecture**, not a pre-trained checkpoint or any bundled
dataset. It is built specifically around **your own self-recorded dataset**
(`src/record_session.py`) rather than a public dataset — `feature_extraction.py` and
`dataset.py` work directly off whatever labeled videos you place in `data/raw/`.

## 8. Personalization (driver calibration)

A note on scope: the abstract text you originally gave me does not use the word
"personalization" — it specifies eye closure / yawning / head movement, LSTM temporal
modeling, and an intelligent alert mechanism. Personalization is included here as an
**added extension**, not a re-discovery of something already in your abstract; if your
actual signed synopsis document mentions it, align the wording there.

The reasoning for including it: the base paper applies one fixed global EAR threshold to
every driver, but resting eye openness genuinely varies by person (eye shape, eyelid
hooding, glasses, camera angle) — a threshold tuned for one face is routinely wrong for
another. This project addresses that with:

- `src/calibrate_driver.py` — records ~10-15s of a specific driver looking normally
  alert, and saves their baseline to `models/driver_profiles/<driver_id>.json`.
- `src/calibration.py` — defines the transform: EAR/MAR become a **ratio** to that
  driver's own baseline (≈1.0 = normal openness), and head pose becomes a **delta** from
  their own resting orientation.
- `src/dataset.py: personalize_dataframe` applies the **same transform at training
  time**, per subject in your recorded dataset, so the shared LSTM is trained on
  person-relative features from the start (enabled by default; see `config.yaml ->
  personalization.enabled`, or pass `--no_personalize` to `train.py` /
  `compare_models.py` to disable it and train on raw absolute features instead).
- `src/realtime_infer.py --driver_id <name>` applies that same calibrated baseline at
  deployment time, so training and inference stay consistent.

If you'd rather NOT claim personalization as part of this project (e.g. because your
signed abstract/synopsis is fixed and reviewers will check it against the document),
just don't run `calibrate_driver.py` / pass `--no_personalize` — the rest of the system
works identically without it.

## 9. Existing vs. Proposed model comparison

`src/compare_models.py` is the head-to-head evaluation you'll present at your review: it
trains a rule-based baseline (re-implementing the base paper's fixed 10-consecutive-frame
alarm logic), a non-temporal ML baseline (RandomForest on aggregated window statistics), and
the proposed LSTM, all on the *same* windows from your own dataset, and writes
`docs/results/comparison_summary.csv` + `docs/results/accuracy_comparison.png`. See
`docs/novelty_and_review_guide.md` for how to present this comparison, what novelty claims
are defensible, and likely viva questions.
