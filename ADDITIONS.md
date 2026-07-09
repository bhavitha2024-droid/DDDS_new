# Additions (ported from the second reviewed project)

This file documents two things added to this project as-is, from the other
"drowsiness-detection" project that was compared against it. **Nothing else in this
repository was changed.**

## 1. `src/low_light.py`

A standalone low-light enhancement module: gamma correction -> CLAHE -> histogram
equalization, stacked. It is a more aggressive 3-stage alternative to the CLAHE-only
step already used elsewhere in this project (`src/preprocessing.py`, untouched).

It is **not** automatically wired into the pipeline. To use it, import it yourself, e.g.
in `feature_extraction.py` or `realtime_infer.py`:

```python
from low_light import enhance_low_light
frame = enhance_low_light(frame)   # before passing frame to the feature extractor
```

## 2. `dashboard_streamlit.py`

A read-only Streamlit dashboard for reviewing your recorded sessions and your
existing-vs-proposed model comparison, useful for your project demo/review. It reads
files this project already produces:

- `data/processed/features.csv` (from `src/feature_extraction.py`)
- `docs/results/comparison_summary.csv` (from `src/compare_models.py`)
- `models/train_summary.json` (from `src/train.py`)

It also has an optional tab to visually demo `low_light.py` (upload a frame, see
before/after).

Run it with:

```bash
pip install -r requirements.txt -r requirements-additions.txt
streamlit run dashboard_streamlit.py
```

It does not touch the webcam, the LSTM, or the alert system -- `src/realtime_infer.py`
still owns real-time detection exactly as before.

## 3. `src/realtime_infer_live.py` + dashboard "Live Monitor" tab

A new script, separate from `src/realtime_infer.py` (which is untouched and still
works with no logging if that's all you want). `realtime_infer_live.py` reuses the
exact same existing modules (`feature_extraction.py`, `model.py`, `utils.py`,
`alert_system.py`, `calibration.py` -- none of them modified) but additionally
appends one row per processed frame to `logs/live_features.csv`
(timestamp, EAR_left/right/avg, MAR, pitch/yaw/roll, alert level).

The dashboard's new **Live Monitor** tab tails that CSV and redraws EAR/MAR/pose/
status charts roughly once a second while the checkbox "Start live monitoring" is
ticked -- this is what gives you the "graphs updating in the browser while the
detector runs" experience, the same way the second reviewed project's
detector + Streamlit dashboard worked together.

To use it, run these in two separate terminals:

```bash
# Terminal 1: webcam + LSTM + alerts + live CSV logging
python src/realtime_infer_live.py --source 0 --checkpoint models/best_model.pt

# Terminal 2: dashboard
pip install -r requirements.txt -r requirements-additions.txt
streamlit run dashboard_streamlit.py
# -> open http://localhost:8501, go to the "Live Monitor" tab, check
#    "Start live monitoring"
```

## 4. `src/cross_dataset_eval.py` -- cross-dataset generalization test

A new, inference-only script. Does NOT modify `evaluate.py`, `dataset.py`, or
`model.py` -- it reuses their exact same windowing/personalization/normalization
functions so results are directly comparable to your internal test accuracy.

It runs your already-trained `models/best_model.pt` on features extracted from an
EXTERNAL dataset (e.g. UTA-RLDD/NTHU-DDD/YawDD) and reports accuracy/precision/
recall/F1, saving a row to `docs/results/cross_dataset_summary.csv` (a new file --
your existing `comparison_summary.csv` is untouched).

### How to use it

1. Get a small external dataset (5-10 videos is enough for a project-scope claim).
   UTA-RLDD is public and doesn't need a request form, unlike NTHU-DDD/YawDD.

2. Rename the external videos to match this project's existing naming convention
   (already expected by `feature_extraction.py`, unchanged):
   `<subject>_<label>_<session>.mp4`, where label is `alert`, `drowsy`, or
   `highdrowsy`. Put them in a NEW folder, e.g. `data/external_raw/` (your existing
   `data/raw/` is untouched).

3. Extract features using your EXISTING, unmodified script:

   ```bash
   python src/feature_extraction.py --video_dir data/external_raw \
       --out data/processed/external_features.csv
   ```

4. Run the new evaluation script:

   ```bash
   python src/cross_dataset_eval.py \
       --checkpoint models/best_model.pt \
       --features data/processed/external_features.csv \
       --dataset_name UTA-RLDD \
       --out docs/results/cross_dataset_summary.csv
   ```

5. Report whatever accuracy comes out, honestly, alongside your internal
   95.75% test accuracy -- a lower cross-dataset number is normal and expected,
   and is a legitimate, useful finding for your report/paper, not a failure.

## 5. `src/verify_failure_case.py` + `src/failure_case_demo.py` -- clear "existing fails, proposed succeeds" demonstration

Two new, additive-only files built for exactly this ask: a controlled scenario
where the existing (base-paper-style) approach demonstrably fails and the proposed
LSTM demonstrably succeeds.

**The specific, honest weakness targeted:** the base paper's decision rule
(re-implemented unmodified in `src/baseline_model.py: RuleBasedBaseline`) only
alarms if eyes are closed for N *consecutive* frames. **Rapid/frequent blinking**
(a real, literature-documented drowsiness sign -- increased blink rate / PERCLOS)
defeats this structurally: eyes can be closed 65-70% of a window and the rule still
says "Alert", because no single run ever reaches the consecutive-frame threshold.
This was verified analytically:

```
Rule-based (base-paper style) prediction on RAPID BLINKING pattern: Alert
(Eyes were closed 20 out of 30 frames = 67% of the time)
```

### Step 1 -- verify your LSTM's actual behavior on this pattern

```bash
python src/verify_failure_case.py --checkpoint models/best_model.pt
```

This feeds the same synthetic rapid-blinking window to both models and prints both
predictions side by side. Two outcomes:

- **LSTM already predicts Drowsy/Highly Drowsy** -- you have your demo case, go to
  Step 2.
- **LSTM also predicts Alert** -- expected if your recorded "drowsy" clips were all
  sustained long closures rather than rapid blinking. Fix (legitimate, not a
  shortcut -- rapid blinking is a real drowsiness cue that belongs in training data
  anyway): record 1-2 additional short clips of yourself blinking rapidly/
  repeatedly, label them following your existing convention (e.g.
  `bhavitha_drowsy_02.mp4`), then re-run your EXISTING, unmodified
  `feature_extraction.py` and `train.py`. Re-run `verify_failure_case.py` to
  confirm the LSTM now catches the pattern.

### Step 2 -- live side-by-side demo

```bash
python src/failure_case_demo.py --source 0 --checkpoint models/best_model.pt
```

Shows both "EXISTING (Rule-Based)" and "PROPOSED (LSTM)" predictions on screen at
the same time, live, from the same webcam feed. Blink rapidly/repeatedly for the
clearest effect -- a red "MISMATCH" banner appears whenever the two disagree,
which is exactly the "existing model fails, proposed model succeeds" moment your
guide asked for. Also logs every comparison to a new file,
`logs/failure_case_comparison.csv`, which you can screenshot/cite as evidence in
your report.

Note: `failure_case_demo.py` is generic -- it just compares whatever both models
output live, every window. You can reuse this SAME script for the head-nodding
demo below too (act out a slow head-nod instead of/alongside rapid blinking);
no separate live-demo script is needed.

## 6. `src/verify_headnod_case.py` -- a second, even more guaranteed failure case

Targets a different, arguably stronger weakness: the base paper **never computes
head pose at all** (confirmed in `feature_extraction.py`'s own docstring), and
`RuleBasedBaseline.predict_window()` (in `src/baseline_model.py`, unmodified) only
ever reads EAR (index 2) and MAR (index 3) -- columns 4/5/6 (pitch/yaw/roll) are
never touched. This means: a window where the driver's head droops/nods forward
while eyes stay open and there's no yawn is **guaranteed, by code inspection, not
just typical behaviour** -- to be classified "Alert" by the existing approach,
every single time.

```bash
python src/verify_headnod_case.py --checkpoint models/best_model.pt
```

Same two outcomes as `verify_failure_case.py`:
- LSTM already predicts Drowsy/Highly Drowsy -> use `failure_case_demo.py` live,
  acting out a slow head-nod (eyes open, no yawn) for your demo.
- LSTM also says Alert -> record 1-2 short clips of yourself slowly nodding off
  with eyes open (no exaggerated eye closing or yawning), label them following
  your existing convention (e.g. `bhavitha_drowsy_03.mp4`), and retrain using
  your existing, unmodified `feature_extraction.py` + `train.py`.

## 7. `src/verify_lowlight_case.py` -- low-light face-detection comparison

Targets the base paper's own self-admitted weakness: it states accuracy drops in
low light/glare because its detector struggles to see the driver at all. This
script runs your existing, unmodified `FeatureExtractor` (MediaPipe FaceMesh +
CLAHE) on a dark frame twice -- once as-is, once enhanced with the additive
`low_light.py` module (gamma + CLAHE + histogram equalization) -- and reports
whether a face was detected in each case.

```bash
# Using a real low-light photo:
python src/verify_lowlight_case.py --image path/to/dark_photo.jpg

# Or capture from webcam and simulate darkness:
python src/verify_lowlight_case.py --source 0 --simulate_darkness 0.15
```

Saves both the raw and enhanced frames to `logs/lowlight_raw.jpg` and
`logs/lowlight_enhanced.jpg` -- a clean before/after screenshot for your report,
showing "face detected: NO (raw) -> YES (enhanced)" when the contrast is clear.
