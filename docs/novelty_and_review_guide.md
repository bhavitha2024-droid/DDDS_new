# Project Review Guide: Level Assessment, Novelty, and Viva Preparation

## 1. Is this M.Tech-major-project level and research-oriented?

**Yes, with the honest caveats below.** Here's what makes it sit at that level, and what you
need to genuinely do (not just claim) to defend it as such.

**What already puts it at M.Tech level:**
- It doesn't just reproduce the base paper — it identifies specific, citable limitations in a
  published IJACSA 2021 paper and proposes an architectural response to each one (see
  `docs/comparison_with_base_paper.md`). That "gap identification → targeted solution" structure
  is exactly what a review committee looks for.
- It reframes the problem correctly: drowsiness is a *temporal/behavioral* phenomenon, and the
  base paper's own literature review cites PERCLOS (percentage of eye closure over time) as the
  established gold-standard cue — yet its own method reduces this to one fixed frame-count rule.
  Recognizing and fixing that gap (base paper's own stated future work, in fact) is a legitimate,
  defensible research contribution, not just an engineering exercise.
- It includes a genuine **ablation study** (rule-based vs. non-temporal ML vs. LSTM, all fed the
  *same* features) rather than a single "look, deep learning is better" comparison. That isolates
  *why* your model works, which is what separates a research project from a course project.
- It has a full experimental pipeline: your own IRB-free self-collected dataset, defined train/val/
  test protocol, standard metrics (accuracy/precision/recall/F1/confusion matrix), and reproducible
  code — reviewers can ask "show me the numbers" and you can.

**What you must actually do for this claim to hold at review time (this is the honest part):**
- **Collect a real dataset** using `src/record_session.py` — multiple sessions, ideally multiple
  subjects, varied lighting. A single 2-minute clip per class will not survive scrutiny; aim for
  the protocol in that script's docstring.
- **Run `compare_models.py` and report the real numbers it produces.** I cannot and will not
  pre-fill an accuracy number for you — a fabricated number is the single fastest way to lose
  credibility in a viva if a panelist asks you to explain a result you can't reproduce. Whatever
  the script prints is what you present.
- **Report both successes and limitations honestly** (e.g., accuracy on a small self-collected
  dataset, subject-independent generalization, class imbalance if your "highly drowsy" clips are
  fewer). Panels respond well to a "Conclusion & Limitations" section that mirrors the honesty of
  the base paper's own Section VIII — it signals maturity, not weakness.

## 2. Novelty points you can state with confidence

Use these as your "what is novel here" answer — each is something the base paper explicitly does
NOT do:

1. **Temporal sequence modeling (LSTM) replacing a fixed frame-count heuristic.** The base paper's
   drowsiness decision is "eyes closed for exactly 10 consecutive frames." This project replaces
   that hard-coded rule with a learned model over a sliding window, generalizing PERCLOS-style
   reasoning instead of hand-picking one threshold.
2. **Multi-cue temporal fusion.** Eye closure, yawning (MAR), and head pose (pitch/yaw/roll via
   solvePnP) are fused into a single per-timestep feature vector for the LSTM. The base paper
   explicitly lists incorporating yawning into the drowsiness decision as *future work it did not
   complete* — you completed it.
3. **Graded, cooldown-managed alerting** (Alert / Drowsy / Highly Drowsy) instead of a single
   binary alarm — an early-warning capability the base paper does not have.
4. **A controlled ablation isolating the value of temporal order.** Rule-based and non-temporal-ML
   baselines see the *exact same* aggregated information as the LSTM; only the LSTM sees frame
   order. This directly demonstrates (with your own numbers) that sequential modeling — not just
   "more features" — is what improves detection.
5. **Explicit low-light robustness design** (CLAHE + geometric ratio/angle features instead of raw
   pixel classification) addressing the base paper's self-reported weak point.
6. **Self-collected, ethically simpler dataset.** Because features are anonymized geometric
   ratios/angles (not raw face images stored for training), your dataset avoids some of the
   licensing/privacy overhead of large public face datasets while still being fully your own.

## 3. Likely viva questions and how to answer them

**Q: Why LSTM and not a CNN-LSTM directly on raw video frames?**
A: LSTM on hand-engineered EAR/MAR/head-pose features is far lighter (real-time on commodity
hardware, matching the base paper's own stated "no expensive hardware" goal), interpretable
(you can literally show which cue triggered an alert), and — critically for a self-collected
dataset — needs far less data to train reliably than an end-to-end CNN-LSTM would.

**Q: How do you know the improvement comes from the LSTM and not just from extra features?**
A: Point to the ablation in `compare_models.py` — the non-temporal baseline gets the *same*
window of features (as aggregated statistics), so any accuracy gap between it and the LSTM
isolates the contribution of sequence order specifically.

**Q: How large is your dataset, and does that limit your claims?**
A: Answer honestly with your actual numbers of subjects/sessions/frames. Frame the accuracy as
"promising on a controlled self-collected dataset" and name subject-independent generalization
and dataset scale as explicit future work — this is a legitimate, standard limitation to state
and does not weaken your novelty claims.

**Q: What's the real-world deployment story?**
A: `src/realtime_infer.py` is a working live demo — feature extraction + LSTM + alerting running
on a webcam feed in real time, which you can literally show at your review.

## 4. What NOT to say

Do not claim a specific accuracy figure until you have actually run `compare_models.py` on your
own recorded data and gotten that number back. Do not claim comparison against the base paper's
own reported 98% mAP as if it were the same metric — that number is a PASCAL VOC object-detection
mAP on a *different task* (frame-level open/closed/yawn/no-yawn detection), not a drowsiness-state
classification accuracy. Compare against your own rule-based/non-temporal baselines instead, since
that is the fair, apples-to-apples comparison this project is built to produce.
