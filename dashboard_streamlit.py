"""
dashboard_streamlit.py

Standalone review dashboard, added alongside the existing project (ADDITIVE ONLY --
no existing file in this repository is modified or depended on being changed).

Ported/adapted from the second reviewed project's Streamlit dashboard idea, but wired
to THIS repo's actual outputs:

    - data/processed/features.csv          (per-frame EAR/MAR/pose, from feature_extraction.py)
    - docs/results/comparison_summary.csv  (rule-based vs RandomForest vs LSTM metrics,
                                             from compare_models.py)
    - models/train_summary.json            (best_val_acc / test_acc / num_windows,
                                             from train.py)

It also includes an optional "Low-light preview" tab that uses the new low_light.py
module (also just added) so you can visually demonstrate the gamma+CLAHE+histogram-
equalization enhancement on a sample frame during your project review/demo.

This dashboard is READ-ONLY / offline: it does not run the webcam, the LSTM, or the
alert system -- realtime_infer.py still owns that. This is purely for reviewing your
collected data and your existing/proposed model comparison in a nicer UI for your demo.

Run:
    streamlit run dashboard_streamlit.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

BASE_DIR = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "features.csv")
COMPARISON_CSV = os.path.join(BASE_DIR, "docs", "results", "comparison_summary.csv")
TRAIN_SUMMARY_JSON = os.path.join(BASE_DIR, "models", "train_summary.json")

LABEL_NAMES = {0: "Alert", 1: "Drowsy", 2: "Highly Drowsy"}

st.set_page_config(page_title="Drowsiness Detection - Review Dashboard", layout="wide")
st.title("Intelligent Driver Drowsiness Detection - Review Dashboard")
st.success(
"Real-Time Intelligent Driver Drowsiness Detection using Personalized LSTM"
)
st.markdown("""
### Intelligent Driver Monitoring Dashboard

This dashboard visualizes

• Driver State

• EAR

• MAR

• Head Pose

• LSTM Prediction

• Existing vs Proposed Comparison

• Low-Light Enhancement
""")

LIVE_LOG_CSV = os.path.join(BASE_DIR, "logs", "live_features.csv")

tab_live, tab_data, tab_results, tab_lowlight = st.tabs(
    ["Live Monitor", "Recorded Sessions (EAR/MAR/Pose)", "Existing vs Proposed Model", "Low-Light Preview"]
)

# ---------------------------------------------------------------------------
# Tab 0: live monitor (tails logs/live_features.csv written by
# src/realtime_infer_live.py while it's running against the webcam)
# ---------------------------------------------------------------------------
with tab_live:
    st.write(
        "Shows live EAR / MAR / head-pose / alert-level while "
        "`python src/realtime_infer_live.py --source 0 --checkpoint models/best_model.pt` "
        "is running in another terminal. This does not run the webcam or the LSTM "
        "itself -- it only reads the CSV that script writes to, the same way the "
        "second reviewed project's detector + Streamlit dashboard worked together."
    )

    max_points = st.slider("Number of recent frames to plot", 50, 1000, 300, step=50)
    run_live = st.checkbox("Start live monitoring", value=False)

    placeholder_status = st.empty()
    placeholder_ear = st.empty()
    placeholder_mar = st.empty()
    placeholder_pose = st.empty()

    if run_live:
        if not os.path.exists(LIVE_LOG_CSV):
            st.warning(
                f"`{LIVE_LOG_CSV}` does not exist yet. Start "
                "`python src/realtime_infer_live.py --source 0 --checkpoint "
                "models/best_model.pt` first, then re-check this box."
            )
        else:
            # Bounded live loop: refreshes in place for up to ~10 minutes per
            # check of the box, then stops so the tab doesn't run forever.
            for _ in range(600):
                if not os.path.exists(LIVE_LOG_CSV):
                    break
                try:
                    live_df = pd.read_csv(LIVE_LOG_CSV).tail(max_points)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    time.sleep(1.0)
                    continue
                if live_df.empty:
                    placeholder_status.info("Waiting for frames... (no face detected yet?)")
                else:
                    latest = live_df.iloc[-1]
                    st.subheader("Current Driver Status")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Prediction",
                            latest["level_name"]
                        )
                    with c2:
                        st.metric(
                            "Confidence",
                            f"{latest['confidence']*100:.2f}%"
                        )
                    with c3:
                        st.metric(
                            "FPS",
                            f"{latest['fps']:.1f}"
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "EAR Avg",
                            f"{latest['EAR_avg']:.3f}"
                        )
                        c2.metric(
                            "MAR",
                            f"{latest['MAR']:.3f}"
                        )
                        c3.metric(
                            "Frames",
                            len(live_df)
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "Pitch",
                            f"{latest['pitch']:.2f}"
                        )
                        c2.metric(
                            "Yaw",
                            f"{latest['yaw']:.2f}"
                        )
                        c3.metric(
                            "Roll",
                            f"{latest['roll']:.2f}"
                        )
                        if latest["level"] == 0:
                            st.success("🟢 Driver is ALERT")

                        elif latest["level"] == 1:
                            st.warning("🟠 Driver is DROWSY")

                        else:
                            st.error("🔴 Driver is HIGHLY DROWSY")
                    live_df = live_df.reset_index(drop=True)
                    placeholder_ear.line_chart(live_df[["EAR_left", "EAR_right", "EAR_avg"]])
                    placeholder_mar.line_chart(live_df[["MAR"]])
                    placeholder_pose.line_chart(live_df[["pitch", "yaw", "roll"]])
                    st.subheader("Recent Predictions")
                    st.dataframe(
                        live_df[
                            [
                                "level_name",
                                "confidence"
                            ]
                        ].tail(10),
                        use_container_width=True
                    )
                time.sleep(1.0)
    else:
        st.info("Check \"Start live monitoring\" above while the live script is running.")


# ---------------------------------------------------------------------------
# Tab 1: recorded feature data
# ---------------------------------------------------------------------------
with tab_data:
    if not os.path.exists(FEATURES_CSV):
        st.warning(
            f"No features file found at `{FEATURES_CSV}` yet. Run "
            "`python src/feature_extraction.py --video_dir data/raw "
            "--out data/processed/features.csv` first."
        )
    else:
        df = pd.read_csv(FEATURES_CSV)
        st.write(f"Loaded **{len(df)}** frames from **{df['video'].nunique()}** recorded video(s).")

        col1, col2 = st.columns(2)
        with col1:
            subjects = sorted(df["subject"].unique().tolist())
            subject = st.selectbox("Subject", subjects)
        with col2:
            videos = sorted(df[df["subject"] == subject]["video"].unique().tolist())
            video = st.selectbox("Video / session", videos)

        session_df = df[(df["subject"] == subject) & (df["video"] == video)].sort_values("frame_idx")

        if session_df.empty:
            st.info("No frames for this selection.")
        else:
            label_val = int(session_df["label"].iloc[0]) if "label" in session_df.columns else None
            if label_val is not None:
                st.metric("Session label", LABEL_NAMES.get(label_val, str(label_val)))

            st.subheader("Eye Aspect Ratio (EAR) over time")
            st.line_chart(session_df.set_index("frame_idx")[["EAR_left", "EAR_right", "EAR_avg"]])

            st.subheader("Mouth Aspect Ratio (MAR) over time")
            st.line_chart(session_df.set_index("frame_idx")[["MAR"]])

            st.subheader("Head pose over time")
            st.line_chart(session_df.set_index("frame_idx")[["pitch", "yaw", "roll"]])

        st.subheader("Class balance across all recorded sessions")
        if "label" in df.columns:
            counts = df.drop_duplicates("video")["label"].map(LABEL_NAMES).value_counts()
            st.bar_chart(counts)

# ---------------------------------------------------------------------------
# Tab 2: existing (rule-based / RF) vs proposed (LSTM) comparison
# ---------------------------------------------------------------------------
with tab_results:
    if not os.path.exists(COMPARISON_CSV):
        st.warning(
            f"No comparison results found at `{COMPARISON_CSV}` yet. Run "
            "`python src/compare_models.py --features data/processed/features.csv "
            "--window 30 --out docs/results` first."
        )
    else:
        comp_df = pd.read_csv(COMPARISON_CSV)
        st.write("Metrics computed by `src/compare_models.py` on your own recorded data:")
        st.dataframe(comp_df, use_container_width=True)

        metric = st.selectbox("Metric to chart", ["Accuracy", "Precision", "Recall", "F1"])
        chart_df = comp_df.set_index("Model")[[metric]]
        st.bar_chart(chart_df)

    if os.path.exists(TRAIN_SUMMARY_JSON):
        with open(TRAIN_SUMMARY_JSON) as f:
            summary = json.load(f)
        st.subheader("LSTM training summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best validation accuracy", f"{summary.get('best_val_acc', 0):.4f}")
        c2.metric("Test accuracy", f"{summary.get('test_acc', 0):.4f}")
        c3.metric("Number of windows", summary.get("num_windows", "-"))

# ---------------------------------------------------------------------------
# Tab 3: low-light preview (uses the newly added low_light.py, optional demo)
# ---------------------------------------------------------------------------
with tab_lowlight:
    st.write(
        "Optional demo of the additive `src/low_light.py` module "
        "(gamma correction -> CLAHE -> histogram equalization) on an uploaded frame. "
        "This module is not wired into the main pipeline automatically -- it is "
        "available for you to import and use in `preprocessing.py` / "
        "`feature_extraction.py` / `realtime_infer.py` yourself if you want the "
        "stronger 3-stage enhancement for very low-light conditions."
    )
    uploaded = st.file_uploader("Upload a sample frame (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        import cv2
        from src.low_light import enhance_low_light

        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        enhanced_bgr = enhance_low_light(frame_bgr)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col2:
            st.image(
                cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB),
                caption="Enhanced (gamma + CLAHE + hist-eq)",
                use_container_width=True,
            )
