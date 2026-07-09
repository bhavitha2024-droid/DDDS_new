"""
record_session.py

Records YOUR OWN labeled drowsiness dataset from a webcam — no public dataset
(NTHU-DDD / YawDD / etc.) required. This directly answers the requirement that the
dataset must be self-collected: you (and, ideally, a few volunteer subjects for
generalization) sit in front of the camera and perform each state on command.

Each recording is saved with a filename that `feature_extraction.py` already knows
how to parse via `label_from_filename`:

    <subject>_alert_<session>.mp4        -> label 0 (Alert)
    <subject>_drowsy_<session>.mp4       -> label 1 (Drowsy)
    <subject>_highdrowsy_<session>.mp4   -> label 2 (Highly Drowsy)

Suggested protocol for a solid M.Tech-level custom dataset (documents well in your
report's "Dataset" chapter):
  - Record >= 3-5 subjects if possible (classmates/family) for generalization; if only
    yourself is available, record multiple sessions on different days/lighting.
  - Record >= 5-10 sessions per class per subject, each 20-40 seconds.
  - Vary lighting (daylight, indoor lamp, low light) to explicitly test the
    CLAHE-based robustness claim in this project.
  - "Alert": normal open eyes, occasional natural blinks, looking at road/screen.
  - "Drowsy": slow blinks, heavier eyelids, occasional yawns, mild head droop.
  - "Highly Drowsy": prolonged eye closure, repeated yawns, head nodding.

Usage:
    python record_session.py --subject bhavitha --label alert --duration 30
    python record_session.py --subject bhavitha --label drowsy --duration 30
    python record_session.py --subject bhavitha --label highdrowsy --duration 30
"""

import argparse
import os
import time

import cv2


LABELS = ["alert", "drowsy", "highdrowsy"]


def record(subject: str, label: str, duration: int, out_dir: str, camera_index: int = 0):
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")

    os.makedirs(out_dir, exist_ok=True)
    session_id = int(time.time())
    filename = f"{subject}_{label}_{session_id}.mp4"
    out_path = os.path.join(out_dir, filename)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Recording '{label}' for subject '{subject}' -> {out_path}")
    print("Get into position. Recording starts in 3 seconds...")
    for i in (3, 2, 1):
        print(i)
        time.sleep(1)

    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

        remaining = duration - (time.time() - start)
        display = frame.copy()
        cv2.putText(display, f"REC [{label.upper()}] {remaining:0.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Recording - press q to stop early", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Record a labeled session for your own dataset.")
    parser.add_argument("--subject", required=True, help="Subject/person identifier, e.g. 'bhavitha'.")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--duration", type=int, default=30, help="Seconds to record.")
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    record(args.subject, args.label, args.duration, args.out_dir, args.camera_index)


if __name__ == "__main__":
    main()
