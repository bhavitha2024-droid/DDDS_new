"""
cross_dataset_eval.py

ADDITIVE ONLY -- new file. Does not modify evaluate.py, dataset.py, or model.py.

Runs your ALREADY-TRAINED checkpoint (models/best_model.pt) on a NEW features CSV
(e.g. extracted from an external public dataset like UTA-RLDD/NTHU-DDD/YawDD) to test
generalization beyond your own recorded sessions. This is inference-only:

    - No retraining happens.
    - Your existing model, features.csv, and comparison_summary.csv are untouched.
    - It reuses the exact same windowing/personalization/normalization functions
      from dataset.py that evaluate.py already uses, so results are directly
      comparable to your internal test accuracy.

Usage:
    python src/cross_dataset_eval.py \
        --checkpoint models/best_model.pt \
        --features data/processed/external_features.csv \
        --dataset_name UTA-RLDD \
        --out docs/results/cross_dataset_summary.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__)))
from dataset import (  # noqa: E402
    build_windows,
    normalize_features,
    personalize_dataframe,
    DrowsinessSequenceDataset,
)
from model import DrowsinessLSTM  # noqa: E402

CLASS_NAMES = ["Alert", "Drowsy", "Highly Drowsy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True,
                         help="Features CSV extracted from the EXTERNAL dataset "
                              "(e.g. data/processed/external_features.csv)")
    parser.add_argument("--dataset_name", required=True,
                         help="Label for this external dataset, e.g. 'UTA-RLDD'")
    parser.add_argument("--out", default="docs/results/cross_dataset_summary.csv",
                         help="Where to append/save the cross-dataset results row.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    df = pd.read_csv(args.features)
    if ckpt.get("personalize", True):
        # Same per-subject relative-baseline transform used at training time,
        # computed fresh on this new dataset's own subjects.
        df = personalize_dataframe(df)

    X, y = build_windows(df, cfg["window"]["sequence_length"], cfg["window"]["stride"])

    # IMPORTANT: reuse the mean/std saved at TRAINING time, do not recompute from
    # the external data -- this is what makes the comparison meaningful.
    mean = np.array(ckpt["feature_mean"])
    std = np.array(ckpt["feature_std"])
    X_norm, _, _ = normalize_features(X, mean, std)

    ds = DrowsinessSequenceDataset(X_norm, y)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"])

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

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(f"\n=== Cross-dataset evaluation: {args.dataset_name} ===")
    print(f"Windows evaluated: {len(all_labels)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1 (macro):        {f1:.4f}\n")
    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    row = pd.DataFrame([{
        "Dataset": args.dataset_name,
        "NumWindows": len(all_labels),
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }])
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        # replace any previous row for the same dataset_name, keep others
        existing = existing[existing["Dataset"] != args.dataset_name]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(args.out, index=False)
    print(f"Saved/updated results row in {args.out}")


if __name__ == "__main__":
    main()
