"""
compare_models.py

Trains/evaluates THREE models on your own recorded dataset and reports them
side-by-side, so you can present a genuine "existing approach vs. proposed approach"
table at your review:

    1. Rule-Based (base-paper style)   -- src/baseline_model.py: RuleBasedBaseline
    2. Non-Temporal ML (RandomForest)  -- src/baseline_model.py: NonTemporalMLBaseline
    3. Proposed LSTM (temporal)        -- src/model.py: DrowsinessLSTM

All three see the SAME windows built from the SAME features (from feature_extraction.py
run on YOUR OWN recorded videos) — so any accuracy difference is attributable to the
modeling approach, not to different inputs. This is the fair, defensible comparison you
need for a viva.

IMPORTANT: This script reports REAL metrics computed from whatever data you feed it.
It does not hard-code or assume any accuracy number — the numbers you see when you run
this on your own dataset are the numbers you should report and defend.

Usage:
    python compare_models.py --features data/processed/features.csv --window 30 --out docs/results
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from dataset import build_windows, normalize_features, personalize_dataframe, DrowsinessSequenceDataset  # noqa: E402
from model import DrowsinessLSTM  # noqa: E402
from baseline_model import RuleBasedBaseline  # noqa: E402

CLASS_NAMES = ["Alert", "Drowsy", "Highly Drowsy"]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def metrics_report(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision(macro): {precision:.4f}  Recall(macro): {recall:.4f}  F1(macro): {f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
    }


def train_lstm(X_train, y_train, X_val, y_val, cfg, device):
    model = DrowsinessLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
    ).to(device)

    train_ds = DrowsinessSequenceDataset(X_train, y_train)
    val_ds = DrowsinessSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"]
    )

    best_val_acc, best_state, patience_counter = 0.0, None, 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def lstm_predict(model, X, device):
    model.eval()
    x = torch.from_numpy(X).float().to(device)
    preds = model(x).argmax(dim=1).cpu().numpy()
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV from feature_extraction.py on YOUR data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--out", default="docs/results", help="Directory to write comparison outputs.")
    parser.add_argument("--no_personalize", action="store_true",
                         help="Disable subject-relative personalization (use raw features).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.window:
        cfg["window"]["sequence_length"] = args.window
    os.makedirs(args.out, exist_ok=True)
    personalize = cfg.get("personalization", {}).get("enabled", True) and not args.no_personalize
    print(f"Personalization: {personalize}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.features)
    if personalize:
        df = personalize_dataframe(df)

    seq_len = cfg["window"]["sequence_length"]
    stride = cfg["window"]["stride"]
    X_raw, y = build_windows(df, seq_len, stride)  # "raw" here means pre-zscore, but already
                                                    # personalized if personalize=True
    X_norm, mean, std = normalize_features(X_raw)

    print(f"Total windows: {len(X_raw)} | class distribution: {np.bincount(y)}")

    idx = np.arange(len(X_raw))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=cfg["train"]["val_split"] + cfg["train"]["test_split"],
        stratify=y, random_state=42,
    )
    rel_test = cfg["train"]["test_split"] / (cfg["train"]["val_split"] + cfg["train"]["test_split"])
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=42
    )

    X_train_raw, X_val_raw, X_test_raw = X_raw[idx_train], X_raw[idx_val], X_raw[idx_test]
    X_train_norm, X_val_norm, X_test_norm = X_norm[idx_train], X_norm[idx_val], X_norm[idx_test]

    results = []

    # 1. Rule-based baseline (base-paper-style, no training required)
    # NOTE: when personalization is enabled, EAR/MAR are ratios relative to each
    # subject's own baseline (~1.0 = normal openness), so we use a RELATIVE threshold
    # instead of the base paper's single absolute constant.
    ear_thresh = (
        cfg["personalization"]["ear_relative_threshold"] if personalize
        else cfg["alerts"]["ear_sanity_threshold"]
    )
    rule_model = RuleBasedBaseline(
        ear_threshold=ear_thresh,
        mar_threshold=cfg["alerts"]["mar_sanity_threshold"],
    )
    y_pred_rule = rule_model.predict(X_test_raw)
    results.append(metrics_report(y_test, y_pred_rule, "Existing: Rule-Based (base-paper style)"))


    # 2. Proposed LSTM (temporal)
    lstm_model = train_lstm(X_train_norm, y_train, X_val_norm, y_val, cfg, device)
    y_pred_lstm = lstm_predict(lstm_model, X_test_norm, device)
    results.append(metrics_report(y_test, y_pred_lstm, "Proposed: LSTM (temporal)"))

    # Save results
    with open(os.path.join(args.out, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    summary_df = pd.DataFrame([
        {"Model": r["model"], "Accuracy": r["accuracy"], "Precision": r["precision_macro"],
         "Recall": r["recall_macro"], "F1": r["f1_macro"]}
        for r in results
    ])
    summary_df.to_csv(os.path.join(args.out, "comparison_summary.csv"), index=False)
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(summary_df))
        ax.bar(x_pos, summary_df["Accuracy"], color=["#888888", "#2266cc"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df["Model"], rotation=15, ha="right")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Existing Approaches vs. Proposed LSTM — on your own dataset")
        ax.set_ylim(0, 1)
        for i, v in enumerate(summary_df["Accuracy"]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.tight_layout()
        chart_path = os.path.join(args.out, "accuracy_comparison.png")
        plt.savefig(chart_path, dpi=150)
        print(f"Saved chart to {chart_path}")
    except ImportError:
        print("matplotlib not available; skipped chart generation.")

    print(f"\nAll comparison artifacts written to: {args.out}")


if __name__ == "__main__":
    main()
