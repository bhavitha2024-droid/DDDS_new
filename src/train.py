"""
train.py

Trains the DrowsinessLSTM on windowed sequences built from the feature CSV
(output of feature_extraction.py). Saves the best checkpoint (by validation
accuracy) plus the feature normalization stats, both required at inference time.

Usage:
    python train.py --features data/processed/features.csv --epochs 40 --window 30
"""

import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from dataset import DrowsinessSequenceDataset, load_and_prepare  # noqa: E402
from model import DrowsinessLSTM  # noqa: E402


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features CSV.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--window", type=int, default=None, help="Override sequence_length.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_personalize", action="store_true",
                         help="Disable subject-relative personalization (use raw features).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.window:
        cfg["window"]["sequence_length"] = args.window
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    personalize = cfg.get("personalization", {}).get("enabled", True) and not args.no_personalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | personalization: {personalize}")

    X, y, mean, std = load_and_prepare(
        args.features,
        cfg["window"]["sequence_length"],
        cfg["window"]["stride"],
        personalize=personalize,
    )
    # -------- Save Driver Profiles --------
    df = pd.read_csv(args.features)

    os.makedirs("models/driver_profiles", exist_ok=True)

    feature_cols = [
        "EAR_left",
        "EAR_right",
        "EAR_avg",
        "MAR",
        "pitch",
        "yaw",
        "roll"
    ]

    for subject in df["subject"].unique():
        profile = {}

        subject_df = df[df["subject"] == subject]

        for col in feature_cols:
            profile[col] = {
                "mean": float(subject_df[col].mean()),
                "std": float(subject_df[col].std())
            }

        with open(f"models/driver_profiles/{subject}.json", "w") as f:
            json.dump(profile, f, indent=4)

    print("Driver profiles saved.")
    print(f"Built {len(X)} windows. Class distribution: {np.bincount(y)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg["train"]["val_split"] + cfg["train"]["test_split"],
        stratify=y, random_state=42,
    )
    rel_test = cfg["train"]["test_split"] / (
        cfg["train"]["val_split"] + cfg["train"]["test_split"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=42
    )

    train_ds = DrowsinessSequenceDataset(X_train, y_train)
    val_ds = DrowsinessSequenceDataset(X_val, y_val)
    test_ds = DrowsinessSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])

    model = DrowsinessLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": cfg,
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "personalize": personalize,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"  -> New best model saved to {ckpt_path} (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    # Save training history
    history = pd.DataFrame({
        "Epoch": range(1, len(train_losses) + 1),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train Accuracy": train_accs,
        "Validation Accuracy": val_accs,
    })

    history.to_csv(
        os.path.join(cfg["train"]["checkpoint_dir"], "training_history.csv"),
        index=False
    )

    # Loss graph
    plt.figure(figsize=(8,5))
    plt.plot(history["Epoch"], history["Train Loss"], label="Training Loss")
    plt.plot(history["Epoch"], history["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg["train"]["checkpoint_dir"], "loss_curve.png"))
    plt.close()

    # Accuracy graph
    plt.figure(figsize=(8,5))
    plt.plot(history["Epoch"], history["Train Accuracy"], label="Training Accuracy")
    plt.plot(history["Epoch"], history["Validation Accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg["train"]["checkpoint_dir"], "accuracy_curve.png"))
    plt.close()

    print("Graphs saved successfully.")

    with open(os.path.join(cfg["train"]["checkpoint_dir"], "train_summary.json"), "w") as f:
        json.dump(
            {"best_val_acc": best_val_acc, "test_acc": test_acc, "num_windows": len(X)}, f, indent=2
        )


if __name__ == "__main__":
    main()
