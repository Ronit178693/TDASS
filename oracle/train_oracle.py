"""
train_oracle.py — B5: Oracle Training Pipeline
================================================
Trains both the LSTM action predictor and the intent classifier
on the enhanced synthetic dataset.

Features:
  - Train/val split with stratified sampling
  - Loss curves saved as matplotlib plots
  - Model checkpointing (best val accuracy)
  - Classification report on validation set
  - Progress bars via tqdm (falls back to print)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.data_loader import (
    get_dataloaders, POSTURE_CLASSES, ACTION_CLASSES, FEATURE_COLS
)
from oracle.lstm_predictor import LSTMPredictor
from oracle.intent_classifier import IntentClassifier

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "csv_path":      "Element_Logic/Synthetic_dataset/battle_data.csv",
    "window_size":   10,
    "batch_size":    256,
    "val_split":     0.2,
    "epochs":        30,
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.3,
    "checkpoint_dir": "oracle/checkpoints",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}

ACTION_NAMES = ["Stay", "Up", "Down", "Left", "Right", "Ranged"]


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def train_epoch_posture(model, loader, criterion, optimizer, device):
    """Train one epoch for posture classifier."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, _, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_idx=1):
    """Evaluate model. target_idx: 1=action, 2=posture in batch tuple."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        X_batch = batch[0].to(device)
        y_batch = batch[target_idx].to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────
def train_oracle(config=None):
    """Train both action predictor and intent classifier."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Create checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    print("╔═══════════════════════════════════════════════════╗")
    print("║  TDSS Oracle — Training Pipeline                 ║")
    print(f"║  Device: {cfg['device']:<42}║")
    print("╚═══════════════════════════════════════════════════╝")

    # ── Load data ──
    print("\n📊 Loading and preprocessing data...")
    train_loader, val_loader, posture_encoder, num_features = get_dataloaders(
        cfg["csv_path"], cfg["window_size"], cfg["batch_size"], cfg["val_split"]
    )

    # ═══════════════════════════════════════════
    # 1. Train LSTM Action Predictor
    # ═══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  Phase B2: LSTM Action Predictor")
    print("=" * 55)

    action_model = LSTMPredictor(
        input_size=num_features,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=ACTION_CLASSES,
        dropout=cfg["dropout"],
    ).to(cfg["device"])

    print(f"  Parameters: {sum(p.numel() for p in action_model.parameters()):,}")

    action_criterion = nn.CrossEntropyLoss()
    action_optimizer = optim.Adam(
        action_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    action_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        action_optimizer, mode="min", factor=0.5, patience=3
    )

    best_action_acc = 0.0
    action_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            action_model, train_loader, action_criterion,
            action_optimizer, cfg["device"]
        )
        val_loss, val_acc, _, _ = evaluate(
            action_model, val_loader, action_criterion, cfg["device"], target_idx=1
        )
        action_scheduler.step(val_loss)
        dt = time.time() - t0

        action_history["train_loss"].append(train_loss)
        action_history["val_loss"].append(val_loss)
        action_history["train_acc"].append(train_acc)
        action_history["val_acc"].append(val_acc)

        marker = " ★" if val_acc > best_action_acc else ""
        print(f"  Epoch {epoch:>2}/{cfg['epochs']}  "
              f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"Acc: {train_acc:.3f}/{val_acc:.3f}  "
              f"[{dt:.1f}s]{marker}")

        if val_acc > best_action_acc:
            best_action_acc = val_acc
            torch.save(action_model.state_dict(),
                       os.path.join(cfg["checkpoint_dir"], "action_predictor_best.pt"))

    # Final evaluation
    _, _, a_preds, a_labels = evaluate(
        action_model, val_loader, action_criterion, cfg["device"], target_idx=1
    )
    print(f"\n  Best Action Accuracy: {best_action_acc:.4f}")
    print("\n  Action Classification Report:")
    print(classification_report(a_labels, a_preds, target_names=ACTION_NAMES,
                                labels=range(ACTION_CLASSES), zero_division=0))

    # ═══════════════════════════════════════════
    # 2. Train Intent Classifier
    # ═══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  Phase B3: Intent Classifier (Posture)")
    print("=" * 55)

    intent_model = IntentClassifier(
        input_size=num_features,
        hidden_size=96,
        num_layers=cfg["num_layers"],
        num_postures=len(POSTURE_CLASSES),
        dropout=cfg["dropout"],
    ).to(cfg["device"])

    print(f"  Parameters: {sum(p.numel() for p in intent_model.parameters()):,}")

    intent_criterion = nn.CrossEntropyLoss()
    intent_optimizer = optim.Adam(
        intent_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    intent_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        intent_optimizer, mode="min", factor=0.5, patience=3
    )

    best_intent_acc = 0.0
    intent_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch_posture(
            intent_model, train_loader, intent_criterion,
            intent_optimizer, cfg["device"]
        )
        val_loss, val_acc, _, _ = evaluate(
            intent_model, val_loader, intent_criterion, cfg["device"], target_idx=2
        )
        intent_scheduler.step(val_loss)
        dt = time.time() - t0

        intent_history["train_loss"].append(train_loss)
        intent_history["val_loss"].append(val_loss)
        intent_history["train_acc"].append(train_acc)
        intent_history["val_acc"].append(val_acc)

        marker = " ★" if val_acc > best_intent_acc else ""
        print(f"  Epoch {epoch:>2}/{cfg['epochs']}  "
              f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"Acc: {train_acc:.3f}/{val_acc:.3f}  "
              f"[{dt:.1f}s]{marker}")

        if val_acc > best_intent_acc:
            best_intent_acc = val_acc
            torch.save(intent_model.state_dict(),
                       os.path.join(cfg["checkpoint_dir"], "intent_classifier_best.pt"))

    # Final evaluation
    _, _, p_preds, p_labels = evaluate(
        intent_model, val_loader, intent_criterion, cfg["device"], target_idx=2
    )
    print(f"\n  Best Intent Accuracy: {best_intent_acc:.4f}")
    print("\n  Posture Classification Report:")
    print(classification_report(p_labels, p_preds, target_names=POSTURE_CLASSES,
                                labels=range(len(POSTURE_CLASSES)), zero_division=0))

    # ── Save loss curves ──
    _save_loss_curves(action_history, intent_history, cfg["checkpoint_dir"])

    # ── Save training summary ──
    summary = {
        "best_action_accuracy": best_action_acc,
        "best_intent_accuracy": best_intent_acc,
        "epochs": cfg["epochs"],
        "window_size": cfg["window_size"],
        "num_features": num_features,
        "hidden_size": cfg["hidden_size"],
        "device": cfg["device"],
    }
    with open(os.path.join(cfg["checkpoint_dir"], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("  Oracle Training Complete!")
    print(f"  Action predictor:  {best_action_acc:.4f} accuracy")
    print(f"  Intent classifier: {best_intent_acc:.4f} accuracy")
    print(f"  Checkpoints saved: {cfg['checkpoint_dir']}/")
    print("=" * 55)

    return action_model, intent_model


def _save_loss_curves(action_hist, intent_hist, output_dir):
    """Save matplotlib loss/accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("TDSS Oracle — Training Curves", fontsize=14, fontweight="bold")

        # Action loss
        axes[0, 0].plot(action_hist["train_loss"], label="Train", color="#3498db")
        axes[0, 0].plot(action_hist["val_loss"], label="Val", color="#e74c3c")
        axes[0, 0].set_title("Action Predictor — Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Action accuracy
        axes[0, 1].plot(action_hist["train_acc"], label="Train", color="#3498db")
        axes[0, 1].plot(action_hist["val_acc"], label="Val", color="#e74c3c")
        axes[0, 1].set_title("Action Predictor — Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Intent loss
        axes[1, 0].plot(intent_hist["train_loss"], label="Train", color="#2ecc71")
        axes[1, 0].plot(intent_hist["val_loss"], label="Val", color="#e67e22")
        axes[1, 0].set_title("Intent Classifier — Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Intent accuracy
        axes[1, 1].plot(intent_hist["train_acc"], label="Train", color="#2ecc71")
        axes[1, 1].plot(intent_hist["val_acc"], label="Val", color="#e67e22")
        axes[1, 1].set_title("Intent Classifier — Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\n  📈 Loss curves saved: {path}")
    except Exception as e:
        print(f"\n  ⚠ Could not save loss curves: {e}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_oracle()
