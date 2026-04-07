# Training script docstring.
"""
# Source file identity.
train_oracle.py — B5: Oracle Training Pipeline
# Section header.
================================================
# Core functionality overview.
Trains both the LSTM action predictor and the intent classifier
# Data description.
on the enhanced synthetic dataset.
# Empty line.

# List of key training features.
Features:
# Data splitting strategy.
  - Train/val split with stratified sampling
# Visualization feature.
  - Loss curves saved as matplotlib plots
# Checkpointing feature.
  - Model checkpointing (best val accuracy)
# Reporting feature.
  - Classification report on validation set
# UI notification feature.
  - Progress bars via tqdm (falls back to print)
# Closing docstring.
"""
# Empty line.

# Standards OS tools.
import os
# System standard library.
import sys
# Time management.
import time
# Data format management.
import json
# Numerical array management.
import numpy as np
# Deep learning core.
import torch
# Neural network components.
import torch.nn as nn
# Optimization algorithms.
import torch.optim as optim
# Accuracy metrics and evaluation reporting.
from sklearn.metrics import classification_report, confusion_matrix
# Empty line.

# Logic to find the root folder for relative imports.
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Empty line.

# Import project-specific data constants.
from oracle.data_loader import (
    get_dataloaders, POSTURE_CLASSES, ACTION_CLASSES, FEATURE_COLS
)
# Internal model architectures.
from oracle.lstm_predictor import LSTMPredictor
from oracle.intent_classifier import IntentClassifier
# Empty line.

# Logic for dynamic progress bar terminal UI.
try:
# Attempt to load specialized progress bar.
    from tqdm import tqdm
# Fallback for systems without tqdm.
except ImportError:
# Mock function if loading fails.
    def tqdm(iterable, **kwargs):
# Return standard iterator.
        return iterable
# Empty line.
# Empty line.


# Decorative separator line.
# ──────────────────────────────────────────────
# Training configuration header.
# Config
# Decorative separator line.
# ──────────────────────────────────────────────
# Global map for training hyper-parameters.
DEFAULT_CONFIG = {
# Path to source data.
    "csv_path":      "Element_Logic/Synthetic_dataset/battle_data.csv",
# Temporal window depth.
    "window_size":   10,
# Data training chunk size.
    "batch_size":    256,
# Holdout percentage for testing.
    "val_split":     0.2,
# Total iterations through data.
    "epochs":        30,
# Learning rate speed.
    "lr":            1e-3,
# L2 regularization weight.
    "weight_decay":  1e-5,
# Width of the hidden layers.
    "hidden_size":   128,
# Depth of the stacked layers.
    "num_layers":    2,
# Probability of dropping neurons to prevent bias.
    "dropout":       0.3,
# Target folder for saving trained models.
    "checkpoint_dir": "oracle/checkpoints",
# Logic choosing between GPU or CPU.
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
# Closing bracket.
}
# Empty line.

# Descriptive names for action codes 0-5.
ACTION_NAMES = ["Stay", "Up", "Down", "Left", "Right", "Ranged"]
# Empty line.
# Empty line.


# Decorative separator line.
# ──────────────────────────────────────────────
# Function definition section for training.
# Training loop
# Decorative separator line.
# ──────────────────────────────────────────────
# Internal function to run a single training step.
def train_epoch(model, loader, criterion, optimizer, device):
# Docstring block.
    """Train one epoch. Returns avg loss and accuracy."""
# Set model to learning mode.
    model.train()
# Initialize counters.
    total_loss, correct, total = 0.0, 0, 0
# Empty line.

# Iterate over data batches.
    for X_batch, y_batch, _ in loader:
# Push input to target hardware.
        X_batch = X_batch.to(device)
# Push label to target hardware.
        y_batch = y_batch.to(device)
# Empty line.

# Wipe previous gradients.
        optimizer.zero_grad()
# Pass data through neural network.
        logits = model(X_batch)
# Calculate the error compared to labels.
        loss = criterion(logits, y_batch)
# Perform backpropagation of error.
        loss.backward()
# Limit gradient explosion for stability.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Update model weights.
        optimizer.step()
# Empty line.

# Accumulate loss for logging.
        total_loss += loss.item() * X_batch.size(0)
# Convert probs to single index predictions.
        preds = torch.argmax(logits, dim=-1)
# Increment successful match count.
        correct += (preds == y_batch).sum().item()
# Track total data processed.
        total += X_batch.size(0)
# Empty line.

# Return the averages.
    return total_loss / total, correct / total
# Empty line.
# Empty line.


# Specific trainer for the Intent branch.
def train_epoch_posture(model, loader, criterion, optimizer, device):
# Docstring.
    """Train one epoch for posture classifier."""
# Learning mode.
    model.train()
# Reset counters.
    total_loss, correct, total = 0.0, 0, 0
# Empty line.

# Iterate batches of features and the second target label.
    for X_batch, _, y_batch in loader:
# Hardware assignment.
        X_batch = X_batch.to(device)
# Label hardware assignment.
        y_batch = y_batch.to(device)
# Empty line.

# Zero gradients.
        optimizer.zero_grad()
# Forward pass.
        logits = model(X_batch)
# Loss calculation.
        loss = criterion(logits, y_batch)
# Backward pass.
        loss.backward()
# Stabilization.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Weights update.
        optimizer.step()
# Empty line.

# Track stats.
        total_loss += loss.item() * X_batch.size(0)
# Get final class.
        preds = torch.argmax(logits, dim=-1)
# Count accuracy.
        correct += (preds == y_batch).sum().item()
# Increment count.
        total += X_batch.size(0)
# Empty line.

# Return epoch results.
    return total_loss / total, correct / total
# Empty line.
# Empty line.


# Global evaluation function (not training).
@torch.no_grad()
def evaluate(model, loader, criterion, device, target_idx=1):
# Docstring block.
    """Evaluate model. target_idx: 1=action, 2=posture in batch tuple."""
# Switch to non-learning evaluation mode.
    model.eval()
# Counters.
    total_loss, correct, total = 0.0, 0, 0
# Containers for final metrics.
    all_preds, all_labels = [], []
# Empty line.

# Batch loop.
    for batch in loader:
# Features.
        X_batch = batch[0].to(device)
# Specific label based on target index.
        y_batch = batch[target_idx].to(device)
# Empty line.

# Forward pass.
        logits = model(X_batch)
# Error check.
        loss = criterion(logits, y_batch)
# Empty line.

# Stats tracking.
        total_loss += loss.item() * X_batch.size(0)
# Prediction.
        preds = torch.argmax(logits, dim=-1)
# Hit count.
        correct += (preds == y_batch).sum().item()
# Running total.
        total += X_batch.size(0)
# Empty line.

# Convert findings back for CPU metrics.
        all_preds.extend(preds.cpu().numpy())
# Convert labels back.
        all_labels.extend(y_batch.cpu().numpy())
# Empty line.

# Final averages.
    avg_loss = total_loss / total
# Final Accuracy.
    accuracy = correct / total
# Return metrics.
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
# Empty line.
# Empty line.


# Decorative separator line.
# ──────────────────────────────────────────────
# High-level pipeline management function.
# Main training function
# Decorative separator line.
# ──────────────────────────────────────────────
# Orchestrates training of both Oracle brain subsystems.
def train_oracle(config=None):
# Docstring.
    """Train both action predictor and intent classifier."""
# Merge default config with any overrides.
    cfg = {**DEFAULT_CONFIG, **(config or {})}
# Empty line.

# Folder generation.
    # Create checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
# Empty line.

# Visual console branding.
    print("╔═══════════════════════════════════════════════════╗")
    print("║  TDSS Oracle — Training Pipeline                 ║")
    print(f"║  Device: {cfg['device']:<42}║")
    print("╚═══════════════════════════════════════════════════╝")
# Empty line.

# Logic for data preparation.
    # ── Load data ──
    print("\n📊 Loading and preprocessing data...")
# Invoke global data loader script.
    train_loader, val_loader, posture_encoder, num_features = get_dataloaders(
        cfg["csv_path"], cfg["window_size"], cfg["batch_size"], cfg["val_split"]
    )
# Empty line.

# Visual phase tracker.
    # ═══════════════════════════════════════════
    # 1. Train LSTM Action Predictor
    # ═══════════════════════════════════════════
# Divider line.
    print("\n" + "=" * 55)
# Title.
    print("  Phase B2: LSTM Action Predictor")
# Divider line.
    print("=" * 55)
# Empty line.

# Instantiate B2 model.
    action_model = LSTMPredictor(
# Dimensions.
        input_size=num_features,
# Complexity of interior layers.
        hidden_size=cfg["hidden_size"],
# Stacking config.
        num_layers=cfg["num_layers"],
# Move categories count.
        num_classes=ACTION_CLASSES,
# Regularization config.
        dropout=cfg["dropout"],
# Targeted hardware initialization.
    ).to(cfg["device"])
# Empty line.

# Inform about neural network size.
    print(f"  Parameters: {sum(p.numel() for p in action_model.parameters()):,}")
# Empty line.

# Define error metric.
    action_criterion = nn.CrossEntropyLoss()
# Define optimizer algorithm.
    action_optimizer = optim.Adam(
# Parameters and learning speed selection.
        action_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
# Intelligence to slow down training speed automatically as needed.
    action_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# Watch loss and reduce speed by half if stuck.
        action_optimizer, mode="min", factor=0.5, patience=3
    )
# Empty line.

# Placeholder for top score tracking.
    best_action_acc = 0.0
# Log for plotting curves later.
    action_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
# Empty line.

# Loop through iterations.
    for epoch in range(1, cfg["epochs"] + 1):
# Timer start.
        t0 = time.time()
# Execute training function.
        train_loss, train_acc = train_epoch(
# Pass training components.
            action_model, train_loader, action_criterion,
            action_optimizer, cfg["device"]
        )
# Execute evaluation function on unseen data.
        val_loss, val_acc, _, _ = evaluate(
            action_model, val_loader, action_criterion, cfg["device"], target_idx=1
        )
# Update scheduler regarding progress.
        action_scheduler.step(val_loss)
# Calculation duration.
        dt = time.time() - t0
# Empty line.

# Save stats to log.
        action_history["train_loss"].append(train_loss)
        action_history["val_loss"].append(val_loss)
        action_history["train_acc"].append(train_acc)
        action_history["val_acc"].append(val_acc)
# Empty line.

# Dynamic star indicator if results improved.
        marker = " ★" if val_acc > best_action_acc else ""
# Progress report output to console.
        print(f"  Epoch {epoch:>2}/{cfg['epochs']}  "
              f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"Acc: {train_acc:.3f}/{val_acc:.3f}  "
              f"[{dt:.1f}s]{marker}")
# Empty line.

# Decision to save weight file or skip.
        if val_acc > best_action_acc:
# Update high score.
            best_action_acc = val_acc
# Serializing and writing the model state to disk.
            torch.save(action_model.state_dict(),
                       os.path.join(cfg["checkpoint_dir"], "action_predictor_best.pt"))
# Empty line.

# One last run to verify performance.
    # Final evaluation
    _, _, a_preds, a_labels = evaluate(
# Full evaluation on val set.
        action_model, val_loader, action_criterion, cfg["device"], target_idx=1
    )
# Output results summary.
    print(f"\n  Best Action Accuracy: {best_action_acc:.4f}")
    print("\n  Action Classification Report:")
# Humanly readable details on precision and recall for each specific move.
    print(classification_report(a_labels, a_preds, target_names=ACTION_NAMES,
                                labels=range(ACTION_CLASSES), zero_division=0))
# Empty line.

# Visual transition.
    # ═══════════════════════════════════════════
    # 2. Train Intent Classifier
    # ═══════════════════════════════════════════
# Border line.
    print("\n" + "=" * 55)
# Title.
    print("  Phase B3: Intent Classifier (Posture)")
# Border line.
    print("=" * 55)
# Empty line.

# Instantiate B3 branch.
    intent_model = IntentClassifier(
# Same feature count as action model.
        input_size=num_features,
# Specialized width.
        hidden_size=96,
# Stacking depth.
        num_layers=cfg["num_layers"],
# 5 posture categories.
        num_postures=len(POSTURE_CLASSES),
# Prob ratio.
        dropout=cfg["dropout"],
# Acceleration hardware.
    ).to(cfg["device"])
# Empty line.

# Report size.
    print(f"  Parameters: {sum(p.numel() for p in intent_model.parameters()):,}")
# Empty line.

# Error definition.
    intent_criterion = nn.CrossEntropyLoss()
# Logic setup for solver.
    intent_optimizer = optim.Adam(
# Parameters and learning rate.
        intent_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
# LR management.
    intent_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# Watch loss every few turns and reduce speed as needed.
        intent_optimizer, mode="min", factor=0.5, patience=3
    )
# Empty line.

# Counter.
    best_intent_acc = 0.0
# Log.
    intent_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
# Empty line.

# Iteration loop for phase 2.
    for epoch in range(1, cfg["epochs"] + 1):
# Timer start.
        t0 = time.time()
# Step execution.
        train_loss, train_acc = train_epoch_posture(
            intent_model, train_loader, intent_criterion,
            intent_optimizer, cfg["device"]
        )
# Step validation.
        val_loss, val_acc, _, _ = evaluate(
            intent_model, val_loader, intent_criterion, cfg["device"], target_idx=2
        )
# Speed management update.
        intent_scheduler.step(val_loss)
# Step duration tally.
        dt = time.time() - t0
# Empty line.

# Log values.
        intent_history["train_loss"].append(train_loss)
        intent_history["val_loss"].append(val_loss)
        intent_history["train_acc"].append(train_acc)
        intent_history["val_acc"].append(val_acc)
# Empty line.

# Score check.
        marker = " ★" if val_acc > best_intent_acc else ""
# Progress text output.
        print(f"  Epoch {epoch:>2}/{cfg['epochs']}  "
              f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"Acc: {train_acc:.3f}/{val_acc:.3f}  "
              f"[{dt:.1f}s]{marker}")
# Empty line.

# Condition for saving best results to file.
        if val_acc > best_intent_acc:
# Accuracy update.
            best_intent_acc = val_acc
# Save file to disk.
            torch.save(intent_model.state_dict(),
                       os.path.join(cfg["checkpoint_dir"], "intent_classifier_best.pt"))
# Empty line.

# Final confirmation.
    # Final evaluation
    _, _, p_preds, p_labels = evaluate(
# One last pass.
        intent_model, val_loader, intent_criterion, cfg["device"], target_idx=2
    )
# Log result summary.
    print(f"\n  Best Intent Accuracy: {best_intent_acc:.4f}")
    print("\n  Posture Classification Report:")
# Inform on error trends for each strategy (Scout, Attack, etc).
    print(classification_report(p_labels, p_preds, target_names=POSTURE_CLASSES,
                                labels=range(len(POSTURE_CLASSES)), zero_division=0))
# Empty line.

# Visual documentation logic.
    # ── Save loss curves ──
# Call plotter function correctly mapping all stored logs.
    _save_loss_curves(action_history, intent_history, cfg["checkpoint_dir"])
# Empty line.

# Archival summary logic.
    # ── Save training summary ──
# JSON structure for review.
    summary = {
# Performance stats.
        "best_action_accuracy": best_action_acc,
        "best_intent_accuracy": best_intent_acc,
# Params.
        "epochs": cfg["epochs"],
        "window_size": cfg["window_size"],
        "num_features": num_features,
        "hidden_size": cfg["hidden_size"],
        "device": cfg["device"],
# Closing dict.
    }
# Write text to filesystem.
    with open(os.path.join(cfg["checkpoint_dir"], "training_summary.json"), "w") as f:
# Map python dict to nicely spaced text.
        json.dump(summary, f, indent=2)
# Empty line.

# Final visual closing.
    print("\n" + "=" * 55)
    print("  Oracle Training Complete!")
# Stats output.
    print(f"  Action predictor:  {best_action_acc:.4f} accuracy")
    print(f"  Intent classifier: {best_intent_acc:.4f} accuracy")
# File location report.
    print(f"  Checkpoints saved: {cfg['checkpoint_dir']}/")
    print("=" * 55)
# Empty line.

# Finish function and pass models back up.
    return action_model, intent_model
# Empty line.
# Empty line.


# Logic to create image files charting data.
def _save_loss_curves(action_hist, intent_hist, output_dir):
# Docstring.
    """Save matplotlib loss/accuracy curves."""
# Safety logic to handle systems without plotting tools.
    try:
# Import plotter library.
        import matplotlib
# Prevent window popups on servers without GUI displays.
        matplotlib.use("Agg")
# Import specific visualization tools.
        import matplotlib.pyplot as plt
# Empty line.

# Create 2x2 grid of graphs.
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Set the main image header.
        fig.suptitle("TDSS Oracle — Training Curves", fontsize=14, fontweight="bold")
# Empty line.

# Section for action branch.
        # Action loss
# Color Blue for training loss.
        axes[0, 0].plot(action_hist["train_loss"], label="Train", color="#3498db")
# Color Red for validation loss.
        axes[0, 0].plot(action_hist["val_loss"], label="Val", color="#e74c3c")
# Header.
        axes[0, 0].set_title("Action Predictor — Loss")
# Axis label.
        axes[0, 0].set_xlabel("Epoch")
# Show colored labels.
        axes[0, 0].legend()
# Visual guidance lines.
        axes[0, 0].grid(True, alpha=0.3)
# Empty line.

# Action accuracy chart.
        # Action accuracy
# Blue training line.
        axes[0, 1].plot(action_hist["train_acc"], label="Train", color="#3498db")
# Red val line.
        axes[0, 1].plot(action_hist["val_acc"], label="Val", color="#e74c3c")
# Title.
        axes[0, 1].set_title("Action Predictor — Accuracy")
# Axis.
        axes[0, 1].set_xlabel("Epoch")
# Legend mapping.
        axes[0, 1].legend()
# Visual grid.
        axes[0, 1].grid(True, alpha=0.3)
# Empty line.

# Phase 2 graphs (Green/Orange).
        # Intent loss
# Green for intent branch training.
        axes[1, 0].plot(intent_hist["train_loss"], label="Train", color="#2ecc71")
# Orange for intent validation.
        axes[1, 0].plot(intent_hist["val_loss"], label="Val", color="#e67e22")
# Graph header.
        axes[1, 0].set_title("Intent Classifier — Loss")
# Label axis.
        axes[1, 0].set_xlabel("Epoch")
# Legend logic.
        axes[1, 0].legend()
# Visual transparency.
        axes[1, 0].grid(True, alpha=0.3)
# Empty line.

# Intent Acc chart.
        # Intent accuracy
# Green line.
        axes[1, 1].plot(intent_hist["train_acc"], label="Train", color="#2ecc71")
# Orange line.
        axes[1, 1].plot(intent_hist["val_acc"], label="Val", color="#e67e22")
# Title.
        axes[1, 1].set_title("Intent Classifier — Accuracy")
# Axis.
        axes[1, 1].set_xlabel("Epoch")
# Labeling.
        axes[1, 1].legend()
# Grid logic.
        axes[1, 1].grid(True, alpha=0.3)
# Empty line.

# Layout auto-spacing logic.
        plt.tight_layout()
# Construct file destination string.
        path = os.path.join(output_dir, "training_curves.png")
# Encode image and record it to disk location.
        plt.savefig(path, dpi=150)
# Clean up graphics memory.
        plt.close()
# Log success to console.
        print(f"\n  📈 Loss curves saved: {path}")
# Logic catch if image generation fails.
    except Exception as e:
# Error message.
        print(f"\n  ⚠ Could not save loss curves: {e}")
# Empty line.
# Empty line.


# Decorative separator line.
# ──────────────────────────────────────────────
# Runtime entry condition block.
# Entry point
# Decorative separator line.
# ──────────────────────────────────────────────
# Check for direct script execution.
if __name__ == "__main__":
# Initiate training procedure.
    train_oracle()
# Closing marker line.
