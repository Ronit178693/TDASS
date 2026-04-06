"""
data_loader.py — B1: Data Preprocessing for Oracle
====================================================
Loads battle_data.csv, creates windowed sequences (last N steps) for LSTM,
and provides PyTorch Dataset / DataLoader utilities.

Features extracted per timestep:
  - red_x, red_y, blue_x, blue_y          (positions)
  - red_hp, red_ammo, red_fuel             (resources)
  - red_morale, red_elevation              (state)
  - distance, fog_visible                  (tactical)
  - red_prev_action                        (history)

Targets:
  - red_next_action   (6 classes: Stay/Up/Down/Left/Right/RangedAttack)
  - red_posture       (5 classes: SCOUT/ATTACK/RETREAT/FLANK/RESUPPLY)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
# Feature & label columns
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "red_x", "red_y", "blue_x", "blue_y",
    "red_hp", "red_ammo", "red_fuel",
    "red_morale", "red_elevation",
    "distance", "fog_visible",
    "red_prev_action",
]

POSTURE_CLASSES = ["SCOUT", "ATTACK", "RETREAT", "FLANK", "RESUPPLY"]
ACTION_CLASSES  = 6  # 0-5

# Normalization ranges (for min-max scaling)
NORM_RANGES = {
    "red_x": (0, 9), "red_y": (0, 9),
    "blue_x": (0, 9), "blue_y": (0, 9),
    "red_hp": (0, 100), "red_ammo": (0, 50), "red_fuel": (0, 100),
    "red_morale": (20, 150), "red_elevation": (0, 2),
    "distance": (0, 18), "fog_visible": (0, 1),
    "red_prev_action": (0, 5),
}


def normalize_features(df, feature_cols=FEATURE_COLS):
    """Min-max normalize feature columns in-place, returns normalized copy."""
    df_norm = df.copy()
    for col in feature_cols:
        lo, hi = NORM_RANGES.get(col, (df[col].min(), df[col].max()))
        rng = hi - lo if hi != lo else 1.0
        df_norm[col] = (df[col].astype(float) - lo) / rng
    return df_norm


def load_battle_data(csv_path, feature_cols=FEATURE_COLS):
    """
    Load and preprocess battle_data.csv.

    Returns:
        df: raw DataFrame
        df_norm: normalized DataFrame
        posture_encoder: fitted LabelEncoder for postures
    """
    df = pd.read_csv(csv_path)

    # Ensure numeric types
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encode posture labels
    posture_encoder = LabelEncoder()
    posture_encoder.fit(POSTURE_CLASSES)
    # Handle any unseen labels gracefully
    df["posture_label"] = df["red_posture"].apply(
        lambda x: posture_encoder.transform([x])[0] if x in POSTURE_CLASSES else 0
    )

    # Action label (already int 0-5)
    df["action_label"] = df["red_next_action"].astype(int)

    # Normalize
    df_norm = normalize_features(df, feature_cols)

    return df, df_norm, posture_encoder


# ──────────────────────────────────────────────
# Windowed sequence builder
# ──────────────────────────────────────────────
def build_sequences(df_norm, df_raw, window_size=10, feature_cols=FEATURE_COLS):
    """
    Build sliding-window sequences grouped by (match_id, red_id).

    Args:
        df_norm:      normalized DataFrame
        df_raw:       raw DataFrame (for labels)
        window_size:  number of timesteps per sequence

    Returns:
        X: np.ndarray (N, window_size, num_features)
        y_action: np.ndarray (N,)  — next action class
        y_posture: np.ndarray (N,) — posture class
        positions: np.ndarray (N, 2) — (red_x, red_y) at prediction time
    """
    X_list, ya_list, yp_list, pos_list = [], [], [], []

    grouped = df_norm.groupby(["match_id", "red_id"])

    for (match_id, red_id), group in grouped:
        features = group[feature_cols].values
        raw_group = df_raw.loc[group.index]
        actions   = raw_group["action_label"].values
        postures  = raw_group["posture_label"].values
        red_xs    = raw_group["red_x"].values
        red_ys    = raw_group["red_y"].values

        if len(features) < window_size + 1:
            continue

        for i in range(window_size, len(features)):
            window = features[i - window_size : i]
            X_list.append(window)
            ya_list.append(actions[i])
            yp_list.append(postures[i])
            pos_list.append([red_xs[i], red_ys[i]])

    X = np.array(X_list, dtype=np.float32)
    y_action  = np.array(ya_list, dtype=np.int64)
    y_posture = np.array(yp_list, dtype=np.int64)
    positions = np.array(pos_list, dtype=np.float32)

    return X, y_action, y_posture, positions


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────
class BattleSequenceDataset(Dataset):
    """PyTorch Dataset wrapping windowed battle sequences."""

    def __init__(self, X, y_action, y_posture):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_action  = torch.tensor(y_action, dtype=torch.long)
        self.y_posture = torch.tensor(y_posture, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_action[idx], self.y_posture[idx]


def get_dataloaders(csv_path, window_size=10, batch_size=128,
                    val_split=0.2, seed=42):
    """
    End-to-end: load CSV → build sequences → split → return DataLoaders.

    Returns:
        train_loader, val_loader, posture_encoder, num_features
    """
    df, df_norm, posture_encoder = load_battle_data(csv_path)
    X, y_action, y_posture, _ = build_sequences(df_norm, df, window_size)

    print(f"  Sequences built: {len(X):,}")
    print(f"  Feature dim:     {X.shape[2]}")
    print(f"  Action classes:  {ACTION_CLASSES}")
    print(f"  Posture classes: {len(POSTURE_CLASSES)}")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - val_split))

    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_ds = BattleSequenceDataset(X[train_idx], y_action[train_idx], y_posture[train_idx])
    val_ds   = BattleSequenceDataset(X[val_idx],   y_action[val_idx],   y_posture[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")

    return train_loader, val_loader, posture_encoder, X.shape[2]
