# Document string for the data loader module.
"""
# Source file name and module purpose.
data_loader.py — B1: Data Preprocessing for Oracle
# Visual separator for documentation.
====================================================
# Summary of script functionality.
Loads battle_data.csv, creates windowed sequences (last N steps) for LSTM,
# Details on hardware utility classes provided.
and provides PyTorch Dataset / DataLoader utilities.
# Empty line.

# List of input variables being analyzed.
Features extracted per timestep:
# Positional coordinates for both factions.
  - red_x, red_y, blue_x, blue_y          (positions)
# Vital resource statistics.
  - red_hp, red_ammo, red_fuel             (resources)
# Internal psychological and environmental state.
  - red_morale, red_elevation              (state)
# Distances and visibility conditions.
  - distance, fog_visible                  (tactical)
# Historical action context.
  - red_prev_action                        (history)
# Empty line.

# Description of the target outputs for prediction.
Targets:
# Movement and combat action categories.
  - red_next_action   (6 classes: Stay/Up/Down/Left/Right/RangedAttack)
# Strategic intent categories.
  - red_posture       (5 classes: SCOUT/ATTACK/RETREAT/FLANK/RESUPPLY)
# Closing docstring.
"""
# Empty line.

# Numerical computations library.
import numpy as np
# Data manipulation and analysis library.
import pandas as pd
# Machine learning framework.
import torch
# Specific utilities for managing data batches.
from torch.utils.data import Dataset, DataLoader
# Utility to convert labels into numerical formats.
from sklearn.preprocessing import LabelEncoder
# Empty line.


# Decorative line for visual grouping.
# ──────────────────────────────────────────────
# Header for the feature and label configuration section.
# Feature & label columns
# Decorative line for visual grouping.
# ──────────────────────────────────────────────
# List of column names used as model inputs.
FEATURE_COLS = [
# X and Y coordinates.
    "red_x", "red_y", "blue_x", "blue_y",
# Health, ammunition, and fuel stats.
    "red_hp", "red_ammo", "red_fuel",
# Morale and altitude stats.
    "red_morale", "red_elevation",
# Tactical distance and fog status.
    "distance", "fog_visible",
# Previous step's action code.
    "red_prev_action",
# Closing list bracket.
]
# Empty line.

# List of strings defining enemy strategic postures.
POSTURE_CLASSES = ["SCOUT", "ATTACK", "RETREAT", "FLANK", "RESUPPLY"]
# Integer count of possible discrete actions.
ACTION_CLASSES  = 6  # 0-5
# Empty line.

# Dictionary defining normalization boundaries for min-max scaling.
# Normalization ranges (for min-max scaling)
NORM_RANGES = {
# X and Y coordinate limits.
    "red_x": (0, 9), "red_y": (0, 9),
# Foe X and Y coordinate limits.
    "blue_x": (0, 9), "blue_y": (0, 9),
# Resource value limits.
    "red_hp": (0, 100), "red_ammo": (0, 50), "red_fuel": (0, 100),
# Psychology and environment limits.
    "red_morale": (20, 150), "red_elevation": (0, 2),
# Tactical distance and visibility limits.
    "distance": (0, 18), "fog_visible": (0, 1),
# Action index limits.
    "red_prev_action": (0, 5),
# Closing dictionary bracket.
}
# Empty line.


# Function to scale features into a 0-1 range.
def normalize_features(df, feature_cols=FEATURE_COLS):
# Docstring explaining the min-max normalization process.
    """Min-max normalize feature columns in-place, returns normalized copy."""
# Create a copy of the dataframe to avoid source mutation.
    df_norm = df.copy()
# Iterate through each column designated for features.
    for col in feature_cols:
# Retrieve scaling bounds from the configuration dictionary.
        lo, hi = NORM_RANGES.get(col, (df[col].min(), df[col].max()))
# Calculate the range spread, defaulting to 1.0 to avoid division by zero.
        rng = hi - lo if hi != lo else 1.0
# Apply the min-max formula and store as floats.
        df_norm[col] = (df[col].astype(float) - lo) / rng
# Return the processed dataframe.
    return df_norm
# Empty line.


# Main data entry point for reading CSV files.
def load_battle_data(csv_path, feature_cols=FEATURE_COLS):
# Docstring block.
    """
# Overview of loading and preprocessing.
    Load and preprocess battle_data.csv.
# Empty line.

# Description of the three return objects.
    Returns:
# The original dataframe.
        df: raw DataFrame
# The scaled dataframe.
        df_norm: normalized DataFrame
# The encoder for text labels.
        posture_encoder: fitted LabelEncoder for postures
# Closing docstring.
    """
# Read the file from disk using pandas.
    df = pd.read_csv(csv_path)
# Empty line.

# Loop to ensure all feature columns are numeric type.
    # Ensure numeric types
    for col in feature_cols:
# Convert to number and fill gaps with zeros.
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
# Empty line.

# Logic for converting text postures into numbers.
    # Encode posture labels
# Initialize the encoder object.
    posture_encoder = LabelEncoder()
# Fit the encoder to the predefined category strings.
    posture_encoder.fit(POSTURE_CLASSES)
# Map text postures in the dataframe to their corresponding integer codes.
    # Handle any unseen labels gracefully
    df["posture_label"] = df["red_posture"].apply(
# Inline function to handle matching or defaulting to zero.
        lambda x: posture_encoder.transform([x])[0] if x in POSTURE_CLASSES else 0
# Closing apply function.
    )
# Empty line.

# Explicitly cast next actions to integers.
    # Action label (already int 0-5)
    df["action_label"] = df["red_next_action"].astype(int)
# Empty line.

# Apply the normalization scaling defined earlier.
    # Normalize
    df_norm = normalize_features(df, feature_cols)
# Empty line.

# Return the trio of data objects.
    return df, df_norm, posture_encoder
# Empty line.


# Decorative line.
# ──────────────────────────────────────────────
# Header for sequence construction section.
# Windowed sequence builder
# Decorative line.
# ──────────────────────────────────────────────
# Function to convert flat tabular data into windowed 3D tensors.
def build_sequences(df_norm, df_raw, window_size=10, feature_cols=FEATURE_COLS):
# Docstring block.
    """
# Description of sliding window technique grouped by entity IDs.
    Build sliding-window sequences grouped by (match_id, red_id).
# Empty line.

# Explanation of arguments.
    Args:
# The scaled dataset.
        df_norm:      normalized DataFrame
# The original dataset.
        df_raw:       raw DataFrame (for labels)
# The longitudinal depth of history.
        window_size:  number of timesteps per sequence
# Empty line.

# Explanation of return values.
    Returns:
# 3D input array.
        X: np.ndarray (N, window_size, num_features)
# Action targets.
        y_action: np.ndarray (N,)  — next action class
# Intent targets.
        y_posture: np.ndarray (N,) — posture class
# Spatial coordinates at the target time.
        positions: np.ndarray (N, 2) — (red_x, red_y) at prediction time
# Closing docstring.
    """
# Initialize empty lists to collect sequence fragments.
    X_list, ya_list, yp_list, pos_list = [], [], [], []
# Empty line.

# Split data by match session and specific unit ID to maintain temporal integrity.
    grouped = df_norm.groupby(["match_id", "red_id"])
# Empty line.

# Iterate over each group of chronological steps per unit.
    for (match_id, red_id), group in grouped:
# Extract pure numerical feature values.
        features = group[feature_cols].values
# Match numerical indices to the raw labels dataframe.
        raw_group = df_raw.loc[group.index]
# Extract targets and coordinates.
        actions   = raw_group["action_label"].values
        postures  = raw_group["posture_label"].values
        red_xs    = raw_group["red_x"].values
        red_ys    = raw_group["red_y"].values
# Empty line.

# Filter out paths too short to form a single window.
        if len(features) < window_size + 1:
# Skip to the next unit.
            continue
# Empty line.

# Sliding window iteration across the unit's timeline.
        for i in range(window_size, len(features)):
# Slice out a window of 'window_size' steps.
            window = features[i - window_size : i]
# Append the slice to the input list.
            X_list.append(window)
# Append the single target value corresponding to the step immediately after the window.
            ya_list.append(actions[i])
            yp_list.append(postures[i])
# Append spatial coordinates for heatmap validation.
            pos_list.append([red_xs[i], red_ys[i]])
# Empty line.

# Convert collected lists into solid numpy arrays.
    X = np.array(X_list, dtype=np.float32)
# Convert target labels to 64-bit integers.
    y_action  = np.array(ya_list, dtype=np.int64)
    y_posture = np.array(yp_list, dtype=np.int64)
# Keep coordinates as floats.
    positions = np.array(pos_list, dtype=np.float32)
# Empty line.

# Return the full dataset tensors.
    return X, y_action, y_posture, positions
# Empty line.


# Decorative line.
# ──────────────────────────────────────────────
# Header for PyTorch integration section.
# PyTorch Dataset
# Decorative line.
# ──────────────────────────────────────────────
# Class for feeding battle data into PyTorch trainers.
class BattleSequenceDataset(Dataset):
# Docstring.
    """PyTorch Dataset wrapping windowed battle sequences."""
# Empty line.

# Setup function to convert numpy arrays into torch tensors.
    def __init__(self, X, y_action, y_posture):
# Convert inputs to float tensors.
        self.X = torch.tensor(X, dtype=torch.float32)
# Convert targets to long tensors (required for classification cross-entropy).
        self.y_action  = torch.tensor(y_action, dtype=torch.long)
        self.y_posture = torch.tensor(y_posture, dtype=torch.long)
# Empty line.

# Return the total count of sequences.
    def __len__(self):
# Count of the input array.
        return len(self.X)
# Empty line.

# Return a specific item by its index.
    def __getitem__(self, idx):
# Return a tuple containing the input window and both labels.
        return self.X[idx], self.y_action[idx], self.y_posture[idx]
# Empty line.


# End-to-end wrapper for loading everything in one go.
def get_dataloaders(csv_path, window_size=10, batch_size=128,
                    val_split=0.2, seed=42):
# Docstring block detailing the process flow.
    """
# Flow: Disk -> Sequence -> Split -> Loaders.
    End-to-end: load CSV → build sequences → split → return DataLoaders.
# Empty line.

# Return items description.
    Returns:
# Torch loaders and metadata.
        train_loader, val_loader, posture_encoder, num_features
# Closing docstring.
    """
# Perform raw loading and normalization.
    df, df_norm, posture_encoder = load_battle_data(csv_path)
# Transform tabular data into windowed sequences.
    X, y_action, y_posture, _ = build_sequences(df_norm, df, window_size)
# Empty line.

# Debug output of dataset statistics.
    print(f"  Sequences built: {len(X):,}")
    print(f"  Feature dim:     {X.shape[2]}")
    print(f"  Action classes:  {ACTION_CLASSES}")
    print(f"  Posture classes: {len(POSTURE_CLASSES)}")
# Empty line.

# Procedural split for training vs validation.
    # Shuffle and split
# Initialize random state for reproducibility.
    rng = np.random.RandomState(seed)
# Generate a shuffled list of all data indices.
    indices = rng.permutation(len(X))
# Determine the integer split point based on the percentage.
    split = int(len(X) * (1 - val_split))
# Empty line.

# Assign indices to either training or validation sets.
    train_idx = indices[:split]
    val_idx   = indices[split:]
# Empty line.

# Wrap the partitioned numpy data into our custom Dataset classes.
    train_ds = BattleSequenceDataset(X[train_idx], y_action[train_idx], y_posture[train_idx])
    val_ds   = BattleSequenceDataset(X[val_idx],   y_action[val_idx],   y_posture[val_idx])
# Empty line.

# Create non-interactive loaders that handle mini-batching on the GPU/CPU.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
# Empty line.

# Final count report.
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
# Empty line.

# Final return of all constructed components.
    return train_loader, val_loader, posture_encoder, X.shape[2]
