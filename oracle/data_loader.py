# Comprehensive data loading and preprocessing utility for the Oracle AI subsystem.
"""
# This module serves as the primary data pipeline, transforming raw battle logs into structured sequences for the LSTM.
data_loader.py — B1: Data Preprocessing for Oracle
# Visual divider for internal documentation structure.
====================================================
# The logic here handles sliding window generation, min-max feature scaling, and categorical encoding.
Loads battle_data.csv, creates windowed sequences (last N steps) for LSTM,
# It ensures the PyTorch models receive normalized, temporally consistent tensors.
and provides PyTorch Dataset / DataLoader utilities.

# The feature set encompasses spatial, resource, psychological, and tactical dimensions.
Features extracted per timestep:
# Unit coordinates allow the model to learn spatial positioning and flanking patterns.
  - red_x, red_y, blue_x, blue_y          (positions)
# Tracking vital stats like HP and Ammo helps the model predict Retreat or Resupply intents.
  - red_hp, red_ammo, red_fuel             (resources)
# Morale and elevation provide the "soft" tactical context of the engagement.
  - red_morale, red_elevation              (state)
# Distance and visibility metrics define the visibility constraints (Fog of War) for the AI.
  - distance, fog_visible                  (tactical)
# Historical action context allows the LSTM to recognize repetitive or chained behaviors.
  - red_prev_action                        (history)

# The model is trained to predict two distinct levels of enemy behavior.
Targets:
# Micros-tactical level: The specific grid-based move or combat action taken in the next frame.
  - red_next_action   (6 classes: Stay/Up/Down/Left/Right/RangedAttack)
# Macro-strategic level: The overarching tactical intent or "posture" of the enemy unit.
  - red_posture       (5 classes: SCOUT/ATTACK/RETREAT/FLANK/RESUPPLY)
"""

# Import numpy for high-performance numerical array operations and sequence manipulation.
import numpy as np
# Import pandas for flexible CSV parsing and tabular data transformations.
import pandas as pd
# Import the core PyTorch library for deep learning tensor operations.
import torch
# Import standard PyTorch utilities for batching and dataset management.
from torch.utils.data import Dataset, DataLoader
# Import LabelEncoder to map categorical posture strings into numerical classification targets.
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# Global Configuration: Define the numerical signatures of the battlefield.
# ──────────────────────────────────────────────

# Explicit list of columns that represent the input state space for the neural network.
FEATURE_COLS = [
    # Red and Blue grid coordinates, enabling the model to learn spatial relationships.
    "red_x", "red_y", "blue_x", "blue_y",
    # Vital unit metrics used to infer tactical urgency (e.g., low HP suggests a retreat).
    "red_hp", "red_ammo", "red_fuel",
    # Psychological and environmental variables that modify combat effectiveness.
    "red_morale", "red_elevation",
    # Pre-calculated tactical features that simplify the model's spatial reasoning.
    "distance", "fog_visible",
    # The previous action ensures the model has a "short-term memory" of immediate history.
    "red_prev_action",
]

# The high-level behaviors we want the 'Intent Classifier' to recognize in the enemy.
POSTURE_CLASSES = ["SCOUT", "ATTACK", "RETREAT", "FLANK", "RESUPPLY"]
# Total number of discrete actions available in the BattleEnv Gymnasium environment.
ACTION_CLASSES  = 6  # 0-5

# Hardcoded normalization bounds to ensure consistent feature scaling across training and inference.
NORM_RANGES = {
    # Boundaries for unit positions on the 10x10 tactical grid.
    "red_x": (0, 9), "red_y": (0, 9),
    "blue_x": (0, 9), "blue_y": (0, 9),
    # Maximum possible values for unit resources as defined in the environment constants.
    "red_hp": (0, 100), "red_ammo": (0, 50), "red_fuel": (0, 100),
    # Morale and Elevation ranges used to scale these state variables between 0 and 1.
    "red_morale": (20, 150), "red_elevation": (0, 2),
    # Maximum Manhattan distance on a 10x10 grid and binary visibility flag.
    "distance": (0, 18), "fog_visible": (0, 1),
    # Action indices ranging from 0 (Stay) to 5 (RangedAttack).
    "red_prev_action": (0, 5),
}

# ──────────────────────────────────────────────
# Data Transformation Logic
# ──────────────────────────────────────────────

# Scaler function that implements Min-Max normalization for numerical stability in the LSTM.
def normalize_features(df, feature_cols=FEATURE_COLS):
    """Min-max normalize feature columns to the [0, 1] range based on NORM_RANGES."""
    # Create a deep copy of the DataFrame to ensure we don't modify the source data during preprocessing.
    df_norm = df.copy()
    # Iterate through every feature column required by the model.
    for col in feature_cols:
        # Fetch the predefined min/max values for this specific feature; fall back to data min/max if not found.
        lo, hi = NORM_RANGES.get(col, (df[col].min(), df[col].max()))
        # Determine the denominator for scaling; default to 1.0 if the column is constant to avoid division errors.
        rng = hi - lo if hi != lo else 1.0
        # Perform the actual scaling: (Value - Min) / (Max - Min).
        df_norm[col] = (df[col].astype(float) - lo) / rng
    # Return the fully normalized dataset, now ready for neural network consumption.
    return df_norm

# Primary loader function that converts the raw CSV log into a structured data format.
def load_battle_data(csv_path, feature_cols=FEATURE_COLS):
    """Loads raw battle logs, parses categorical labels, and applies normalization."""
    # Use pandas to load the generated CSV from the simulation runs.
    df = pd.read_csv(csv_path)

    # Sanitize the input by ensuring all expected feature columns are treated as numeric data.
    for col in feature_cols:
        # Convert to numbers and fill any missing or corrupt entries with 0.
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Initialize the SKLearn LabelEncoder to transform string postures (e.g., 'ATTACK') into class indices.
    posture_encoder = LabelEncoder()
    # Fit the encoder to our fixed set of strategic postures.
    posture_encoder.fit(POSTURE_CLASSES)
    # Map the enemy's categorical 'posture' into a 'posture_label' column for classification training.
    df["posture_label"] = df["red_posture"].apply(
        # Handle lookup: if a posture is unknown, default to the first class (index 0).
        lambda x: posture_encoder.transform([x])[0] if x in POSTURE_CLASSES else 0
    )

    # Ensure the 'next_action' target is represented as an integer Action ID (0-5).
    df["action_label"] = df["red_next_action"].astype(int)

    # Apply global normalization scaling to prepare the data for the gradient descent process.
    df_norm = normalize_features(df, feature_cols)

    # Return the raw data, the normalized data, and the encoder for later use during inference or evaluation.
    return df, df_norm, posture_encoder

# ──────────────────────────────────────────────
# Sequence Engineering: Constructing the sliding window history.
# ──────────────────────────────────────────────

# Core function that converts flat tabular data into 3D tensors (Samples, Time, Features) for the LSTM.
def build_sequences(df_norm, df_raw, window_size=10, feature_cols=FEATURE_COLS):
    """Constructs 3D time-series sequences grouped by specific unit IDs to prevent data leakage."""
    # Lists used to accumulate the 3D X tensors and their corresponding 1D y labels.
    X_list, ya_list, yp_list, pos_list = [], [], [], []

    # Extremely important: Group by match and unit ID so we don't accidentally blend different units' histories.
    grouped = df_norm.groupby(["match_id", "red_id"])

    # Process each unique unit's trajectory through a single match as a separate timeline.
    for (match_id, red_id), group in grouped:
        # Extract the sequence of normalized feature vectors.
        features = group[feature_cols].values
        # Link the normalized group back to the raw labels using their shared DataFrame index.
        raw_group = df_raw.loc[group.index]
        # Extract the ground-truth targets (Actions and Postures) for supervised learning.
        actions   = raw_group["action_label"].values
        postures  = raw_group["posture_label"].values
        # Store coordinates separately for visualizing heatmap predictions later.
        red_xs    = raw_group["red_x"].values
        red_ys    = raw_group["red_y"].values

        # If a unit has fewer steps than our required history window, we cannot generate a sequence for it.
        if len(features) < window_size + 1:
            # Skip this unit and proceed to the next available trajectory.
            continue

        # Implement the sliding window: For every step 'i', look back 'window_size' steps.
        for i in range(window_size, len(features)):
            # Capture the past 'window_size' steps as the input 'X'.
            window = features[i - window_size : i]
            # Add this historical context to our training buffer.
            X_list.append(window)
            # The label 'y' is always the action/posture taken at step 'i' (immediately following the window).
            ya_list.append(actions[i])
            yp_list.append(postures[i])
            # Save the coordinates so we know where this target action actually occurred on the map.
            pos_list.append([red_xs[i], red_ys[i]])

    # Convert the Python lists into high-performance 32-bit float numpy arrays for model training.
    X = np.array(X_list, dtype=np.float32)
    # Labels must be 64-bit integers (Long) for PyTorch classification loss functions.
    y_action  = np.array(ya_list, dtype=np.int64)
    y_posture = np.array(yp_list, dtype=np.int64)
    # Positions remain as floats for spatial precision.
    positions = np.array(pos_list, dtype=np.float32)

    # Return the assembled 3D input tensor and targets.
    return X, y_action, y_posture, positions

# ──────────────────────────────────────────────
# Deep Learning Integration: PyTorch Dataset Wrapper.
# ──────────────────────────────────────────────

# Custom Dataset class that allows PyTorch DataLoaders to fetch and batch our sequences efficiently.
class BattleSequenceDataset(Dataset):
    """A standard PyTorch Dataset used to wrap and serve tactical sequence data."""

    # Constructor that converts pre-built numpy arrays into PyTorch Tensors.
    def __init__(self, X, y_action, y_posture):
        # The 3D input tensor for the LSTM layers.
        self.X = torch.tensor(X, dtype=torch.float32)
        # The ground-truth tactical action labels (the target classes).
        self.y_action  = torch.tensor(y_action, dtype=torch.long)
        # The ground-truth strategic intent labels for the secondary classifier.
        self.y_posture = torch.tensor(y_posture, dtype=torch.long)

    # Returns the total number of sequences available in this specific dataset partition.
    def __len__(self):
        # Simply returns the length of the input tensor array.
        return len(self.X)

    # Standard getter that returns a single (input, action_target, posture_target) tuple for a given index.
    def __getitem__(self, idx):
        # Fetches the specific data point from internal tensor storage.
        return self.X[idx], self.y_action[idx], self.y_posture[idx]

# Master orchestrator function that manages the entire pipeline from file to PyTorch Loaders.
def get_dataloaders(csv_path, window_size=10, batch_size=128, val_split=0.2, seed=42):
    """Executes the full pipeline: Load -> Preprocess -> Sequence -> Split -> Loaders."""
    # Step 1: Load and normalize the raw data from the simulation logs.
    df, df_norm, posture_encoder = load_battle_data(csv_path)
    # Step 2: Transform the tabular data into windowed time-series sequences.
    X, y_action, y_posture, _ = build_sequences(df_norm, df, window_size)

    # Print descriptive logs to help monitor dataset health and dimensions.
    print(f"  Sequences built: {len(X):,}")
    print(f"  Feature dim:     {X.shape[2]}")
    print(f"  Action classes:  {ACTION_CLASSES}")
    print(f"  Posture classes: {len(POSTURE_CLASSES)}")

    # Step 3: Implement a randomized split to separate 'Training' data from 'Validation' data.
    # Instantiate a repeatable random number generator for deterministic splits.
    rng = np.random.RandomState(seed)
    # Creates a shuffled array of index pointers for all available sequences.
    indices = rng.permutation(len(X))
    # Calculate the exact dividing point based on the val_split ratio.
    split = int(len(X) * (1 - val_split))

    # Divide the index pointers into two distinct sets.
    train_idx = indices[:split]
    val_idx   = indices[split:]

    # Step 4: Create the actual Training and Validation Dataset objects.
    train_ds = BattleSequenceDataset(X[train_idx], y_action[train_idx], y_posture[train_idx])
    val_ds   = BattleSequenceDataset(X[val_idx],   y_action[val_idx],   y_posture[val_idx])

    # Step 5: Wrap the Datasets in Loaders that handle automated batching and shuffling.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # Final report to verify batch counts.
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")

    # Return the ready-to-use loaders, the encoder for labels, and the count of input features.
    return train_loader, val_loader, posture_encoder, X.shape[2]
