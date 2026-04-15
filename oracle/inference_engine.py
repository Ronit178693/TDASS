# Real-time stateful inference wrapper for deploying the Oracle intelligence during live simulations.
"""
# This module acts as the interface between the raw environment state and the deep learning models.
inference_engine.py — B6: Real-time Intelligence Integration
=============================================================
# It manages a rolling history buffer to provide the required temporal context for the LSTM branches.
A clean wrapper to load trained Oracle models and run inference
during live simulations/visualizations.

# Standard usage pattern for integrating the Oracle into a tactical execution loop:
Usage:
  # Initialize the engine (automatically loads weights from the checkpoint directory).
  engine = OracleInferenceEngine()
  # In each simulation step, pass the current unit state to get a probabilistic forecast.
  prediction = engine.update(current_state_dict)
"""

# Import standard library for managing filesystem paths.
import os
# Import core PyTorch for model loading and tensor-based inference.
import torch
# Import numpy for high-performance array manipulation and post-processing of model outputs.
import numpy as np
# Import deque to implement a fixed-length FIFO (First-In-First-Out) buffer for time-series history.
from collections import deque

# Import architecture and configuration from the Oracle subsystem.
from oracle.lstm_predictor import LSTMPredictor
from oracle.intent_classifier import IntentClassifier
from oracle.heatmap_generator import HeatmapGenerator
from oracle.data_loader import FEATURE_COLS, NORM_RANGES, POSTURE_CLASSES

# The primary class responsible for maintaining state and executing model passes during gameplay.
class OracleInferenceEngine:
    """
    High-level controller for Oracle predictions. 
    It 'remembers' the past observations for a specific unit to satisfy the LSTM's window requirements.
    """

    # Constructor that initializes the models and prepares the historical buffers.
    def __init__(
        self,
        # Path where the pre-trained .pt weight files are stored.
        checkpoint_dir="oracle/checkpoints",
        # History depth; must match the window_size used during model training.
        window_size=10,
        # Execution device (CPU by default for inference stability, or 'cuda' for high-speed batching).
        device="cpu",
        # Physical boundary of the battlefield grid.
        grid_size=10
    ):
        # Store configurations as object properties for runtime reference.
        self.device = device
        self.window_size = window_size
        self.grid_size = grid_size

        # Initialize a deque with a fixed length. Deque is O(1) for adding/removing items from ends.
        # This rolling buffer ensures we always have the 'last 10' steps available for the LSTM.
        self.history = deque(maxlen=window_size)

        # ─── Model Loading Phase ───
        
        # Instantiate the neural architectures defined in B2 and B3.
        self.action_model = LSTMPredictor(input_size=len(FEATURE_COLS)).to(device)
        self.intent_model = IntentClassifier(input_size=len(FEATURE_COLS)).to(device)

        # Define the file mapping for our high-fidelity trained weights.
        paths = {
            "action": os.path.join(checkpoint_dir, "action_predictor_best.pt"),
            "intent": os.path.join(checkpoint_dir, "intent_classifier_best.pt")
        }

        # Attempt to load established intelligence into the initialized brains.
        for key, path in paths.items():
            if os.path.exists(path):
                # Target the correct model branch for weight assignment.
                model = self.action_model if key == "action" else self.intent_model
                # Load the state dictionary from the drive; map specifically to the current device.
                model.load_state_dict(torch.load(path, map_location=device))
                # CRITICAL: Switch to 'eval' mode. This disables dropout, ensuring deterministic predictions.
                model.eval()
                print(f"  [Oracle] Loaded {key} model from {path}")
            else:
                # Failsafe if the environment is fresh or checkpoints were moved.
                print(f"  [Oracle] WARNING: Checkpoint {path} not found. Using raw weights.")

        # Initialize the Heatmap Generator to translate action probabilities into spatial visualizations.
        self.heatmap_gen = HeatmapGenerator(grid_size=grid_size)

    # Conversion utility to ensure live environment data matches the model's expected 0.0 - 1.0 range.
    def _normalize_row(self, row_dict):
        """Maps raw environment units (HP, coordinates, etc.) into the normalized [0, 1] feature space."""
        vec = []
        # Sort data exactly according to the FEATURE_COLS list order used during training.
        for col in FEATURE_COLS:
            # Extract the raw float value, defaulting to zero if the key is missing in the state dict.
            val = float(row_dict.get(col, 0))
            # Retrieve the min/max bounds established during the dataset creation phase.
            lo, hi = NORM_RANGES.get(col, (0, 1))
            # Perform min-max scaling; handle zero-range columns to avoid division by zero.
            norm_val = (val - lo) / (hi - lo) if hi != lo else 0.0
            vec.append(norm_val)
        # Return a float32 numpy array ready for tensor conversion.
        return np.array(vec, dtype=np.float32)

    # The main runtime update method that processes a single environment tick.
    def update(self, current_state_dict):
        """
        Ingests the current grid state and returns a comprehensive intelligence package.
        
        Args:
            current_state_dict: Dictionary containing raw keys like 'red_x', 'red_hp', etc.
        
        Returns:
            A dictionary containing action probabilities, strategic posture, and spatial heatmap.
        """
        # Step 1: Preprocess the incoming observation and push it into our temporal memory.
        vec = self._normalize_row(current_state_dict)
        self.history.append(vec)

        # Step 2: Buffer check. The LSTM requires a full window of history before it can make an accurate prediction.
        if len(self.history) < self.window_size:
            # Return None until the 'history' buffer is warm.
            return None

        # Step 3: Fast Tensor Preparation. We convert the history list to a 3D tensor (1, Window, Features).
        input_array = np.array(list(self.history), dtype=np.float32)
        # unsqeeze(0) adds the 'Batch' dimension which PyTorch models expect.
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(self.device)

        # Step 4: Multi-Branch Inference.
        # Use 'torch.no_grad()' to prevent the GPU from tracking gradients, which saves massive amounts of RAM and speed.
        with torch.no_grad():
            # Run the Action Predictor and compute Softmax to turn raw logits into 0-1 probabilities.
            action_logits = self.action_model(input_tensor)
            action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
            
            # Run the Intent Classifier and compute Softmax to identify the most likely strategic posture.
            intent_logits = self.intent_model(input_tensor)
            posture_probs = torch.softmax(intent_logits, dim=-1).cpu().numpy()[0]

        # Identify the high-level intent by selecting the class with the maximum probability.
        posture_idx = np.argmax(posture_probs)
        posture_name = POSTURE_CLASSES[posture_idx]
        # Record our confidence level (probability of the winning class).
        confidence = float(posture_probs[posture_idx])

        # Step 5: Spatial Analysis. Translate the 'Action Probabilities' into a move-prediction heatmap.
        # We use the current unit coordinate as the 'center' for the probability diffusion.
        curr_pos = (int(current_state_dict['red_x']), int(current_state_dict['red_y']))
        
        # The Heatmap Generator uses the forecast to highlight the tiles the enemy is statistically likely to move toward.
        heatmap = self.heatmap_gen.from_action_probs(action_probs, curr_pos, n_steps=3)

        # Return the finalized Intelligence Report.
        return {
            "action_probs": action_probs,
            "posture": posture_name,
            "confidence": confidence,
            "heatmap": heatmap,
            # The single most likely discrete action (0-5) the model expects next.
            "predicted_action": int(np.argmax(action_probs))
        }

    # Utility to wipe the engine's memory between different simulation matches.
    def reset(self):
        """Clears the historical rolling buffer to prevent state bleeding between separate matches."""
        self.history.clear()
