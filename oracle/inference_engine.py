"""
inference_engine.py — B6: Real-time Intelligence Integration
=============================================================
A clean wrapper to load trained Oracle models and run inference
during live simulations/visualizations.

Usage:
  engine = OracleInferenceEngine()
  prediction = engine.update(current_state_dict)
"""

import os
import torch
import numpy as np
from collections import deque

from oracle.lstm_predictor import LSTMPredictor
from oracle.intent_classifier import IntentClassifier
from oracle.heatmap_generator import HeatmapGenerator
from oracle.data_loader import FEATURE_COLS, NORM_RANGES, POSTURE_CLASSES


class OracleInferenceEngine:
    """
    Stateful inference engine for the TDSS Oracle.
    Maintains a rolling history of the battlefield to run LSTM calls.
    """

    def __init__(
        self,
        checkpoint_dir="oracle/checkpoints",
        window_size=10,
        device="cpu",
        grid_size=10
    ):
        self.device = device
        self.window_size = window_size
        self.grid_size = grid_size
        
        # rolling buffer for features: deque of length window_size
        self.history = deque(maxlen=window_size)

        # Load Models
        self.action_model = LSTMPredictor(input_size=len(FEATURE_COLS)).to(device)
        self.intent_model = IntentClassifier(input_size=len(FEATURE_COLS)).to(device)
        
        paths = {
            "action": os.path.join(checkpoint_dir, "action_predictor_best.pt"),
            "intent": os.path.join(checkpoint_dir, "intent_classifier_best.pt")
        }

        for key, path in paths.items():
            if os.path.exists(path):
                model = self.action_model if key == "action" else self.intent_model
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                print(f"  [Oracle] Loaded {key} model from {path}")
            else:
                print(f"  [Oracle] WARNING: Checkpoint {path} not found. Using raw weights.")

        # Heatmap Generator
        self.heatmap_gen = HeatmapGenerator(grid_size=grid_size)

    def _normalize_row(self, row_dict):
        """Convert state dict to normalized feature vector."""
        vec = []
        for col in FEATURE_COLS:
            val = float(row_dict.get(col, 0))
            lo, hi = NORM_RANGES.get(col, (0, 1))
            norm_val = (val - lo) / (hi - lo) if hi != lo else 0.0
            vec.append(norm_val)
        return np.array(vec, dtype=np.float32)

    def update(self, current_state_dict):
        """
        Update history and run inference.
        
        Args:
            current_state_dict: dict containing keys in FEATURE_COLS
        
        Returns:
            dict containing action_probs, posture, confidence, and heatmap.
        """
        # 1. Normalize and append to history
        vec = self._normalize_row(current_state_dict)
        self.history.append(vec)

        # check if we have enough history
        if len(self.history) < self.window_size:
            return None

        # 2. Optimized Tensor Conversion (Prevents simulation freezing)
        input_array = np.array(list(self.history), dtype=np.float32)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(self.device)

        # 3. Predict Action & Intent
        with torch.no_grad():
            action_probs = torch.softmax(self.action_model(input_tensor), dim=-1).cpu().numpy()[0]
            posture_probs = torch.softmax(self.intent_model(input_tensor), dim=-1).cpu().numpy()[0]

        posture_idx = np.argmax(posture_probs)
        posture_name = POSTURE_CLASSES[posture_idx]
        confidence = float(posture_probs[posture_idx])

        # 4. Generate Heatmap (Monte Carlo or Diffusion)
        # Using current position from state_dict
        curr_pos = (int(current_state_dict['red_x']), int(current_state_dict['red_y']))
        
        # For real-time, we use action bits + diffusion for speed
        heatmap = self.heatmap_gen.from_action_probs(action_probs, curr_pos, n_steps=3)

        return {
            "action_probs": action_probs,
            "posture": posture_name,
            "confidence": confidence,
            "heatmap": heatmap,
            "predicted_action": int(np.argmax(action_probs))
        }

    def reset(self):
        """Clear history (call this at the start of a new match)."""
        self.history.clear()
