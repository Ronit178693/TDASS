"""
heatmap_generator.py — B4: Probability Heatmap Generator
==========================================================
Generates a 10×10 probability grid predicting where a Red unit
is most likely to be in N future steps, given its current state
and the trained LSTM action predictor.

Methods:
  1. Monte Carlo rollout — simulate N steps using the LSTM's
     predicted action distribution, averaging positions.
  2. Gaussian diffusion — fast approximation using movement
     direction probabilities from the LSTM.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# Movement deltas: action -> (dr, dc)
ACTION_DELTAS = {
    0: (0,  0),   # Stay
    1: (-1, 0),   # Up
    2: (1,  0),   # Down
    3: (0, -1),   # Left
    4: (0,  1),   # Right
    5: (0,  0),   # RangedAttack (no movement)
}


class HeatmapGenerator:
    """
    Generate probability heatmaps for enemy future positions.

    Args:
        grid_size:    battlefield grid dimension (default 10)
        terrain_map:  numpy array of terrain codes (for impassable checks)
    """

    def __init__(self, grid_size=10, terrain_map=None):
        self.grid_size = grid_size
        self.terrain_map = terrain_map
        # Impassable terrain codes
        self.impassable = {1, 3}  # Wall, Water

    def _is_passable(self, r, c):
        """Check if a cell is passable."""
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        if self.terrain_map is not None:
            return int(self.terrain_map[r, c]) not in self.impassable
        return True

    def from_monte_carlo(self, lstm_model, current_sequence, current_pos,
                          n_steps=5, n_simulations=200, device="cpu"):
        """
        Monte Carlo rollout using LSTM action predictions.

        Args:
            lstm_model:        trained LSTMPredictor
            current_sequence:  tensor (1, seq_len, features) — recent history
            current_pos:       (row, col) — current Red position
            n_steps:           how many steps into the future to predict
            n_simulations:     number of Monte Carlo samples
            device:            torch device

        Returns:
            heatmap: (grid_size, grid_size) numpy array of probabilities
        """
        lstm_model.eval()
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        seq = current_sequence.clone().to(device)

        for _ in range(n_simulations):
            r, c = current_pos
            sim_seq = seq.clone()

            for step in range(n_steps):
                with torch.no_grad():
                    logits = lstm_model(sim_seq)
                    probs = torch.softmax(logits, dim=-1).squeeze()

                # Sample action from distribution
                action = torch.multinomial(probs, 1).item()
                dr, dc = ACTION_DELTAS.get(action, (0, 0))
                nr, nc = r + dr, c + dc

                if self._is_passable(nr, nc):
                    r, c = nr, nc

                # Update sequence (shift window, append new pseudo-features)
                # This is an approximation — we shift the window and repeat
                # the last feature vector with updated position
                new_feat = sim_seq[0, -1, :].clone()
                # Update position features (indices 0,1 = red_x, red_y normalized)
                new_feat[0] = r / (self.grid_size - 1)
                new_feat[1] = c / (self.grid_size - 1)
                new_feat[-1] = action / 5.0  # red_prev_action normalized
                sim_seq = torch.cat([sim_seq[:, 1:, :],
                                     new_feat.unsqueeze(0).unsqueeze(0)], dim=1)

            # Record final position
            heatmap[r, c] += 1.0

        # Normalize to probability distribution
        total = heatmap.sum()
        if total > 0:
            heatmap /= total

        return heatmap

    def from_action_probs(self, action_probs, current_pos, n_steps=5, sigma=1.0):
        """
        Fast Gaussian diffusion approximation using action probabilities.

        Args:
            action_probs: numpy array (6,) — probability of each action
            current_pos:  (row, col)
            n_steps:      prediction horizon
            sigma:        Gaussian blur sigma for diffusion

        Returns:
            heatmap: (grid_size, grid_size) numpy array of probabilities
        """
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        # Compute expected displacement per step
        for action, (dr, dc) in ACTION_DELTAS.items():
            prob = action_probs[action] if action < len(action_probs) else 0.0
            for step in range(1, n_steps + 1):
                nr = int(current_pos[0] + dr * step)
                nc = int(current_pos[1] + dc * step)
                if self._is_passable(nr, nc):
                    heatmap[nr, nc] += prob / n_steps

        # Apply Gaussian blur for diffusion
        heatmap = gaussian_filter(heatmap, sigma=sigma * np.sqrt(n_steps))

        # Mask impassable cells
        if self.terrain_map is not None:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if int(self.terrain_map[r, c]) in self.impassable:
                        heatmap[r, c] = 0.0

        # Normalize
        total = heatmap.sum()
        if total > 0:
            heatmap /= total
        else:
            # Fallback: uniform over current position
            heatmap[current_pos[0], current_pos[1]] = 1.0

        return heatmap

    def render_ascii(self, heatmap, title="Probability Heatmap"):
        """Print a text visualization of the heatmap."""
        print(f"\n── {title} ──")
        print("    " + " ".join(f"{c:>4}" for c in range(self.grid_size)))

        for r in range(self.grid_size):
            row_str = f"{r:>2}  "
            for c in range(self.grid_size):
                val = heatmap[r, c]
                if val >= 0.15:
                    row_str += f"\033[91m{val:4.2f}\033[0m "
                elif val >= 0.05:
                    row_str += f"\033[93m{val:4.2f}\033[0m "
                elif val > 0.01:
                    row_str += f"\033[92m{val:4.2f}\033[0m "
                else:
                    row_str += f"{val:4.2f} "
            print(row_str)
