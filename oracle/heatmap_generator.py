# Spatial probability forecasting engine for visualizing tactical risk.
"""
# This module translates high-level action probabilities into a concrete 2D spatial distribution.
heatmap_generator.py — B4: Probability Heatmap Generator
==========================================================
# It calculates where a specific Red unit is statistically most likely to be located in future frames.
# This serves as the 'Intuition' layer, allowing human operators to visualize enemy movement intentions.

Available Forecasting Methodologies:
# 1. Recursive Monte Carlo: Uses the LSTM in a closed-loop to simulate N future deployment timelines.
  - Accurate but computationally intensive (parallel logic paths).
# 2. Gaussian Diffusion: Spreads probabilities along movement vectors using iterative blurring.
  - Runtime-optimized (fast approximation) for real-time visualization.
"""

# NumPy for high-performance grid manipulation and probability normalization.
import numpy as np
# PyTorch for executing model rollouts during the Monte Carlo phase.
import torch
# SciPy-based Gaussian filter for approximating movement diffusion.
from scipy.ndimage import gaussian_filter

# Mapping Action IDs to spatial coordinate offsets.
ACTION_DELTAS = {
    0: (0,  0),   # Stay
    1: (-1, 0),   # Up
    2: (1,  0),   # Down
    3: (0, -1),   # Left
    4: (0,  1),   # Right
    5: (0,  0),   # RangedAttack (Unit remains stationary while firing)
}

# The primary class for generating and visualizing spatial risk.
class HeatmapGenerator:
    """
    Translates model-level action predictions into human-interpretable risk maps.
    Enforces terrain constraints to ensure probability doesn't leak into impassable zones.
    """

    def __init__(self, grid_size=10, terrain_map=None):
        """
        Initializes the spatial context of the battlefield.
        Args:
            grid_size: Physical dimensions of the square grid.
            terrain_map: Optional grid of terrain IDs for impassable cell masking.
        """
        self.grid_size = grid_size
        self.terrain_map = terrain_map
        # Identify codes that physically block unit movement (Wall=1, Water=3).
        self.impassable = {1, 3} 

    # Internal validator to ensure predictions remain within tactical boundaries.
    def _is_passable(self, r, c):
        """Returns True if the coordinate is within bounds and not a physical obstacle."""
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        if self.terrain_map is not None:
            # Check against the provided terrain matrix.
            return int(self.terrain_map[r, c]) not in self.impassable
        return True

    # ── Algorithm A: Recursive Monte Carlo Simulation ──
    def from_monte_carlo(self, lstm_model, current_sequence, current_pos,
                          n_steps=5, n_simulations=200, device="cpu"):
        """
        Performs iterative rollouts using the LSTM to sample a distribution of future locations.
        
        Args:
            lstm_model: The trained LSTMPredictor (Branch B2).
            current_sequence: The last 10 steps of feature engineering.
            current_pos: Initial (row, col) at T=0.
            n_steps: Temporal horizon (depth of the future prediction).
            n_simulations: Width of the search (number of parallel 'futures' to sample).
        """
        # Ensure model is in evaluation mode to disable stochastic dropout during simulation.
        lstm_model.eval()
        # Initialize an empty intensity grid to aggregate final landing positions.
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        # Pre-process the sequence for device-specific acceleration.
        seq = current_sequence.clone().to(device)

        # Execute 'n_simulations' separate future histories.
        for _ in range(n_simulations):
            r, c = current_pos
            # Create a localized memory copy for this specific 'future' timeline.
            sim_seq = seq.clone()

            # Iterate through the temporal horizon.
            for step in range(n_steps):
                # Inference Pass: Ask the brain what it would do in this simulated state.
                with torch.no_grad():
                    logits = lstm_model(sim_seq)
                    # Convert raw logits to a probability distribution across the 6 actions.
                    probs = torch.softmax(logits, dim=-1).squeeze()

                # Step Select: Sample one action based on the model's confidence distribution.
                # This ensures we capture the 'full spread' of possibilities across all simulations.
                action = torch.multinomial(probs, 1).item()
                dr, dc = ACTION_DELTAS.get(action, (0, 0))
                nr, nc = r + dr, c + dc

                # Valid Move Verification.
                if self._is_passable(nr, nc):
                    r, c = nr, nc

                # Temporal Loopback: Update the 'memory' of the unit with the simulated movement.
                # Here we manually compute the normalized features to feed back into the LSTM.
                new_feat = sim_seq[0, -1, :].clone()
                # Update coordinate features [0,1].
                new_feat[0] = r / (self.grid_size - 1)
                new_feat[1] = c / (self.grid_size - 1)
                # Update action history bit.
                new_feat[-1] = action / 5.0 
                # Shift the LSTM window: Drop oldest step, append the new simulated step.
                sim_seq = torch.cat([sim_seq[:, 1:, :],
                                     new_feat.unsqueeze(0).unsqueeze(0)], dim=1)

            # Record where the unit was located at the end of this simulated timeline.
            heatmap[r, c] += 1.0

        # Normalize the grid so the sum of all cell values equals 1.0 (True Probability).
        total = heatmap.sum()
        if total > 0:
            heatmap /= total
        return heatmap

    # ── Algorithm B: Vector-based Gaussian Diffusion ──
    def from_action_probs(self, action_probs, current_pos, n_steps=5, sigma=1.0):
        """
        A high-speed approximation of future risk using immediate action confidence and blurring.
        
        Args:
            action_probs: Current frame Action Predictor output (softmax).
            current_pos: Initial coordinate.
            n_steps: Future projection depth.
            sigma: Kernel size for the diffusion blur.
        """
        # Start with an empty grid.
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        # Iterate over all possible movement vectors.
        for action, (dr, dc) in ACTION_DELTAS.items():
            # Get the model's confidence in this specific direction.
            prob = action_probs[action] if action < len(action_probs) else 0.0
            # Project the displacement across the requested temporal horizon.
            for step in range(1, n_steps + 1):
                nr = int(current_pos[0] + dr * step)
                nc = int(current_pos[1] + dc * step)
                if self._is_passable(nr, nc):
                    # Distribute probability along the movement vector.
                    heatmap[nr, nc] += prob / n_steps

        # Apply Gaussian filtering to simulate the 'spread' of uncertainty over time.
        # Diffusion effectively widens the risk zone as we look further into the future.
        heatmap = gaussian_filter(heatmap, sigma=sigma * np.sqrt(n_steps))

        # Re-apply terrain masks: Zero out probability in cells where tiles are impassable.
        if self.terrain_map is not None:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if int(self.terrain_map[r, c]) in self.impassable:
                        heatmap[r, c] = 0.0

        # Final normalization to ensure a valid probability density function.
        total = heatmap.sum()
        if total > 0:
            heatmap /= total
        else:
            # Failsafe: If all paths are blocked, probability remains at current position.
            heatmap[current_pos[0], current_pos[1]] = 1.0
        return heatmap

    # Utility for verifying heatmap quality via terminal output.
    def render_ascii(self, heatmap, title="Tactical Risk Heatmap"):
        """Prints a colored ASCII grid where intensity represents enemy presence probability."""
        print(f"\n── {title} ──")
        print("    " + " ".join(f"{c:>4}" for c in range(self.grid_size)))
        for r in range(self.grid_size):
            row_str = f"{r:>2}  "
            for c in range(self.grid_size):
                val = heatmap[r, c]
                # Dynamic Threshold Coloring:
                if val >= 0.15:   # High Risk (Red)
                    row_str += f"\033[91m{val:4.2f}\033[0m "
                elif val >= 0.05: # Moderate Risk (Yellow)
                    row_str += f"\033[93m{val:4.2f}\033[0m "
                elif val > 0.01:  # Low Risk (Green)
                    row_str += f"\033[92m{val:4.2f}\033[0m "
                else:             # Negligible (Grey)
                    row_str += f"{val:4.2f} "
            print(row_str)
