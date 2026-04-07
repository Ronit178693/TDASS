# Docstring for the heatmap generator module.
"""
# Source file name and module purpose.
heatmap_generator.py — B4: Probability Heatmap Generator
# Visual separator for documentation.
==========================================================
# Functional description of 10×10 grid prediction.
Generates a 10×10 probability grid predicting where a Red unit
# Context on predicting temporal future steps based on LSTM analysis.
is most likely to be in N future steps, given its current state
# Reference to using the trained LSTM Action Predictor.
and the trained LSTM action predictor.
# Empty line.

# List of the two available calculation methods.
Methods:
# Method 1: Detailed simulation based on the LSTM's probability distribution.
  1. Monte Carlo rollout — simulate N steps using the LSTM's
# Description of step-averaging in method 1.
     predicted action distribution, averaging positions.
# Method 2: Fast mathematical approximation through Gaussian blurring.
  2. Gaussian diffusion — fast approximation using movement
# Explanation of mapping movement directions in method 2.
     direction probabilities from the LSTM.
# Closing docstring.
"""
# Empty line.

# Linear algebra and array math library.
import numpy as np
# Machine learning framework for prediction calls.
import torch
# Mathematical image/grid processing for the diffusion blur effect.
from scipy.ndimage import gaussian_filter
# Empty line.


# Transformation map: maps action ID to coordinate offsets.
# Movement deltas: action -> (dr, dc)
ACTION_DELTAS = {
# Action 0: Stationary.
    0: (0,  0),   # Stay
# Action 1: Upward move.
    1: (-1, 0),   # Up
# Action 2: Downward move.
    2: (1,  0),   # Down
# Action 3: Leftward move.
    3: (0, -1),   # Left
# Action 4: Rightward move.
    4: (0,  1),   # Right
# Action 5: Attack move (Stay-equivalent).
    5: (0,  0),   # RangedAttack (no movement)
# Closing map bracket.
}
# Empty line.


# Class containing logic for spatial probability forecasting.
class HeatmapGenerator:
# Docstring block detailing engine purpose.
    """
# Generates heatmaps representing future foe locations.
    Generate probability heatmaps for enemy future positions.
# Empty line.

# Documentation for initialization parameters.
    Args:
# The square dimensions of the field (defaulting to 10).
        grid_size:    battlefield grid dimension (default 10)
# A reference grid to verify if a tile allows movement.
        terrain_map:  numpy array of terrain codes (for impassable checks)
# Closing docstring.
    """
# Empty line.

# Initialization block to establish field constraints.
    def __init__(self, grid_size=10, terrain_map=None):
# Store the limit dimension.
        self.grid_size = grid_size
# Store the terrain reference array.
        self.terrain_map = terrain_map
# Define an immutable set of codes that block movement.
        # Impassable terrain codes
        self.impassable = {1, 3}  # Wall, Water
# Empty line.

# Logic helper to determine if a specific tile is accessible.
    def _is_passable(self, r, c):
# Docstring.
        """Check if a cell is passable."""
# Ensure the coordinates are within the 10×10 bounds.
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
# Return false if out of bounds.
            return False
# Verify against the terrain map if one exists.
        if self.terrain_map is not None:
# Return true only if the terrain code is not in the forbidden set.
            return int(self.terrain_map[r, c]) not in self.impassable
# Default to true if no terrain restriction is present.
        return True
# Empty line.

# Method 1: Generates accurate probabilities by simulating 200 "Parallel Timelines".
    def from_monte_carlo(self, lstm_model, current_sequence, current_pos,
                          n_steps=5, n_simulations=200, device="cpu"):
# Docstring explaining the rollout technique.
        """
# Monte Carlo prediction utilizing the LSTM's action logic.
        Monte Carlo rollout using LSTM action predictions.
# Empty line.

# Parameter descriptions.
        Args:
# The trained predictor model.
            lstm_model:        trained LSTMPredictor
# The most recent 10 steps of data (tensor format).
            current_sequence:  tensor (1, seq_len, features) — recent history
# Current X/Y coordinate.
            current_pos:       (row, col) — current Red position
# Count of future steps to simulate.
            n_steps:           how many steps into the future to predict
# Precision of the average (higher = better quality, lower = faster).
            n_simulations:     number of Monte Carlo samples
# Device handling (CPU/GPU).
            device:            torch device
# Empty line.

# Return objects description.
        Returns:
# The probability grid.
            heatmap: (grid_size, grid_size) numpy array of probabilities
# Closing docstring.
        """
# Set model to evaluation mode for inference.
        lstm_model.eval()
# Create a flat 10×10 grid of zeros to record end positions.
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
# Empty line.

# Ensure the sequence history is loaded correctly onto the target device.
        seq = current_sequence.clone().to(device)
# Empty line.

# Execute the designated number of simulations.
        for _ in range(n_simulations):
# Reset initial position for each timeline trial.
            r, c = current_pos
# Clone the historical sequence to modify it during the "fake" timeline steps.
            sim_seq = seq.clone()
# Empty line.

# Step through the future N cycles.
            for step in range(n_steps):
# Disable gradient tracking for speed during inference.
                with torch.no_grad():
# Ask the Oracle which action is most likely.
                    logits = lstm_model(sim_seq)
# Convert prediction scores to a 0-1 probability curve.
                    probs = torch.softmax(logits, dim=-1).squeeze()
# Empty line.

# Procedural dice roll to pick one action based on the model's confidence.
                # Sample action from distribution
                action = torch.multinomial(probs, 1).item()
# Map that action to a direction change (e.g., Up, Down).
                dr, dc = ACTION_DELTAS.get(action, (0, 0))
# Calculate the newly predicted coordinate.
                nr, nc = r + dr, c + dc
# Empty line.

# Verify if the unit can actually walk into that new space.
                if self._is_passable(nr, nc):
# Update the current position in this timeline.
                    r, c = nr, nc
# Empty line.

# Recursive logic: update history window with our own "fake" predictions.
                # Update sequence (shift window, append new pseudo-features)
# Note on precision limitations for future reference.
                # This is an approximation — we shift the window and repeat
# Note on duplicating feature vectors for temporal consistency.
                # the last feature vector with updated position
                new_feat = sim_seq[0, -1, :].clone()
# Logic for normalizing coordinates before feeding back to the LSTM.
                # Update position features (indices 0,1 = red_x, red_y normalized)
                new_feat[0] = r / (self.grid_size - 1)
                new_feat[1] = c / (self.grid_size - 1)
# Correct the action history bit for the next recursive call.
                new_feat[-1] = action / 5.0  # red_prev_action normalized
# Discard the oldest sequence step and append this new "simulated" step.
                sim_seq = torch.cat([sim_seq[:, 1:, :],
                                     new_feat.unsqueeze(0).unsqueeze(0)], dim=1)
# Empty line.

# After N steps, mark where the unit ended up in this trial.
            # Record final position
            heatmap[r, c] += 1.0
# Empty line.

# Calculate the final probability spread (0 to 1 range).
        # Normalize to probability distribution
        total = heatmap.sum()
# Avoid divide-by-zero errors.
        if total > 0:
# Divide each cell count by total simulations.
            heatmap /= total
# Empty line.

# Return the resulting grid.
        return heatmap
# Empty line.

# Method 2: Fast, real-time map generation using broad statistical blurring.
    def from_action_probs(self, action_probs, current_pos, n_steps=5, sigma=1.0):
# Docstring block.
        """
# Gaussian Diffusion: mathematically spreads the unit starting from one point.
        Fast Gaussian diffusion approximation using action probabilities.
# Empty line.

# Argument descriptions.
        Args:
# Softmax probabilities array for next action.
            action_probs: numpy array (6,) — probability of each action
# Current tactical coordinate.
            current_pos:  (row, col)
# Temporal horizon.
            n_steps:      prediction horizon
# Blur intensity factor.
            sigma:        Gaussian blur sigma for diffusion
# Empty line.

# Return object description.
        Returns:
# The probability grid.
            heatmap: (grid_size, grid_size) numpy array of probabilities
# Closing docstring.
        """
# Create empty grid.
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
# Empty line.

# Spread probabilities based on expected distance per each action.
        # Compute expected displacement per step
        for action, (dr, dc) in ACTION_DELTAS.items():
# Extract probability weight for certain action index.
            prob = action_probs[action] if action < len(action_probs) else 0.0
# Iterate through each temporal step to project displacement.
            for step in range(1, n_steps + 1):
# Calculate future coordinate approximation.
                nr = int(current_pos[0] + dr * step)
                nc = int(current_pos[1] + dc * step)
# Verify tile legitimacy.
                if self._is_passable(nr, nc):
# Increment grid intensity based on LSTM probability.
                    heatmap[nr, nc] += prob / n_steps
# Empty line.

# Convert sharp point data into a smooth cloud of risk.
        # Apply Gaussian blur for diffusion
        heatmap = gaussian_filter(heatmap, sigma=sigma * np.sqrt(n_steps))
# Empty line.

# Visual cleanup: hide risk in areas where units cannot physically stand.
        # Mask impassable cells
        if self.terrain_map is not None:
# Iterate through rows.
            for r in range(self.grid_size):
# Iterate through columns.
                for c in range(self.grid_size):
# If cell is blocked (Wall/Water), clear its risk value.
                    if int(self.terrain_map[r, c]) in self.impassable:
                        heatmap[r, c] = 0.0
# Empty line.

# Final probability normalization.
        # Normalize
        total = heatmap.sum()
# Validation check.
        if total > 0:
# Divide by sum.
            heatmap /= total
# Fallback logic if the diffusion fails.
        else:
            # Fallback: uniform over current position
            heatmap[current_pos[0], current_pos[1]] = 1.0
# Empty line.

# Return the diffusion map.
        return heatmap
# Empty line.

# Debug utility to show the intensity in the terminal.
    def render_ascii(self, heatmap, title="Probability Heatmap"):
# Docstring.
        """Print a text visualization of the heatmap."""
# Header output.
        print(f"\n── {title} ──")
# Print coordinate numbers on top.
        print("    " + " ".join(f"{c:>4}" for c in range(self.grid_size)))
# Empty line.

# Loop through rows for output.
        for r in range(self.grid_size):
# Start row label.
            row_str = f"{r:>2}  "
# Loop through columns for character assembly.
            for c in range(self.grid_size):
# Extract local probability.
                val = heatmap[r, c]
# Red coloring for high risk.
                if val >= 0.15:
                    row_str += f"\033[91m{val:4.2f}\033[0m "
# Yellow coloring for medium risk.
                elif val >= 0.05:
                    row_str += f"\033[93m{val:4.2f}\033[0m "
# Green coloring for low risk.
                elif val > 0.01:
                    row_str += f"\033[92m{val:4.2f}\033[0m "
# Default grey for empty areas.
                else:
                    row_str += f"{val:4.2f} "
# Output the assembled row to the system console.
            print(row_str)
