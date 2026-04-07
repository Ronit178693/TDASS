# File docstring for the secondary tactical logic engine.
"""
# Header for the feasibility engine file.
feasibility_engine.py — Phase E: Tactical Move Validation
# Visual separator.
=========================================================
# Functional summary of the engine's purpose.
Calculates the viability of proposed tactical actions based on:
# Logistics criteria.
  1. Logistics (Fuel/Ammo)
# Physical criteria.
  2. Physics (Terrain/Elevation/Passability)
# Oracle risk criteria.
  3. Risk (Predicted Enemy Threat from Oracle)
# End block.
"""
# Empty line.

# Import numpy for grid array manipulation and distance calculations.
import numpy as np
# Empty line.

# Class definition for the tactical risk and pathing analyzer.
class FeasibilityEngine:
# Docstring.
    """
    Evaluates the 'Calculated Risk' of proposed tactical maneuvers.
    """
# Empty line.
    
# Constructor to set up cost tables and grid limits.
    def __init__(self, grid_size=10):
# Save dimension scaling.
        self.grid_size = grid_size
# Define fuel consumption per terrain type.
        # Energy cost per terrain type
        self.terrain_costs = {
# Standard plains cost.
            0: 1.0,  # Plains
# Blocked movement (Infinite cost).
            1: 99.0, # Wall (Impassable)
# High friction forest cost.
            2: 2.0,  # Forest
# High cost water cost.
            3: 5.0,  # Water (Difficult)
# Low friction road cost.
            4: 0.5,  # Road (Efficient)
# Complex urban cost.
            5: 1.5   # Urban
# End of costs.
        }
# Empty line.
        
# Method to determine if a unit has enough fuel for a move.
    def calculate_path_cost(self, start_pos, end_pos, terrain_map, fuel_current):
# Docstring.
        """
        Check if a move is logistically possible.
        Uses Manhattan distance as a base, adjusted by terrain.
        """
# Extract start coordinates.
        r1, c1 = start_pos
# Extract end coordinates.
        r2, c2 = end_pos
# Empty line.
        
# Calculate the raw distance between points.
        # Distance
        dist = abs(r1 - r2) + abs(c1 - c2)
# Check for null movement.
        if dist == 0: return 1.0, 0.0 # Feasible
# Empty line.
        
# Calculate estimated fuel usage.
        # Estimate cost (Simple linear path estimate)
# Identify terrain type halfway between points for cost estimation.
        avg_terrain_type = terrain_map[int((r1+r2)/2), int((c1+c2)/2)]
# Pull cost code from dictionary.
        cost_per_tile = self.terrain_costs.get(int(avg_terrain_type), 1.0)
# Final fuel requirement.
        total_cost = dist * cost_per_tile
# Empty line.
        
# Calculate percentage of feasibility based on current fuel.
        feasibility = max(0, 1.0 - (total_cost / (fuel_current + 1e-6)))
# Return fractional feasibility and total fuel cost.
        return feasibility, total_cost
# Empty line.

# Method to cross-reference Oracle heatmaps with proposed coordinates.
    def evaluate_combat_risk(self, unit_pos, target_pos, red_heatmap, unit_hp):
# Docstring.
        """
        Check if moving to target_pos is a 'Death Trap'.
        Uses the Oracle's Heatmap (probability of enemies being there).
        """
# Extract target coordinates.
        r, c = target_pos
# Aggregated peril score.
        # Enemy threat is the sum of probabilities in a 3x3 around the target
        threat_level = 0
# Perform spatial kernel scan.
        for dr in range(-1, 2):
# Column scan.
            for dc in range(-1, 2):
# Calculate neighbor coord.
                nr, nc = r + dr, c + dc
# Bound check.
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
# Add Oracle probability to threat level.
                    threat_level += red_heatmap[nr, nc]
# Empty line.
        
# Cap risk at 100%.
        # Normalize (threat_level of 1.0 means an enemy is definitely there)
        risk = min(1.0, threat_level)
# Empty line.
        
# Calculate final survival probability based on health and risk.
        # Survival is high if HP is high and risk is low
        survival_score = (unit_hp / 100.0) * (1.0 - risk)
# Return score and raw risk level.
        return survival_score, risk
# Empty line.

# Main method to aggregate all tactical factors into a single score.
    def get_maneuver_score(self, unit, target_pos, env_state, red_heatmap):
# Docstring.
        """
        Aggregate Feasibility Score (0% to 100%).
        """
# Logistics evaluation.
        # 1. Logistics check
        f_score, _ = self.calculate_path_cost(unit['pos'], target_pos, env_state['terrain'], unit['fuel'])
# Empty line.
        
# Safety evaluation.
        # 2. Safety check
        s_score, risk = self.evaluate_combat_risk(unit['pos'], target_pos, red_heatmap, unit['hp'])
# Empty line.
        
# Spatial advantage evaluation.
        # 3. Elevation check (moving Upwards is harder)
# Current height.
        curr_elev = env_state['elevation'][unit['pos'][0], unit['pos'][1]]
# Proposed height.
        target_elev = env_state['elevation'][target_pos[0], target_pos[1]]
# Apply 20% penalty if the move is an uphill climb.
        elev_penalty = 1.0 if target_elev <= curr_elev else 0.8 # 20% harder to move up
# Empty line.
        
# Final calculation.
        # Final Weighted Score
# Weighted sum: 40% Logistics, 50% Survival, 10% Elevation.
        total_score = (f_score * 0.4) + (s_score * 0.5) + (elev_penalty * 0.1)
# Empty line.
        
# Package all metrics for the Tactical Brain.
        return {
# Final decision score.
            "overall": total_score,
# Raw fuel viability.
            "logistics": f_score,
# Raw survival score.
            "survival": s_score,
# Raw Oracle threat score.
            "risk": risk
# End dict.
        }
