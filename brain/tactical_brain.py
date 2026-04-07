# File docstring for the primary tactical decision-making orchestrator.
"""
# Header for the tactical brain file.
tactical_brain.py — v14.0 (Stable Decision Architecture)
# Logical separator.
======================================================
# High-level description of class responsibilities.
Orchestrates Oracle Intelligence and Feasibility Analysis.
# Performance note regarding recurrence updates.
Ensures Oracle memory is updated exactly ONCE per turn.
# End of documentation block.
"""
# Empty line.

# Import numpy for final action scoring and argmax operations.
import numpy as np
# Import torch for AI inference calculations (LSTM/Classifiers).
import torch
# Import the centralized Oracle intelligence wrapper.
from oracle.inference_engine import OracleInferenceEngine
# Import the movement-cost and tactical-risk analysis engine.
from feasibility.feasibility_engine import FeasibilityEngine
# Empty line.

# Class definition for the simulation's top-level decision maker.
class TacticalBrain:
# Initialization for the brain's internal components.
    def __init__(self):
# Section comment for identifying logical subgroups.
        # Initialize internal engines
# Instantiate the Oracle on CPU for stable inference.
        self.oracle = OracleInferenceEngine(device="cpu") 
# Instantiate the Feasibility engine for pathing and risk checks.
        self.feasibility = FeasibilityEngine()
# Placeholder for the most recent AI prediction state.
        self.last_prediction = None
# Empty line.

# Core method to determine unit behavior.
    def get_smart_action(self, unit_id, env):
# Docstring.
        """Calculates the best tactical action for a single Blue unit."""
# Identification step.
        # 1. Identification
# Find the specific blue unit object from the environment lists.
        blue_unit = next((u for u in env.blue_units if u['id'] == unit_id), None)
# Existence and health check.
        if not blue_unit or not blue_unit['alive']:
# Default to "Stay" if unit is inactive.
            return 0 # Do nothing if dead or missing
# Empty line.

# Prediction step.
        # 2. Stable Oracle Intelligence (Predict Once)
# Find a target for the Oracle to analyze.
        # We look for the first living Red unit to track as the primary threat
        red_unit = next((u for u in env.red_units if u['alive']), None)
# Default blank risk map.
        h_map = np.zeros((10, 10))
# Empty line.
        
# Check if a target exists before running inference.
        if red_unit:
# Package current tactical features into a dictionary for the Oracle.
            state_dict = {
# Target horizontal coordinate.
                "red_x": red_unit['pos'][1], "red_y": red_unit['pos'][0],
# Self horizontal coordinate.
                "blue_x": blue_unit['pos'][1], "blue_y": blue_unit['pos'][0],
# Resource levels.
                "red_hp": red_unit['hp'], "red_ammo": red_unit['ammo'], "red_fuel": red_unit['fuel'],
# Target psychological state.
                "red_morale": red_unit['morale'],
# Spatial advantage check.
                "red_elevation": env.elevation[red_unit['pos'][0], red_unit['pos'][1]],
# Distance metric (Manhattan).
                "distance": abs(red_unit['pos'][0] - blue_unit['pos'][0]) + abs(red_unit['pos'][1] - blue_unit['pos'][1]),
# Visibility check for AI state.
                "fog_visible": 1 if env.fog_map[red_unit['pos'][0], red_unit['pos'][1]] else 0,
# Recursive action history.
                "red_prev_action": red_unit.get('prev_action', 0)
# End of state dict.
            }
# Critical memory update: Feeds the new state into the Oracle LSTM.
            # The Oracle memory is updated here
            self.last_prediction = self.oracle.update(state_dict)
# Check for successful prediction before using heatmap.
            if self.last_prediction:
# Extract spatial risk probabilities.
                h_map = self.last_prediction['heatmap']
# Empty line.

# Evaluation step.
        # 3. Action Evaluation Loop
# Initialize score list for the 6 possible actions.
        action_scores = []
# Bundle environment features for the feasibility engine.
        env_state = {'terrain': env.terrain_map, 'elevation': env.elevation, 'fog': env.fog_map}
# Empty line.
        
# Loop through 0 (Stay) to 5 (Ranged Attack).
        for action in range(6):
# Predictive positioning logic.
            # Simulate "What-If" position
            tr, tc = blue_unit['pos']
# Upward move simulation.
            if action == 1: tr -= 1
# Downward move simulation.
            elif action == 2: tr += 1
# Left move simulation.
            elif action == 3: tc -= 1
# Right move simulation.
            elif action == 4: tc += 1
# Empty line.
            
# Check if the simulated move is off the grid.
            # Boundary security
            if not (0 <= tr < 10 and 0 <= tc < 10):
# Heavily penalize invalid moves.
                action_scores.append(-999.0)
                continue
# Empty line.

# Call external feasibility engine for base move score.
            # FEASIBILITY ENGINE: Checks terrain costs, fuel, and Oracle risk
            f_results = self.feasibility.get_maneuver_score(blue_unit, (tr, tc), env_state, h_map)
# Empty line.
            
# Tactical decision weighing.
            # STRATEGIC BIAS:
            strategic_bonus = 0
# Logic for attacking.
            if action == 5:
# Combat target search.
                # Combat range check (all Red units)
                in_range = False
# Scan enemy team.
                for r_u in env.red_units:
# Distance check.
                    if r_u['alive']:
                        dist = abs(blue_unit['pos'][0] - r_u['pos'][0]) + abs(blue_unit['pos'][1] - r_u['pos'][1])
# Check against weapon range.
                        if dist <= 3:
                            in_range = True
                            break
# Apply reward for valid combat opportunity.
                if in_range and blue_unit['ammo'] > 0:
                    strategic_bonus += 15.0 # Highest Priority: Engagement
# Apply penalty for empty weapon.
                elif blue_unit['ammo'] <= 0:
                    strategic_bonus -= 20.0 # Refuse to fire without ammo
# Logic for positioning.
            else:
# Bias towards center/enemy.
                # Movement bias toward the suspected enemy position
                if red_unit:
# Calculate potential new distance.
                    new_dist = abs(tr - red_unit['pos'][0]) + abs(tc - red_unit['pos'][1])
# Calculate current distance.
                    curr_dist = abs(blue_unit['pos'][0] - red_unit['pos'][0]) + abs(blue_unit['pos'][1] - red_unit['pos'][1])
# Reward closing the distance.
                    if new_dist < curr_dist:
                        strategic_bonus += 0.5 # Bias toward aggression
# Reward road movement efficiency.
                    if env.terrain_map[tr, tc] == 4: # Road bonus
                        strategic_bonus += 0.2
# Empty line.

# Combine base score and strategic bonuses.
            final_score = f_results['overall'] + strategic_bonus
# Append to final list.
            action_scores.append(final_score)
# Empty line.

# Resolution.
        # 4. Return the Winner
# Return the index of the highest scoring action.
        return int(np.argmax(action_scores))
# Empty line.

# Housekeeping method for mission restart.
    def reset(self):
# Docstring.
        """Full reset for a new match."""
# Wipe Oracle LSTM memory.
        self.oracle.reset()
# Wipe last prediction pointer.
        self.last_prediction = None
