# The primary decision-making orchestrator for autonomous friendly (Blue) units.
"""
# This module acts as the integration layer between perceptual intelligence (Oracle) and physical execution (Actions).
tactical_brain.py — v14.0 (Stable Decision Architecture)
======================================================
# It implements a 'Look-Ahead' decision policy that evaluates future state feasibility.
# Core Responsibilities:
# 1. State Synchronization: Ensures the Oracle's recursive memory is updated exactly once per simulation tick.
# 2. Risk Evaluation: Combines static terrain data with dynamic probability heatmaps to score maneuvers.
# 3. Decision Arbitration: Selects the winning action by weighing strategic goals (Engagement) against survival (Risk).
"""

# NumPy for efficient scoring vector manipulation and argmax selection.
import numpy as np
# PyTorch for interfacing with the underlying neural inference pipelines.
import torch
# The stateful Oracle wrapper for analyzing enemy intent.
from oracle.inference_engine import OracleInferenceEngine
# The spatial analysis engine for calculating movement costs and risk overlays.
from feasibility.feasibility_engine import FeasibilityEngine

class TacticalBrain:
    """
    Main controller for Blue units. 
    Implements a feasibility-weighted decision algorithm that respects the 'Oracle' vision.
    """

    def __init__(self):
        """Initializes the tactical engines and state buffers."""
        # Initialize Oracle on CPU: Localized inference is preferred for real-time control stability.
        self.oracle = OracleInferenceEngine(device="cpu") 
        # Initialize the Feasibility Engine for evaluating terrain/risk cost.
        self.feasibility = FeasibilityEngine()
        # Cache for the most recent prediction frame (intent, confidence, heatmap).
        self.last_prediction = None

    def get_smart_action(self, unit_id, env):
        """
        Calculates the optimal tactical action for a blue unit given current battlefield intel.
        
        Args:
            unit_id: Unique identifier for the blue actor.
            env: Reference to the Gym environment for state access.

        Returns:
            int: The index of the winning action in the action space [0-5].
        """
        # ── PHASE 1: Unit Identification ──
        # Locate the specific actor within the environment's internal manifest.
        blue_unit = next((u for u in env.blue_units if u['id'] == unit_id), None)
        if not blue_unit or not blue_unit['alive']:
            return 0 # Default: Stationary stance if the unit is incapacitated or missing.

        # ── PHASE 2: Intelligence Synchronization (Perception) ──
        # Tactical Constraint: We update the Oracle's LSTM state ONCE per global turn.
        # This provides the 'Cognitive Foundation' for the decision-making pass.
        red_unit = next((u for u in env.red_units if u['alive']), None)
        h_map = np.zeros((10, 10)) # Baseline zero-risk map.
        
        if red_unit:
            # Package temporal features for the LSTM input window.
            state_dict = {
                # Adversary State
                "red_x": red_unit['pos'][1], "red_y": red_unit['pos'][0],
                # Friendly State
                "blue_x": blue_unit['pos'][1], "blue_y": blue_unit['pos'][0],
                # Logistical Status
                "red_hp": red_unit['hp'], "red_ammo": red_unit['ammo'], "red_fuel": red_unit['fuel'],
                # Psychological and Physical signals
                "red_morale": red_unit['morale'],
                "red_elevation": env.elevation[red_unit['pos'][0], red_unit['pos'][1]],
                # Spatial Metrics
                "distance": abs(red_unit['pos'][0] - blue_unit['pos'][0]) + abs(red_unit['pos'][1] - blue_unit['pos'][1]),
                "fog_visible": 1 if env.fog_map[red_unit['pos'][0], red_unit['pos'][1]] else 0,
                # Action History
                "red_prev_action": red_unit.get('prev_action', 0)
            }
            # Execute the Oracle pass: Updates the LSTM's hidden state and generates the prediction dictionary.
            self.last_prediction = self.oracle.update(state_dict)
            if self.last_prediction:
                # Capture the risk heatmap for use in the navigation pass.
                h_map = self.last_prediction['heatmap']

        # ── PHASE 3: Maneuver Evaluation (Planning) ──
        # We perform a "What-If" look-ahead for every possible action in the space.
        action_scores = []
        env_state = {'terrain': env.terrain_map, 'elevation': env.elevation, 'fog': env.fog_map}
        
        for action in range(6):
            # Simulation: Project unit position for the next T+1 frame.
            tr, tc = blue_unit['pos']
            if action == 1: tr -= 1 # North
            elif action == 2: tr += 1 # South
            elif action == 3: tc -= 1 # West
            elif action == 4: tc += 1 # East
            
            # Boundary Sanitization: Heavily penalize out-of-bounds projections.
            if not (0 <= tr < 10 and 0 <= tc < 10):
                action_scores.append(-999.0)
                continue

            # Manuever Scoring (Feasibility Engine):
            # This calculates the physical cost of the move, terrain penalties, and Oracle-predicted risk.
            f_results = self.feasibility.get_maneuver_score(blue_unit, (tr, tc), env_state, h_map)
            
            # Strategic Bias Layer:
            # We apply additional weights to encourage specific mission objectives.
            strategic_bonus = 0
            
            if action == 5: # Branch: RANGED_ATTACK
                # Strategy: Engagement is prioritized if targets are in-range and resources allow.
                in_range = False
                for r_u in env.red_units:
                    if r_u['alive']:
                        dist = abs(blue_unit['pos'][0] - r_u['pos'][0]) + abs(blue_unit['pos'][1] - r_u['pos'][1])
                        if dist <= 3: # Standard engagement envelope.
                            in_range = True
                            break
                
                if in_range and blue_unit['ammo'] > 0:
                    strategic_bonus += 15.0 # Max Priority: Offensive Pressure.
                elif blue_unit['ammo'] <= 0:
                    strategic_bonus -= 20.0 # Refuse to attempt fire on empty bins.
            
            else: # Branch: MOVEMENT (0-4)
                # Strategy: Bias movement toward the suspected location of the enemy.
                if red_unit:
                    new_dist = abs(tr - red_unit['pos'][0]) + abs(tc - red_unit['pos'][1])
                    curr_dist = abs(blue_unit['pos'][0] - red_unit['pos'][0]) + abs(blue_unit['pos'][1] - red_unit['pos'][1])
                    if new_dist < curr_dist:
                        strategic_bonus += 0.5 # Aggression weight.
                
                # Favor Road movement (ID=5/4 depending on schema) for speed.
                if env.terrain_map[tr, tc] == 4: 
                    strategic_bonus += 0.2

            # Aggregate total score for this action candidate.
            action_scores.append(f_results['overall'] + strategic_bonus)

        # ── PHASE 4: Execution Decision ──
        # Selection Policy: Deterministic winner (Max Score).
        return int(np.argmax(action_scores))

    def reset(self):
        """Resets the brain's internal states for a fresh engagement initialization."""
        self.oracle.reset()
        self.last_prediction = None
