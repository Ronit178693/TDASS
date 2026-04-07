"""
tactical_brain.py — v14.0 (Stable Decision Architecture)
======================================================
Orchestrates Oracle Intelligence and Feasibility Analysis.
Ensures Oracle memory is updated exactly ONCE per turn.
"""

import numpy as np
import torch
from oracle.inference_engine import OracleInferenceEngine
from feasibility.feasibility_engine import FeasibilityEngine

class TacticalBrain:
    def __init__(self):
        # Initialize internal engines
        self.oracle = OracleInferenceEngine(device="cpu") 
        self.feasibility = FeasibilityEngine()
        self.last_prediction = None

    def get_smart_action(self, unit_id, env):
        """Calculates the best tactical action for a single Blue unit."""
        # 1. Identification
        blue_unit = next((u for u in env.blue_units if u['id'] == unit_id), None)
        if not blue_unit or not blue_unit['alive']:
            return 0 # Do nothing if dead or missing

        # 2. Stable Oracle Intelligence (Predict Once)
        # We look for the first living Red unit to track as the primary threat
        red_unit = next((u for u in env.red_units if u['alive']), None)
        h_map = np.zeros((10, 10))
        
        if red_unit:
            state_dict = {
                "red_x": red_unit['pos'][1], "red_y": red_unit['pos'][0],
                "blue_x": blue_unit['pos'][1], "blue_y": blue_unit['pos'][0],
                "red_hp": red_unit['hp'], "red_ammo": red_unit['ammo'], "red_fuel": red_unit['fuel'],
                "red_morale": red_unit['morale'],
                "red_elevation": env.elevation[red_unit['pos'][0], red_unit['pos'][1]],
                "distance": abs(red_unit['pos'][0] - blue_unit['pos'][0]) + abs(red_unit['pos'][1] - blue_unit['pos'][1]),
                "fog_visible": 1 if env.fog_map[red_unit['pos'][0], red_unit['pos'][1]] else 0,
                "red_prev_action": red_unit.get('prev_action', 0)
            }
            # The Oracle memory is updated here
            self.last_prediction = self.oracle.update(state_dict)
            if self.last_prediction:
                h_map = self.last_prediction['heatmap']

        # 3. Action Evaluation Loop
        action_scores = []
        env_state = {'terrain': env.terrain_map, 'elevation': env.elevation, 'fog': env.fog_map}
        
        for action in range(6):
            # Simulate "What-If" position
            tr, tc = blue_unit['pos']
            if action == 1: tr -= 1
            elif action == 2: tr += 1
            elif action == 3: tc -= 1
            elif action == 4: tc += 1
            
            # Boundary security
            if not (0 <= tr < 10 and 0 <= tc < 10):
                action_scores.append(-999.0)
                continue

            # FEASIBILITY ENGINE: Checks terrain costs, fuel, and Oracle risk
            f_results = self.feasibility.get_maneuver_score(blue_unit, (tr, tc), env_state, h_map)
            
            # STRATEGIC BIAS:
            strategic_bonus = 0
            if action == 5:
                # Combat range check (all Red units)
                in_range = False
                for r_u in env.red_units:
                    if r_u['alive']:
                        dist = abs(blue_unit['pos'][0] - r_u['pos'][0]) + abs(blue_unit['pos'][1] - r_u['pos'][1])
                        if dist <= 3:
                            in_range = True
                            break
                if in_range and blue_unit['ammo'] > 0:
                    strategic_bonus += 15.0 # Highest Priority: Engagement
                elif blue_unit['ammo'] <= 0:
                    strategic_bonus -= 20.0 # Refuse to fire without ammo
            else:
                # Movement bias toward the suspected enemy position
                if red_unit:
                    new_dist = abs(tr - red_unit['pos'][0]) + abs(tc - red_unit['pos'][1])
                    curr_dist = abs(blue_unit['pos'][0] - red_unit['pos'][0]) + abs(blue_unit['pos'][1] - red_unit['pos'][1])
                    if new_dist < curr_dist:
                        strategic_bonus += 0.5 # Bias toward aggression
                    if env.terrain_map[tr, tc] == 4: # Road bonus
                        strategic_bonus += 0.2

            final_score = f_results['overall'] + strategic_bonus
            action_scores.append(final_score)

        # 4. Return the Winner
        return int(np.argmax(action_scores))

    def reset(self):
        """Full reset for a new match."""
        self.oracle.reset()
        self.last_prediction = None
