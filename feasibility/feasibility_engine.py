"""
feasibility_engine.py — Phase E: Tactical Move Validation
=========================================================
Calculates the viability of proposed tactical actions based on:
  1. Logistics (Fuel/Ammo)
  2. Physics (Terrain/Elevation/Passability)
  3. Risk (Predicted Enemy Threat from Oracle)
"""

import numpy as np

class FeasibilityEngine:
    """
    Evaluates the 'Calculated Risk' of proposed tactical maneuvers.
    """
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        # Energy cost per terrain type
        self.terrain_costs = {
            0: 1.0,  # Plains
            1: 99.0, # Wall (Impassable)
            2: 2.0,  # Forest
            3: 5.0,  # Water (Difficult)
            4: 0.5,  # Road (Efficient)
            5: 1.5   # Urban
        }
        
    def calculate_path_cost(self, start_pos, end_pos, terrain_map, fuel_current):
        """
        Check if a move is logistically possible.
        Uses Manhattan distance as a base, adjusted by terrain.
        """
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        # Distance
        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist == 0: return 1.0, 0.0 # Feasible
        
        # Estimate cost (Simple linear path estimate)
        avg_terrain_type = terrain_map[int((r1+r2)/2), int((c1+c2)/2)]
        cost_per_tile = self.terrain_costs.get(int(avg_terrain_type), 1.0)
        total_cost = dist * cost_per_tile
        
        feasibility = max(0, 1.0 - (total_cost / (fuel_current + 1e-6)))
        return feasibility, total_cost

    def evaluate_combat_risk(self, unit_pos, target_pos, red_heatmap, unit_hp):
        """
        Check if moving to target_pos is a 'Death Trap'.
        Uses the Oracle's Heatmap (probability of enemies being there).
        """
        r, c = target_pos
        # Enemy threat is the sum of probabilities in a 3x3 around the target
        threat_level = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    threat_level += red_heatmap[nr, nc]
        
        # Normalize (threat_level of 1.0 means an enemy is definitely there)
        risk = min(1.0, threat_level)
        
        # Survival is high if HP is high and risk is low
        survival_score = (unit_hp / 100.0) * (1.0 - risk)
        return survival_score, risk

    def get_maneuver_score(self, unit, target_pos, env_state, red_heatmap):
        """
        Aggregate Feasibility Score (0% to 100%).
        """
        # 1. Logistics check
        f_score, _ = self.calculate_path_cost(unit['pos'], target_pos, env_state['terrain'], unit['fuel'])
        
        # 2. Safety check
        s_score, risk = self.evaluate_combat_risk(unit['pos'], target_pos, red_heatmap, unit['hp'])
        
        # 3. Elevation check (moving Upwards is harder)
        curr_elev = env_state['elevation'][unit['pos'][0], unit['pos'][1]]
        target_elev = env_state['elevation'][target_pos[0], target_pos[1]]
        elev_penalty = 1.0 if target_elev <= curr_elev else 0.8 # 20% harder to move up
        
        # Final Weighted Score
        total_score = (f_score * 0.4) + (s_score * 0.5) + (elev_penalty * 0.1)
        
        return {
            "overall": total_score,
            "logistics": f_score,
            "survival": s_score,
            "risk": risk
        }
