# Spatial validation and tactical scoring engine for maneuver feasibility.
"""
# This module implements the 'Safety Filter' that prevents the AI from making logistically or tactically unsound moves.
feasibility_engine.py — Phase E: Tactical Move Validation
=========================================================
# It calculates a composite 'Viability Score' (0.0 to 1.0) for any proposed action.
The scoring pipeline integrates three distinct vectors of analysis:
# 1. Logistics: Verifies fuel reserves against the energetic cost of the terrain (Roads vs. Swamps).
# 2. Physics: Evaluates verticality (Elevation) and physical impassability (Walls/Water).
# 3. Intelligence: Integrates the Oracle's predicted threat heatmaps to avoid high-risk zones ('Death Traps').
"""

# NumPy for efficient spatial kernel operations and array handling.
import numpy as np

class FeasibilityEngine:
    """
    Analyzes the physics and tactical risk of the battlefield grid.
    Translates raw environmental data and AI heatmaps into a single actionable score.
    """
    
    def __init__(self, grid_size=10):
        """Initializes the constraint parameters and movement cost tables."""
        self.grid_size = grid_size
        # Energy/Fuel coefficient per Terrain ID.
        # This table defines the engine's 'A-Star-like' cost heuristics.
        self.terrain_costs = {
            0: 1.0,  # Plains (Baseline movement)
            1: 99.0, # Wall (Physical barrier - functional infinity)
            2: 2.0,  # Forest (High friction/Cover)
            3: 5.0,  # Water (Extreme energetic cost - slow fording)
            4: 0.5,  # Road (Tactical advantage: High efficiency)
            5: 1.5   # Urban (Complex navigation)
        }

    def calculate_path_cost(self, start_pos, end_pos, terrain_map, fuel_current):
        """
        Evaluates the logistical viability of a maneuver.
        
        Args:
            start_pos: Initial (r, c).
            end_pos: Target (r, c).
            terrain_map: Grid of terrain IDs.
            fuel_current: Remaning fuel resources in the unit's tank.

        Returns:
            (feasibility: float, total_cost: float)
        """
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        # Calculate Manhattan distance (Orthogonal grid steps).
        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist == 0: return 1.0, 0.0 # Stationary stance has zero cost.
        
        # Heuristic: Sample the mid-point terrain to estimate the energy cost of the path.
        avg_terrain_type = terrain_map[int((r1+r2)/2), int((c1+c2)/2)]
        cost_per_tile = self.terrain_costs.get(int(avg_terrain_type), 1.0)
        
        # Final energetic requirement.
        total_cost = dist * cost_per_tile
        
        # Feasibility is the fraction of fuel remaining after the move.
        # Values < 0 indicate the move is logistically impossible.
        feasibility = max(0, 1.0 - (total_cost / (fuel_current + 1e-6)))
        return feasibility, total_cost

    def evaluate_combat_risk(self, unit_pos, target_pos, red_heatmap, unit_hp):
        """
        Interprets Oracle heatmaps to identify potential 'Killing Zones'.
        
        Args:
            target_pos: The coordinate we are considering moving into.
            red_heatmap: The Oracle prediction grid (prob. distribution of enemy presence).
            unit_hp: Current health of the unit (resilience to risk).
        """
        r, c = target_pos
        # Convolutional Risk Scan: Sum the probabilities of threats in the 3x3 immediate vicinity.
        # This captures both direct and adjacent threat vectors.
        threat_level = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    # Accumulate spatial probability from the Oracle output.
                    threat_level += red_heatmap[nr, nc]
        
        # Normalize: A threat level index (0.0 to 1.0).
        risk = min(1.0, threat_level)
        
        # Survival Score: A function of physical HP and lack of environmental risk.
        # Low HP units are more sensitive to high-risk zones.
        survival_score = (unit_hp / 100.0) * (1.0 - risk)
        return survival_score, risk

    def get_maneuver_score(self, unit, target_pos, env_state, red_heatmap):
        """
        Aggregates all feasibility signals into a single unified maneuver rank.
        
        Returns a dictionary containing the primary 'overall' rank and sub-metrics.
        """
        # ── Vector 1: Logistics (Can we afford the move?) ──
        f_score, _ = self.calculate_path_cost(unit['pos'], target_pos, env_state['terrain'], unit['fuel'])
        
        # ── Vector 2: Survival (Will we survive the move?) ──
        s_score, risk = self.evaluate_combat_risk(unit['pos'], target_pos, red_heatmap, unit['hp'])
        
        # ── Vector 3: Vertical Advantage (Is the terrain favorable?) ──
        curr_elev = env_state['elevation'][unit['pos'][0], unit['pos'][1]]
        target_elev = env_state['elevation'][target_pos[0], target_pos[1]]
        # Physic simulation: Moving Upwards incurs a 20% penalty to maneuverability.
        elev_penalty = 1.0 if target_elev <= curr_elev else 0.8
        
        # ── COMPOSITE SCORING (The Decision Matrix) ──
        # Final weights: 
        # - Survival (50%): The prime directive.
        # - Logistics (40%): Operational endurance.
        # - Height (10%): Tactical minor advantage.
        total_score = (f_score * 0.4) + (s_score * 0.5) + (elev_penalty * 0.1)
        
        return {
            "overall": total_score,
            "logistics": f_score,
            "survival": s_score,
            "risk": risk
        }
