# Heuristic-based tactical decision engine for the Red Adversarial team.
"""
# This module implements the 'Teacher' logic that generates strategic ground-truth labels for the Oracle.
red_strategy.py — Enhanced Red Bot with Tactical Postures (Phase A4+)
=====================================================================
# The strategy is implemented as a Deterministic Finite State Machine (FSM) triggered by survival metrics.
Autonomous Postures:
# Intel Gathering: Stochastic exploration used when targets are outside the operational bubble.
  SCOUT   — Patrol randomly, gathering situational awareness.
# Lethal Engagement: High-aggression movement and projectile fire targeting the nearest Blue unit.
  ATTACK  — Close the gap and eliminate the threat.
# Survival Pivot: Prioritizes distance and defensive cover over combat when health or morale is critical.
  RETREAT — Disengage and relocate to safety.
# Tactical Maneuver: Uses lateral movement to circumvent the target's frontal LOS/arc.
  FLANK   — Indirect approach to compromise the enemy's formation.
# Logistics Recovery: Prioritizes the acquisition of supply crates to replenish combat throughput.
  RESUPPLY — Navigate toward resource drops when reserves (Ammo/Fuel) are depleted.

# Decision inputs: Real-time analysis of HP, Ammo, Fuel, Morale, Distance, and Elevation.
"""

# Import random for sampling movement directions during low-intent (SCOUT) phases.
import random
# NumPy is used to represent the terrain grid for spatial pathing calculations.
import numpy as np

# ──────────────────────────────────────────────
# Global Strategic Constants
# ──────────────────────────────────────────────

# Posture Identifiers: These strings serve as the core 'Strategic Intents' in our synthetic dataset.
SCOUT    = "SCOUT"
ATTACK   = "ATTACK"
RETREAT  = "RETREAT"
FLANK    = "FLANK"
RESUPPLY = "RESUPPLY"

# Action Identifiers: Must remain synchronized with the Gym Environment's action space (battle_env.py).
STAY  = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4
RANGED_ATTACK = 5

# Human-readable terrain mapping for debugging and terminal rendering.
TERRAIN_NAMES = {0: "Plains", 1: "Wall", 2: "Forest", 3: "Water", 4: "Urban", 5: "Road"}

# Coordinate Deltas: Mapping discrete actions to (Row, Column) vector shifts.
DIRECTION = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
}

# ──────────────────────────────────────────────
# Decision Engine: Posture Transition Logic
# ──────────────────────────────────────────────

def determine_posture(red_unit, nearest_blue, red_all_units=None, supply_drops=None):
    """
    Evaluates the current tactical situation to select the optimal behavioral state.
    
    Args:
        red_unit: The focused actor's status dictionary.
        nearest_blue: Information on the most immediate threat.
        red_all_units: Peer status for potential coordination (Future expansion).
        supply_drops: List of accessible loot locations.

    Returns:
        The winning posture string (SCOUT, ATTACK, etc.) to be used as a label.
    """
    # Baseline: If no hostiles are detected, default to reconnaissance.
    if nearest_blue is None:
        return SCOUT

    # Input Feature Extraction.
    rx, ry = red_unit["pos"]
    bx, by = nearest_blue["pos"]
    dist = abs(rx - bx) + abs(ry - by) # Manhattan distance for grid-based pathing.

    hp     = red_unit["hp"]
    ammo   = red_unit["ammo"]
    fuel   = red_unit["fuel"]
    morale = red_unit.get("morale", 100)

    # ── BRANCH 1: Critical Survival Check (RETREAT/RESUPPLY) ──
    # If the unit is operationally compromised, it must break contact.
    if hp <= 25 or fuel <= 5 or morale <= 30:
        # Before a blind retreat, verify if a nearby supply crate can salvage the situation.
        if supply_drops and (hp <= 40 or ammo <= 5 or fuel <= 15):
            nearest_supply = _find_nearest_supply(red_unit, supply_drops)
            if nearest_supply:
                s_dist = abs(rx - nearest_supply["pos"][0]) + abs(ry - nearest_supply["pos"][1])
                # If a crate is within 4 tiles, prioritize scavenging (RESUPPLY) over fleeing.
                if s_dist <= 4:
                    return RESUPPLY
        return RETREAT

    # ── BRANCH 2: Logistics Optimization (RESUPPLY) ──
    # If resources are moderate-low and a crate is easily attainable, deviate to collect it.
    if supply_drops and (hp <= 50 or ammo <= 10 or fuel <= 20):
        nearest_supply = _find_nearest_supply(red_unit, supply_drops)
        if nearest_supply:
            s_dist = abs(rx - nearest_supply["pos"][0]) + abs(ry - nearest_supply["pos"][1])
            if s_dist <= 5:
                return RESUPPLY

    # ── BRANCH 3: Aggression Thresholds (ATTACK) ──
    # Direct assault is triggered when healthy, armed, and in proximity.
    if dist <= 3 and hp >= 50 and ammo >= 5 and morale >= 60:
        return ATTACK

    # ── BRANCH 4: Tactical Positioning (FLANK) ──
    # If at 'Mid-Range', attempt to create a lateral envelope around the target.
    if 3 < dist <= 6 and hp >= 40 and morale >= 50:
        return FLANK

    # Fallback: Maintain scouting posture if no other tactical triggers are met.
    return SCOUT

# ──────────────────────────────────────────────
# Spatial Heuristics & Navigation
# ──────────────────────────────────────────────

def _find_nearest_supply(unit, supply_drops):
    """Calculates the closest resource crate via Manhattan distance."""
    if not supply_drops: return None
    best, best_d = None, float("inf")
    for s in supply_drops:
        d = abs(unit["pos"][0] - s["pos"][0]) + abs(unit["pos"][1] - s["pos"][1])
        if d < best_d:
            best_d, best = d, s
    return best

def _get_valid_actions(pos, grid_size, terrain_map):
    """Filters movements to exclude map boundaries and impassable terrain (Walls/Water)."""
    valid = []
    for action, (dr, dc) in DIRECTION.items():
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            # Wall (1) and Water (3) are physically impassable in this simulation layer.
            if terrain_map[nr, nc] not in (1, 3):
                valid.append(action)
    return valid

def _action_toward(pos, target, grid_size, terrain_map):
    """Heuristic pathing: Selects the valid move that maximizes distance reduction to the target."""
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid: return STAY # Trapped by obstacles.
    
    best_action, best_dist = STAY, abs(pos[0] - target[0]) + abs(pos[1] - target[1])
    for action in valid:
        vec = DIRECTION[action]
        d = abs((pos[0] + vec[0]) - target[0]) + abs((pos[1] + vec[1]) - target[1])
        if d < best_dist:
            best_dist, best_action = d, action
    return best_action

def _action_toward_cover(pos, grid_size, terrain_map, threat_pos):
    """Survival pathing: Maximizes distance from threat while seeking Forest (2) or Urban (4) tiles."""
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid: return STAY
    
    best_action, best_score = None, -float("inf")
    for action in valid:
        vec = DIRECTION[action]
        nr, nc = pos[0] + vec[0], pos[1] + vec[1]
        # Scoring function: Prioritize distance-to-threat + static cover bonus.
        dist = abs(nr - threat_pos[0]) + abs(nc - threat_pos[1])
        cover_bonus = 2.0 if terrain_map[nr, nc] in (2, 4) else 0.0
        score = dist + cover_bonus
        if score > best_score:
            best_score, best_action = score, action
    return best_action if best_action is not None else STAY

def _action_flank(pos, target, grid_size, terrain_map):
    """Performs lateral movement relative to the target to avoid head-on contact."""
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid: return STAY
    
    rx, ry = pos
    tx, ty = target
    dx, dy = tx - rx, ty - ry
    
    # Identify the lateral axis (perpendicular to the direct line of approach).
    if abs(dx) >= abs(dy):
        lateral = [a for a in [LEFT, RIGHT] if a in valid]
    else:
        lateral = [a for a in [UP, DOWN] if a in valid]

    # If within melee range (dist <= 2), drop the flank and transition to a direct strike.
    if (abs(dx) + abs(dy)) <= 2:
        return _action_toward(pos, target, grid_size, terrain_map)
    
    # Prioritize lateral movement if available; otherwise, close distance normally.
    return random.choice(lateral) if lateral else _action_toward(pos, target, grid_size, terrain_map)

# ──────────────────────────────────────────────
# Master Control Function
# ──────────────────────────────────────────────

def get_red_action(red_unit, blue_units, grid_size, terrain_map,
                   red_all_units=None, supply_drops=None):
    """
    Final decision orchestration for a single Red agent.
    Combines Posture Selection (Intent) with Action Execution (Behavior).
    """
    if not red_unit["alive"]: return STAY, SCOUT

    # Operational Intelligence: Identify the most immediate threat.
    alive_blues = [u for u in blue_units if u["alive"]]
    if not alive_blues: return STAY, SCOUT
    
    # Proximity mapping.
    nearest_blue = min(alive_blues, key=lambda b: abs(red_unit["pos"][0] - b["pos"][0]) +
                                                  abs(red_unit["pos"][1] - b["pos"][1]))

    # Step 1: Strategic Intent determination (The Labeling Phase).
    posture = determine_posture(red_unit, nearest_blue, red_all_units, supply_drops)

    # Step 2: Concrete Action Selection based on the chosen Intent.
    pos, target = red_unit["pos"], nearest_blue["pos"]
    dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    if posture == ATTACK:
        # In ATTACK posture, fire projectiles if in range (3), otherwise move closer.
        if dist <= 3 and red_unit["ammo"] >= 5:
            return RANGED_ATTACK, posture
        return _action_toward(pos, target, grid_size, terrain_map), posture

    elif posture == RETREAT:
        return _action_toward_cover(pos, grid_size, terrain_map, target), posture

    elif posture == FLANK:
        return _action_flank(pos, target, grid_size, terrain_map), posture

    elif posture == RESUPPLY:
        nearest_supply = _find_nearest_supply(red_unit, supply_drops)
        if nearest_supply:
            return _action_toward(pos, nearest_supply["pos"], grid_size, terrain_map), posture
        # Failsafe: If supply point is lost or unreachable, fallback to SCOUT.
        valid = _get_valid_actions(pos, grid_size, terrain_map)
        return (random.choice(valid) if valid else STAY), SCOUT

    else: # SCOUT or default.
        valid = _get_valid_actions(pos, grid_size, terrain_map)
        return (random.choice(valid) if valid else STAY), posture
