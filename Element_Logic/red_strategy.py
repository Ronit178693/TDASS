# File docstring for the Enemy AI controller.
"""
# Header for the red strategy file.
red_strategy.py — Enhanced Red Bot with Tactical Postures (Phase A4+)
# Logical separator.
=====================================================================
# Strategy definitions.
Postures:
# Passive exploration.
  SCOUT   — Patrol randomly, gathering intel.
# Focused aggression.
  ATTACK  — Aggressively move toward the nearest Blue unit.
# Defensive preservation.
  RETREAT — Move away from nearest Blue when HP is low.
# Indirect approach.
  FLANK   — Circle around to approach from the side/rear.
# Resource acquisition.
  RESUPPLY — Move toward nearest supply crate when low on resources.
# Empty line.
# Decision criteria summary.
Posture transitions consider: HP, distance, ammo, fuel, morale, elevation.
# End of documentation block.
"""
# Empty line.

# Import random for stochastic scouting behavior.
import random
# Import numpy for grid array handling.
import numpy as np
# Empty line.

# ──────────────────────────────────────────────
# Visual divider for posture state machine.
# Posture constants
# String ID for scouting.
SCOUT    = "SCOUT"
# String ID for attacking.
ATTACK   = "ATTACK"
# String ID for retreating.
RETREAT  = "RETREAT"
# String ID for flanking.
FLANK    = "FLANK"
# String ID for looting.
RESUPPLY = "RESUPPLY"
# Empty line.

# ──────────────────────────────────────────────
# Visual divider for action synchronization.
# Action constants (must match battle_env.py)
# Index for idling.
STAY  = 0
# Index for moving north.
UP    = 1
# Index for moving south.
DOWN  = 2
# Index for moving west.
LEFT  = 3
# Index for moving east.
RIGHT = 4
# Index for combat engagement.
RANGED_ATTACK = 5
# Empty line.

# List mapping terrain IDs to names for logging.
# Terrain names (kept for backward compat with simulation import)
TERRAIN_NAMES = {0: "Plains", 1: "Wall", 2: "Forest", 3: "Water", 4: "Urban", 5: "Road"}
# Empty line.

# Vector map to convert action indexes to coordinate deltas.
# Direction vectors: action -> (dr, dc)
DIRECTION = {
# North vector.
    UP:    (-1,  0),
# South vector.
    DOWN:  ( 1,  0),
# West vector.
    LEFT:  ( 0, -1),
# East vector.
    RIGHT: ( 0,  1),
# End of dictionary.
}
# Empty line.
# Empty line.


# State machine transition logic.
def determine_posture(red_unit, nearest_blue, red_all_units=None, supply_drops=None):
# Docstring.
    """
    Decide the tactical posture for a red unit based on battlefield state.

    Args:
        red_unit:      dict with keys {pos, hp, ammo, fuel, alive, morale}
        nearest_blue:  dict of nearest blue unit (or None)
        red_all_units: list of all red units (for coordination)
        supply_drops:  list of supply crate dicts (optional)

    Returns:
        str: one of SCOUT, ATTACK, RETREAT, FLANK, RESUPPLY
    """
# Failsafe: if no targets exist, just scout.
    if nearest_blue is None:
        return SCOUT
# Empty line.

# Coordinate extraction.
    rx, ry = red_unit["pos"]
    bx, by = nearest_blue["pos"]
# Calculate Manhattan distance to primary threat.
    dist = abs(rx - bx) + abs(ry - by)
# Empty line.

# Unit attribute extraction.
    hp     = red_unit["hp"]
    ammo   = red_unit["ammo"]
    fuel   = red_unit["fuel"]
    morale = red_unit.get("morale", 100)
# Empty line.

# Check for self-preservation trigger.
    # RETREAT: critically low HP, fuel, or morale
    if hp <= 25 or fuel <= 5 or morale <= 30:
# Intelligence check: Is there a nearby supply crate to go for instead?
        # Check if a supply crate is nearby — go for it instead of blind retreat
        if supply_drops and (hp <= 40 or ammo <= 5 or fuel <= 15):
# Call internal search function.
            nearest_supply = _find_nearest_supply(red_unit, supply_drops)
# Verification.
            if nearest_supply is not None:
# Distance check to supply.
                supply_dist = (abs(rx - nearest_supply["pos"][0]) +
                               abs(ry - nearest_supply["pos"][1]))
# Only divert to supply if truly nearby.
                if supply_dist <= 4:
                    return RESUPPLY
# Execute retreat if no supply available.
        return RETREAT
# Empty line.

# Check for general supply need trigger.
    # RESUPPLY: low resources and a crate is close
    if supply_drops and (hp <= 50 or ammo <= 10 or fuel <= 20):
# Search logic.
        nearest_supply = _find_nearest_supply(red_unit, supply_drops)
# Verification.
        if nearest_supply is not None:
# Map distance.
            supply_dist = (abs(rx - nearest_supply["pos"][0]) +
                           abs(ry - nearest_supply["pos"][1]))
# Diversion threshold.
            if supply_dist <= 5:
                return RESUPPLY
# Empty line.

# Check for aggression trigger.
    # ATTACK: close range, healthy, high morale
    if dist <= 3 and hp >= 50 and ammo >= 5 and morale >= 60:
        return ATTACK
# Empty line.

# Check for flanking strategy trigger.
    # FLANK: medium range, decent health
    if 3 < dist <= 6 and hp >= 40 and morale >= 50:
        return FLANK
# Empty line.

# Default behavior.
    # SCOUT: default
    return SCOUT
# Empty line.
# Empty line.


# Helper find proximity loot.
def _find_nearest_supply(unit, supply_drops):
# Docstring.
    """Find the closest supply crate to a unit, or None."""
# Empty list check.
    if not supply_drops:
        return None
# Track best found.
    best, best_d = None, float("inf")
# Scan list.
    for s in supply_drops:
# Calculation.
        d = abs(unit["pos"][0] - s["pos"][0]) + abs(unit["pos"][1] - s["pos"][1])
# Update logic.
        if d < best_d:
            best_d, best = d, s
# Return crate dict.
    return best
# Empty line.
# Empty line.


# Filter for possible movement directions.
def _get_valid_actions(pos, grid_size, terrain_map):
# Docstring.
    """
    Return list of valid movement actions (1-4) from current position.
    Excludes OOB and impassable terrain (Wall=1, Water=3).
    """
# Results list.
    valid = []
# Scan directions UDLR.
    for action, (dr, dc) in DIRECTION.items():
# Bound calculation.
        nr, nc = pos[0] + dr, pos[1] + dc
# Bound verification.
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
# Terrain collision check.
            terrain = terrain_map[nr, nc]
# Filter out walls/water.
            if terrain not in (1, 3):
                valid.append(action)
# Return list of action IDs.
    return valid
# Empty line.
# Empty line.


# Pathing heuristic to close distance.
def _action_toward(pos, target, grid_size, terrain_map):
# Docstring.
    """Pick the action that minimizes Manhattan distance to target."""
# Filter choices.
    valid = _get_valid_actions(pos, grid_size, terrain_map)
# Failsafe.
    if not valid:
        return STAY
# Initialize tracking.
    best_action = STAY
    best_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
# Evaluation loop.
    for action in valid:
        dr, dc = DIRECTION[action]
        nr, nc = pos[0] + dr, pos[1] + dc
# New distance.
        d = abs(nr - target[0]) + abs(nc - target[1])
# Better check.
        if d < best_dist:
            best_dist = d
            best_action = action
# Best found action.
    return best_action
# Empty line.
# Empty line.


# Pathing heuristic to create distance.
def _action_away(pos, threat, grid_size, terrain_map):
# Docstring.
    """Pick the action that maximizes Manhattan distance from threat."""
# Filter choices.
    valid = _get_valid_actions(pos, grid_size, terrain_map)
# Failsafe.
    if not valid:
        return STAY
# Initialize tracking.
    best_action = random.choice(valid)
    best_dist = abs(pos[0] - threat[0]) + abs(pos[1] - threat[1])
# Evaluation loop.
    for action in valid:
        dr, dc = DIRECTION[action]
        nr, nc = pos[0] + dr, pos[1] + dc
# New distance.
        d = abs(nr - threat[0]) + abs(nc - threat[1])
# Better check for larger distance.
        if d > best_dist:
            best_dist = d
            best_action = action
# Best choice.
    return best_action
# Empty line.
# Empty line.


# Strategic retreat logic that targets defensive tiles.
def _action_toward_cover(pos, grid_size, terrain_map, threat_pos):
# Docstring.
    """
    Move toward nearby forest or urban cover while retreating from threat.
    Falls back to plain retreat if no cover is accessible.
    """
# Valid filter.
    valid = _get_valid_actions(pos, grid_size, terrain_map)
# Failsafe.
    if not valid:
        return STAY
# Score tracking.
    # Prefer cells that are cover (Forest=2, Urban=4) AND move away from threat
    best_action = None
    best_score = -float("inf")
# Scan choices.
    for action in valid:
        dr, dc = DIRECTION[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        terrain = terrain_map[nr, nc]
# Binary cover bonus.
        cover_bonus = 2.0 if terrain in (2, 4) else 0.0
# Survival bonus.
        dist_from_threat = abs(nr - threat_pos[0]) + abs(nc - threat_pos[1])
# Compound score.
        score = dist_from_threat + cover_bonus
# Update check.
        if score > best_score:
            best_score = score
            best_action = action
# Final choice.
    return best_action if best_action is not None else STAY
# Empty line.
# Empty line.


# Indirect approach algorithm.
def _action_flank(pos, target, grid_size, terrain_map):
# Docstring.
    """
    Move perpendicular to the direct approach line, then close in.
    """
# Valid filter.
    valid = _get_valid_actions(pos, grid_size, terrain_map)
# Failsafe.
    if not valid:
        return STAY
# Vector calculation.
    rx, ry = pos
    tx, ty = target
    dx, dy = tx - rx, ty - ry
# Identify lateral axis.
    if abs(dx) >= abs(dy):
        lateral = [a for a in [LEFT, RIGHT] if a in valid]
    else:
        lateral = [a for a in [UP, DOWN] if a in valid]
# Re-evaluate distance.
    dist = abs(dx) + abs(dy)
# Close combat check.
    if dist <= 2:
        return _action_toward(pos, target, grid_size, terrain_map)
# Apply lateral movement if possible.
    if lateral:
        return random.choice(lateral)
# Default path.
    return _action_toward(pos, target, grid_size, terrain_map)
# Empty line.
# Empty line.


# Master integration function for the red team.
def get_red_action(red_unit, blue_units, grid_size, terrain_map,
                   red_all_units=None, supply_drops=None):
# Docstring.
    """
    Main entry point: decide the next action for a single red unit.

    Args:
        red_unit:      dict — the red unit to control
        blue_units:    list[dict] — all alive blue units
        grid_size:     int
        terrain_map:   np.ndarray (grid_size × grid_size) — terrain codes
        red_all_units: list[dict] — all red units (optional)
        supply_drops:  list[dict] — active supply crates (optional)

    Returns:
        (action: int, posture: str)
    """
# Check health.
    if not red_unit["alive"]:
        return STAY, SCOUT
# Get living targets.
    alive_blues = [u for u in blue_units if u["alive"]]
# Empty team check.
    if not alive_blues:
        return STAY, SCOUT
# Strategy target filter.
    nearest_blue = min(
        alive_blues,
        key=lambda b: abs(red_unit["pos"][0] - b["pos"][0]) +
                       abs(red_unit["pos"][1] - b["pos"][1])
    )
# Call posture decision engine.
    # Determine posture (now considers supplies)
    posture = determine_posture(red_unit, nearest_blue, red_all_units, supply_drops)
# Extract coordinates.
    pos    = red_unit["pos"]
    target = nearest_blue["pos"]
# Calculation distance.
    dist   = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
# Empty line.

# Branch by current tactical posture.
    # Execute posture
    if posture == ATTACK:
# Engagement check.
        if dist <= 3 and red_unit["ammo"] >= 5:
            return RANGED_ATTACK, posture
# Close gap logic.
        return _action_toward(pos, target, grid_size, terrain_map), posture
# Safety check.
    elif posture == RETREAT:
# Cover seeking logic.
        return _action_toward_cover(pos, grid_size, terrain_map, target), posture
# Flank check.
    elif posture == FLANK:
# Lateral movement logic.
        return _action_flank(pos, target, grid_size, terrain_map), posture
# Logistics logic.
    elif posture == RESUPPLY:
# Search for crate.
        nearest_supply = _find_nearest_supply(red_unit, supply_drops)
# Loot seeking path.
        if nearest_supply is not None:
            return _action_toward(pos, nearest_supply["pos"], grid_size, terrain_map), posture
# Default if crate lost.
        # Fallback to scout if no supply found
        valid = _get_valid_actions(pos, grid_size, terrain_map)
        return (random.choice(valid) if valid else STAY), SCOUT
# Exploration logic.
    else:  # SCOUT
        valid = _get_valid_actions(pos, grid_size, terrain_map)
# Random pathing.
        if valid:
            return random.choice(valid), posture
# Idling.
        return STAY, posture
# Empty line.
# Empty line.


# ──────────────────────────────────────────────
# Visual divider for compatibility layer.
# Backward-compatible wrapper for old simulation.py
# ──────────────────────────────────────────────
# Legacy integration for Phase A.
def get_red_action_legacy(red_pos, blue_pos, grid_size, obstacles):
# Docstring.
    """
    Legacy API: takes raw positions and returns just an action int.
    """
# Validity check.
    if red_pos is None or blue_pos is None:
        return 0
# Create mock unit objects for internal logic.
    red_unit = {"id": "R0", "team": "red", "pos": list(red_pos),
                "hp": 100, "ammo": 50, "fuel": 100, "alive": True, "morale": 100}
    blue_unit = {"id": "B0", "team": "blue", "pos": list(blue_pos),
                 "hp": 100, "ammo": 50, "fuel": 100, "alive": True, "morale": 100}
# Reconstruct terrain map from obstacle coordinates.
    terrain_map = np.zeros((grid_size, grid_size), dtype=int)
    for obs in obstacles:
        terrain_map[obs[0], obs[1]] = 1
# Execute the modern logic path.
    action, posture = get_red_action(red_unit, [blue_unit], grid_size, terrain_map)
# Handle combat conversion for legacy environments.
    if action == RANGED_ATTACK:
        action = _action_toward(list(red_pos), list(blue_pos), grid_size, terrain_map)
# Final action ID.
    return action
