"""
red_strategy.py — Enhanced Red Bot with Tactical Postures (Phase A4+)
=====================================================================
Postures:
  SCOUT   — Patrol randomly, gathering intel.
  ATTACK  — Aggressively move toward the nearest Blue unit.
  RETREAT — Move away from nearest Blue when HP is low.
  FLANK   — Circle around to approach from the side/rear.
  RESUPPLY — Move toward nearest supply crate when low on resources.

Posture transitions consider: HP, distance, ammo, fuel, morale, elevation.
"""

import random
import numpy as np

# Posture constants
SCOUT    = "SCOUT"
ATTACK   = "ATTACK"
RETREAT  = "RETREAT"
FLANK    = "FLANK"
RESUPPLY = "RESUPPLY"

# Action constants (must match battle_env.py)
STAY  = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4
RANGED_ATTACK = 5

# Terrain names (kept for backward compat with simulation import)
TERRAIN_NAMES = {0: "Plains", 1: "Wall", 2: "Forest", 3: "Water", 4: "Urban", 5: "Road"}

# Direction vectors: action -> (dr, dc)
DIRECTION = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
}


def determine_posture(red_unit, nearest_blue, red_all_units=None, supply_drops=None):
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
    if nearest_blue is None:
        return SCOUT

    rx, ry = red_unit["pos"]
    bx, by = nearest_blue["pos"]
    dist = abs(rx - bx) + abs(ry - by)

    hp     = red_unit["hp"]
    ammo   = red_unit["ammo"]
    fuel   = red_unit["fuel"]
    morale = red_unit.get("morale", 100)

    # RETREAT: critically low HP, fuel, or morale
    if hp <= 25 or fuel <= 5 or morale <= 30:
        # Check if a supply crate is nearby — go for it instead of blind retreat
        if supply_drops and (hp <= 40 or ammo <= 5 or fuel <= 15):
            nearest_supply = _find_nearest_supply(red_unit, supply_drops)
            if nearest_supply is not None:
                supply_dist = (abs(rx - nearest_supply["pos"][0]) +
                               abs(ry - nearest_supply["pos"][1]))
                if supply_dist <= 4:
                    return RESUPPLY
        return RETREAT

    # RESUPPLY: low resources and a crate is close
    if supply_drops and (hp <= 50 or ammo <= 10 or fuel <= 20):
        nearest_supply = _find_nearest_supply(red_unit, supply_drops)
        if nearest_supply is not None:
            supply_dist = (abs(rx - nearest_supply["pos"][0]) +
                           abs(ry - nearest_supply["pos"][1]))
            if supply_dist <= 5:
                return RESUPPLY

    # ATTACK: close range, healthy, high morale
    if dist <= 3 and hp >= 50 and ammo >= 5 and morale >= 60:
        return ATTACK

    # FLANK: medium range, decent health
    if 3 < dist <= 6 and hp >= 40 and morale >= 50:
        return FLANK

    # SCOUT: default
    return SCOUT


def _find_nearest_supply(unit, supply_drops):
    """Find the closest supply crate to a unit, or None."""
    if not supply_drops:
        return None
    best, best_d = None, float("inf")
    for s in supply_drops:
        d = abs(unit["pos"][0] - s["pos"][0]) + abs(unit["pos"][1] - s["pos"][1])
        if d < best_d:
            best_d, best = d, s
    return best


def _get_valid_actions(pos, grid_size, terrain_map):
    """
    Return list of valid movement actions (1-4) from current position.
    Excludes OOB and impassable terrain (Wall=1, Water=3).
    """
    valid = []
    for action, (dr, dc) in DIRECTION.items():
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            terrain = terrain_map[nr, nc]
            if terrain not in (1, 3):
                valid.append(action)
    return valid


def _action_toward(pos, target, grid_size, terrain_map):
    """Pick the action that minimizes Manhattan distance to target."""
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid:
        return STAY

    best_action = STAY
    best_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    for action in valid:
        dr, dc = DIRECTION[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        d = abs(nr - target[0]) + abs(nc - target[1])
        if d < best_dist:
            best_dist = d
            best_action = action

    return best_action


def _action_away(pos, threat, grid_size, terrain_map):
    """Pick the action that maximizes Manhattan distance from threat."""
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid:
        return STAY

    best_action = random.choice(valid)
    best_dist = abs(pos[0] - threat[0]) + abs(pos[1] - threat[1])

    for action in valid:
        dr, dc = DIRECTION[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        d = abs(nr - threat[0]) + abs(nc - threat[1])
        if d > best_dist:
            best_dist = d
            best_action = action

    return best_action


def _action_toward_cover(pos, grid_size, terrain_map, threat_pos):
    """
    Move toward nearby forest or urban cover while retreating from threat.
    Falls back to plain retreat if no cover is accessible.
    """
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid:
        return STAY

    # Prefer cells that are cover (Forest=2, Urban=4) AND move away from threat
    best_action = None
    best_score = -float("inf")

    for action in valid:
        dr, dc = DIRECTION[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        terrain = terrain_map[nr, nc]

        cover_bonus = 2.0 if terrain in (2, 4) else 0.0
        dist_from_threat = abs(nr - threat_pos[0]) + abs(nc - threat_pos[1])
        score = dist_from_threat + cover_bonus

        if score > best_score:
            best_score = score
            best_action = action

    return best_action if best_action is not None else STAY


def _action_flank(pos, target, grid_size, terrain_map):
    """
    Move perpendicular to the direct approach line, then close in.
    """
    valid = _get_valid_actions(pos, grid_size, terrain_map)
    if not valid:
        return STAY

    rx, ry = pos
    tx, ty = target
    dx, dy = tx - rx, ty - ry

    if abs(dx) >= abs(dy):
        lateral = [a for a in [LEFT, RIGHT] if a in valid]
    else:
        lateral = [a for a in [UP, DOWN] if a in valid]

    dist = abs(dx) + abs(dy)
    if dist <= 2:
        return _action_toward(pos, target, grid_size, terrain_map)

    if lateral:
        return random.choice(lateral)

    return _action_toward(pos, target, grid_size, terrain_map)


def get_red_action(red_unit, blue_units, grid_size, terrain_map,
                   red_all_units=None, supply_drops=None):
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
    if not red_unit["alive"]:
        return STAY, SCOUT

    alive_blues = [u for u in blue_units if u["alive"]]
    if not alive_blues:
        return STAY, SCOUT

    nearest_blue = min(
        alive_blues,
        key=lambda b: abs(red_unit["pos"][0] - b["pos"][0]) +
                       abs(red_unit["pos"][1] - b["pos"][1])
    )

    # Determine posture (now considers supplies)
    posture = determine_posture(red_unit, nearest_blue, red_all_units, supply_drops)

    pos    = red_unit["pos"]
    target = nearest_blue["pos"]
    dist   = abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    # Execute posture
    if posture == ATTACK:
        if dist <= 3 and red_unit["ammo"] >= 5:
            return RANGED_ATTACK, posture
        return _action_toward(pos, target, grid_size, terrain_map), posture

    elif posture == RETREAT:
        return _action_toward_cover(pos, grid_size, terrain_map, target), posture

    elif posture == FLANK:
        return _action_flank(pos, target, grid_size, terrain_map), posture

    elif posture == RESUPPLY:
        nearest_supply = _find_nearest_supply(red_unit, supply_drops)
        if nearest_supply is not None:
            return _action_toward(pos, nearest_supply["pos"], grid_size, terrain_map), posture
        # Fallback to scout if no supply found
        valid = _get_valid_actions(pos, grid_size, terrain_map)
        return (random.choice(valid) if valid else STAY), SCOUT

    else:  # SCOUT
        valid = _get_valid_actions(pos, grid_size, terrain_map)
        if valid:
            return random.choice(valid), posture
        return STAY, posture


# ──────────────────────────────────────────────
# Backward-compatible wrapper for old simulation.py
# ──────────────────────────────────────────────
def get_red_action_legacy(red_pos, blue_pos, grid_size, obstacles):
    """
    Legacy API: takes raw positions and returns just an action int.
    """
    if red_pos is None or blue_pos is None:
        return 0

    red_unit = {"id": "R0", "team": "red", "pos": list(red_pos),
                "hp": 100, "ammo": 50, "fuel": 100, "alive": True, "morale": 100}
    blue_unit = {"id": "B0", "team": "blue", "pos": list(blue_pos),
                 "hp": 100, "ammo": 50, "fuel": 100, "alive": True, "morale": 100}

    terrain_map = np.zeros((grid_size, grid_size), dtype=int)
    for obs in obstacles:
        terrain_map[obs[0], obs[1]] = 1

    action, posture = get_red_action(red_unit, [blue_unit], grid_size, terrain_map)
    if action == RANGED_ATTACK:
        action = _action_toward(list(red_pos), list(blue_pos), grid_size, terrain_map)
    return action
