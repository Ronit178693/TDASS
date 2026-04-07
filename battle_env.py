# File docstring explaining the environment setup and Phase A+ features.
"""
# Header for the battle environment file.
battle_env.py — Enhanced Battlefield Environment (Phase A+)
# Separator for visual clarity.
==========================================================
# Description of the Gymnasium-based tactical simulation core.
A Gymnasium environment for multi-unit tactical simulation with:
# Detail on movement logic and terrain diversity.
  - 6 terrain types with movement costs
# Detail on unit resource management (HP, Ammo, Fuel).
  - Multiple Blue / Red units with individual resources (HP, ammo, fuel)
# Detail on the combat systems (Melee and Ranged).
  - Combat mechanics (ranged + melee)
# Detail on visibility restrictions per team.
  - Fog of War — units have limited vision radius
# Detail on random resource spawns on the map.
  - Supply Drops — resource crates spawn on the map
# Detail on tactical high-ground advantages.
  - Elevation layer — high ground grants attack bonus
# Detail on psychological performance system.
  - Morale system — unit effectiveness scales with events
# Detail on the graphical interface components.
  - Rich Pygame renderer with terrain, fog overlay, minimap, particles
# Detail on the OODA-loop data available for external AI.
  - Full OODA-loop data exposure via the info dict
# Empty line for formatting.
# 
# Legend for terrain numerical identifiers.
Grid Terrain Codes (terrain_map layer):
# Definition of Plains type.
  0 = Plains   (cost 1.0)
# Definition of Wall type.
  1 = Wall      (impassable)
# Definition of Forest type.
  2 = Forest    (cost 1.5, provides cover)
# Definition of Water type.
  3 = Water     (impassable)
# Definition of Urban type.
  4 = Urban     (cost 2.0, provides cover)
# Definition of Road type.
  5 = Road      (cost 0.5, fast movement)
# Empty line.
#
# Legend for the unit occupancy map.
Unit Layer (unit_map):
# Open tile code.
  0 = Empty
# Friendly unit code.
  1 = Blue unit
# Hostile unit code.
  2 = Red unit
# End of documentation block.
"""
# Empty line.

# Import the base gymnasium library for environment structure.
import gymnasium as gym
# Import specific space definitions for observations and actions.
from gymnasium import spaces
# Import numpy for high-performance matrix and grid operations.
import numpy as np
# Import pygame for the 2D graphical rendering interface.
import pygame
# Import copy for deep-copying state dictionaries safely.
import copy
# Import random for stochastic supply spawns and terrain generation.
import random
# Import math for distance calculations and combat geometry.
import math
# Empty line.

# ──────────────────────────────────────────────
# Section header for configuration constants.
# Constants
# ──────────────────────────────────────────────
# Integer ID for Plains terrain.
TERRAIN_PLAINS = 0
# Integer ID for Wall terrain (Impassable).
TERRAIN_WALL   = 1
# Integer ID for Forest terrain (Cover).
TERRAIN_FOREST = 2
# Integer ID for Water terrain (Impassable).
TERRAIN_WATER  = 3
# Integer ID for Urban terrain (High Cover).
TERRAIN_URBAN = 4
# Integer ID for Road terrain (Fast Move).
TERRAIN_ROAD   = 5
# Empty line.

# Dictionary mapping terrain IDs to human-readable names.
TERRAIN_NAMES = {0: "Plains", 1: "Wall", 2: "Forest", 3: "Water", 4: "Urban", 5: "Road"}
# Empty line.

# Dictionary defining the movement speed penalty for each terrain.
# Movement cost multiplier per terrain; None = impassable
TERRAIN_COST = {
# Standard movement cost.
    TERRAIN_PLAINS: 1.0,
# Blocked movement.
    TERRAIN_WALL:   None,
# Slow movement.
    TERRAIN_FOREST: 1.5,
# Blocked movement.
    TERRAIN_WATER:  None,
# Very slow movement.
    TERRAIN_URBAN:  2.0,
# Fast movement efficiency.
    TERRAIN_ROAD:   0.5,
# Close dictionary.
}
# Empty line.

# Dictionary defining damage reduction provided by specific tiles.
# Cover bonus — reduces incoming damage by this fraction
TERRAIN_COVER = {
# No protection.
    TERRAIN_PLAINS: 0.0,
# No protection (impassable anyway).
    TERRAIN_WALL:   0.0,
# 30% damage reduction.
    TERRAIN_FOREST: 0.3,
# No protection.
    TERRAIN_WATER:  0.0,
# 40% damage reduction.
    TERRAIN_URBAN:  0.4,
# No protection.
    TERRAIN_ROAD:   0.0,
# Close dictionary.
}
# Empty line.

# Constant for fuel depletion per movement action.
# Base fuel cost per step on Plains
BASE_FUEL_COST = 2.0
# Empty line.

# Section for combat efficiency constants.
# Combat constants
# Damage value for close-quarters fighting.
MELEE_DAMAGE   = 30
# Ammo cost for melee (effectively zero).
MELEE_AMMO     = 0
# Damage value for projectile fighting.
RANGED_DAMAGE  = 20
# Ammo cost for firing projectiles.
RANGED_AMMO    = 5
# Maximum distance (Manhattan) for ranged fire.
RANGED_RANGE   = 3
# Empty line.

# Section for visibility logic constants.
# Fog of War
# Distance threshold for unit sight lines.
FOG_VISION_RADIUS = 4  # Manhattan distance each unit can see
# Empty line.

# Section for resource crate constants.
# Supply drop constants
# HP restored by crate.
SUPPLY_HP    = 25
# Ammo restored by crate.
SUPPLY_AMMO  = 15
# Fuel restored by crate.
SUPPLY_FUEL  = 30
# Chance of a crate spawning at the end of each tick.
SUPPLY_SPAWN_CHANCE = 0.08  # probability per step of a new crate
# Empty line.

# Section for spatial mechanics.
# Elevation
# Damage multiplier applied to high-ground attackers.
ELEVATION_DAMAGE_BONUS = 0.25  # +25% damage from high ground
# Empty line.

# Section for unit psychology.
# Morale
# Starting morale value for all units.
MORALE_DEFAULT   = 100
# Penalty for team losses.
MORALE_KILL_ALLY = -20   # morale lost when friendly dies
# Bonus for achieving kills.
MORALE_KILL_FOE  = 15    # morale gained when enemy killed
# Penalty for intense physical damage.
MORALE_LOW_HP    = -10   # morale penalty when HP < 30
# Lower bound for effectiveness.
MORALE_MIN       = 20
# Upper bound for effectiveness.
MORALE_MAX       = 150
# Empty line.

# List of valid action indexes for unit control.
# Actions: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right, 5=RangedAttack
NUM_ACTIONS = 6
# Empty line.

# ──────────────────────────────────────────────
# Visual divider for helper functions.
# Unit helper
# ──────────────────────────────────────────────
# Modular function to package unit state into a consistent format.
def make_unit(uid, team, row, col, hp=100, ammo=50, fuel=100):
# Docstring.
    """Create a unit state dictionary."""
# Dictionary return block.
    return {
# Unique identifier string.
        "id":      uid,
# Team affiliation (blue or red).
        "team":    team,
# Current XY coordinate pair.
        "pos":     [row, col],
# Visual health points.
        "hp":      hp,
# Current projectile count.
        "ammo":    ammo,
# Current movement energy.
        "fuel":    fuel,
# Boolean existence flag.
        "alive":   True,
# Current psychological scalar.
        "morale":  MORALE_DEFAULT,
# Tally of defeated enemies.
        "kills":   0,
# End of dictionary.
    }
# Empty line.
# Empty line.


# ──────────────────────────────────────────────
# Visual divider for map generation logic.
# Default 10×10 terrain & elevation maps
# ──────────────────────────────────────────────
# Hand-placed terrain generator.
def default_terrain():
# Docstring.
    """Hand-crafted 10×10 terrain with tactical features."""
# Initialize blank plains grid.
    t = np.zeros((10, 10), dtype=int)
# Place forest line on far left.
    t[3, 0:5] = TERRAIN_FOREST
# Extend forest line.
    t[4, 0:3] = TERRAIN_FOREST
# Create water barrier on right.
    t[2:8, 7] = TERRAIN_WATER
# Central urban zone.
    t[4:6, 4:6] = TERRAIN_URBAN
# Bottom road connection.
    t[8, :]    = TERRAIN_ROAD
# Strategic obstacles (Walls).
    t[1, 5]    = TERRAIN_WALL
# Blocking wall.
    t[2, 5]    = TERRAIN_WALL
# Blocking wall.
    t[6, 2]    = TERRAIN_WALL
# Blocking wall.
    t[6, 3]    = TERRAIN_WALL
# Return the finished map.
    return t
# Empty line.
# Empty line.


# Heightmap generator for combat bonuses.
def default_elevation(grid_size=10):
# Docstring.
    """Elevation layer: 0=low, 1=mid, 2=high.  Hills in centre & corners."""
# Initialize flat ground.
    e = np.zeros((grid_size, grid_size), dtype=int)
# Set high ground to central urban center.
    e[4:6, 4:6] = 2   # urban hilltop
# Slight incline at blue spawn.
    e[0:2, 0:2] = 1   # blue spawn ridge
# Slight incline at red spawn.
    e[8:10, 8:10] = 1  # red spawn ridge
# Sniper nest overlooking water.
    e[3, 7] = 2        # sniper bluff
# Finished heightmap.
    return e
# Empty line.
# Empty line.


# ──────────────────────────────────────────────
# Visual divider for visual effects engine.
# Particle system (for renderer)
# ──────────────────────────────────────────────
# Single object in a particle cloud.
class _Particle:
# Optimized memory allocation for many short-lived objects.
    __slots__ = ("x", "y", "vx", "vy", "life", "color", "size")
# Initialization for a new spark.
    def __init__(self, x, y, vx, vy, life, color, size=3):
# Store position and speed vectors.
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
# Duration count.
        self.life = life
# Visual attribute.
        self.color = color
# Visual scale.
        self.size = size
# Empty line.
# Empty line.


# Manager for groups of particles.
class _ParticleSystem:
# Docstring.
    """Lightweight particle emitter for combat FX."""
# Logic setup.
    def __init__(self):
# Empty list for active sparks.
        self.particles: list[_Particle] = []
# Create a burst of particles at a coordinate.
    def emit(self, x, y, color, count=8, speed=3.0, life=15):
# Loop to create each spark in the burst.
        for _ in range(count):
# Randomize direction.
            angle = random.uniform(0, 2 * math.pi)
# Randomize speed.
            spd   = random.uniform(0.5, speed)
# Append new object to tracked list.
            self.particles.append(_Particle(
                x, y,
                math.cos(angle) * spd, math.sin(angle) * spd,
                random.randint(life // 2, life),
                color, random.randint(2, 4)
            ))
# Process logical state of sparks.
    def update(self):
# List for survivors.
        alive = []
# Process each particle.
        for p in self.particles:
# Update X according to speed.
            p.x += p.vx
# Update Y according to speed.
            p.y += p.vy
# Decay duration.
            p.life -= 1
# Retention check.
            if p.life > 0:
                alive.append(p)
# Swap active list to filtered survivors.
        self.particles = alive
# Visual execution.
    def draw(self, surface):
# Loop through sparks.
        for p in self.particles:
# Calculate fade based on remaining duration.
            alpha = max(0, min(255, int(255 * p.life / 15)))
# Ensure color stays within 8-bit bounds.
            c = (min(255, p.color[0]), min(255, p.color[1]), min(255, p.color[2]))
# Execute drawing command on the game screen.
            pygame.draw.circle(surface, c, (int(p.x), int(p.y)), p.size)
# Empty line.
# Empty line.


# ──────────────────────────────────────────────
# Visual divider for primary class definition.
# BattleEnv
# ──────────────────────────────────────────────
# The main tactical engine class.
class BattleEnv(gym.Env):
# Docstring detailing data structure.
    """
    Enhanced Battlefield Gymnasium Environment.

    Observation: dict with keys
      - "terrain_map": (H, W) int   — terrain type per cell
      - "unit_map":    (H, W) int   — 0=empty, 1=blue, 2=red
      - "fog_map":     (H, W) int   — 1=visible to blue, 0=fogged
      - "elevation":   (H, W) int   — elevation level per cell

    Action: int 0-5  (applied to the *active* blue unit via step)
    """
# Supported render configurations.
    metadata = {"render_modes": ["human", "ansi"]}
# Class initialization.
    def __init__(
        self,
        render_mode="ansi",
        grid_size=10,
        terrain_map=None,
        elevation_map=None,
        num_blue=2,
        num_red=2,
        max_steps=200,
        fog_enabled=True,
    ):
# Python inheritance setup.
        super().__init__()
# Save grid dimensions.
        self.grid_size   = grid_size
# Save visual mode.
        self.render_mode = render_mode
# Set team sizes.
        self.num_blue    = num_blue
        self.num_red     = num_red
# Set loop limits.
        self.max_steps   = max_steps
# Reset ticker.
        self.current_step = 0
# Fog control.
        self.fog_enabled  = fog_enabled
# Initialize terrain (uses default if none provided).
        self.terrain_map = np.array(terrain_map, dtype=int) if terrain_map is not None else default_terrain()
# Initialize heightmap (uses default if none provided).
        self.elevation = np.array(elevation_map, dtype=int) if elevation_map is not None else default_elevation(grid_size)
# Initialize visibility mask.
        self.fog_map = np.ones((grid_size, grid_size), dtype=int)
# Initialize empty loot pool.
        self.supply_drops: list[dict] = []
# Set discrete action space.
        self.action_space = spaces.Discrete(NUM_ACTIONS)
# Set complex dictionary observation space.
        self.observation_space = spaces.Dict({
            "terrain_map": spaces.Box(0, 5, shape=(grid_size, grid_size), dtype=int),
            "unit_map":    spaces.Box(0, 2, shape=(grid_size, grid_size), dtype=int),
        })
# Pygame constant: Tile dimension in pixels.
        self.cell_size   = 64
# Pygame constant: Bottom HUD bar height.
        self.hud_height  = 140
# Pygame constant: Side minimap size.
        self.minimap_size = 120
# Calculate scaled window width.
        self.window_w    = self.grid_size * self.cell_size + self.minimap_size + 20
# Calculate scaled window height.
        self.window_h    = self.grid_size * self.cell_size + self.hud_height
# Screen surface placeholder.
        self.screen      = None
# GPU clock placeholder.
        self.clock       = None
# Text rendering placeholder.
        self.font        = None
# Explosion/Combat effect system initialization.
        self.particles   = _ParticleSystem()
# Unit storage lists.
        self.blue_units: list[dict] = []
        self.red_units:  list[dict] = []
# Tactical grid for checking unit location quick-look.
        self.unit_map    = np.zeros((grid_size, grid_size), dtype=int)
# Txt buffer for events.
        self.combat_log: list[str] = []
# End of init.
# Empty line.

# ─── helpers ───────────────────────────────
# Check if a move to this tile is valid.
    def _passable(self, row, col):
# Bound check.
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return False
# Check if terrain code exists in movement cost dictionary.
        return TERRAIN_COST[self.terrain_map[row, col]] is not None
# Efficiently check if another unit is also at this tile.
    def _occupied_by(self, row, col, exclude_uid=None):
# Loop through all units.
        for u in self.blue_units + self.red_units:
# Check HP and position.
            if u["alive"] and u["pos"] == [row, col] and u["id"] != exclude_uid:
                return u
# Return none if empty.
        return None
# Update the unit_map array from the list state.
    def _rebuild_unit_map(self):
# Wipe old grid.
        self.unit_map.fill(0)
# Map Blue to value 1.
        for u in self.blue_units:
            if u["alive"]:
                self.unit_map[u["pos"][0], u["pos"][1]] = 1
# Map Red to value 2.
        for u in self.red_units:
            if u["alive"]:
                self.unit_map[u["pos"][0], u["pos"][1]] = 2
# Smart spawn logic to find empty tiles near starting zones.
    def _find_spawn(self, preferred_positions, taken):
# Loop through the team's zone.
        for pos in preferred_positions:
            r, c = pos
# Check if available.
            if self._passable(r, c) and (r, c) not in taken:
                taken.add((r, c))
                return [r, c]
# Failsafe: brute force random search if zone is full.
        for _ in range(200):
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if self._passable(r, c) and (r, c) not in taken:
                taken.add((r, c))
                return [r, c]
# Final failsafe.
        return [0, 0]
# Convenience function for healthy blue units.
    def _alive_blue(self):
        return [u for u in self.blue_units if u["alive"]]
# Convenience function for healthy red units.
    def _alive_red(self):
        return [u for u in self.red_units if u["alive"]]
# Empty line.

# ─── Fog of War ────────────────────────────
# Recalculate which grid squares are visible.
    def _update_fog(self):
# Docstring.
        """Recompute fog map from blue unit positions."""
# Skip logic if feature is off.
        if not self.fog_enabled:
            self.fog_map.fill(1)
            return
# Start with total darkness.
        self.fog_map.fill(0)
# Perform radial scan for each blue unit.
        for u in self._alive_blue():
            r0, c0 = u["pos"]
# Scan vertical range.
            for dr in range(-FOG_VISION_RADIUS, FOG_VISION_RADIUS + 1):
# Scan horizontal range.
                for dc in range(-FOG_VISION_RADIUS, FOG_VISION_RADIUS + 1):
# Calculate Manhattan distance.
                    if abs(dr) + abs(dc) <= FOG_VISION_RADIUS:
                        nr, nc = r0 + dr, c0 + dc
# Update map if in bounds.
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            self.fog_map[nr, nc] = 1
# Empty line.

# ─── Supply Drops ─────────────────────────
# Randomly generate and place a supply crate.
    def _maybe_spawn_supply(self):
# Docstring.
        """Randomly place a supply crate on an empty passable cell."""
# Chance check.
        if random.random() < SUPPLY_SPAWN_CHANCE:
# Search for valid spot.
            for _ in range(50):
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
# Ensure floor is passable and unit-free.
                if self._passable(r, c) and self._occupied_by(r, c) is None:
# Ensure no overlapping crates.
                    if not any(s["pos"] == [r, c] for s in self.supply_drops):
# Pack crate data into list.
                        self.supply_drops.append({
                            "pos": [r, c], "hp": SUPPLY_HP,
                            "ammo": SUPPLY_AMMO, "fuel": SUPPLY_FUEL,
                        })
# Log event.
                        self.combat_log.append(f"SUPPLY: Crate dropped at ({r},{c})")
                        return
# Check if a unit just stepped on a crate.
    def _check_supply_pickup(self, unit):
# Docstring.
        """If unit is standing on a supply crate, consume it."""
# Search current drops.
        for s in self.supply_drops:
# Coordinate match logic.
            if s["pos"] == unit["pos"]:
# Apply stats.
                unit["hp"]   = min(100, unit["hp"]   + s["hp"])
                unit["ammo"] = min(50,  unit["ammo"] + s["ammo"])
                unit["fuel"] = min(100, unit["fuel"]  + s["fuel"])
# Erase crate.
                self.supply_drops.remove(s)
# Log.
                self.combat_log.append(
                    f"PICKUP: {unit['id']} collected supply (+{s['hp']}HP +{s['ammo']}A +{s['fuel']}F)"
                )
                return
# Empty line.

# ─── Morale helpers ────────────────────────
# Change numerical morale state.
    def _adjust_morale(self, unit, delta):
        unit["morale"] = max(MORALE_MIN, min(MORALE_MAX, unit["morale"] + delta))
# Calculate damage multiplier based on morale score.
    def _morale_multiplier(self, unit):
# Linear mapping for scaling.
        """Morale affects damage output: 100 = 1.0×, 50 = 0.75×, 150 = 1.25×."""
        return 0.5 + 0.5 * (unit["morale"] / MORALE_DEFAULT)
# Update everyone's psychology when a unit falls.
    def _broadcast_morale(self, dead_unit):
# Docstring.
        """Update morale for all units when someone dies."""
# Scan units.
        for u in self.blue_units + self.red_units:
            if not u["alive"]:
                continue
# Apply logic per team identity.
            if u["team"] == dead_unit["team"]:
                self._adjust_morale(u, MORALE_KILL_ALLY)
            else:
                self._adjust_morale(u, MORALE_KILL_FOE)
# Empty line.

# ─── Elevation helper ─────────────────────
# Calculate the advantage of height.
    def _elevation_bonus(self, attacker, defender):
# Docstring.
        """Return damage multiplier based on elevation difference."""
# Pull height from coord map.
        a_elev = self.elevation[attacker["pos"][0], attacker["pos"][1]]
        d_elev = self.elevation[defender["pos"][0], defender["pos"][1]]
# Grant 25% bonus for high ground.
        if a_elev > d_elev:
            return 1.0 + ELEVATION_DAMAGE_BONUS
        return 1.0
# Empty line.

# ─── gym interface ─────────────────────────
# Complete reboot of world state.
    def reset(self, seed=None, options=None):
# Python inheritance setup.
        super().reset(seed=seed)
# Reset tickers.
        self.current_step = 0
# Clear buffers.
        self.combat_log   = []
        self.supply_drops = []
# Initialize spawn tracking.
        taken = set()
# Region for blue team.
        blue_spawns = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
        self.blue_units = []
# Create units.
        for i in range(self.num_blue):
            pos = self._find_spawn(blue_spawns, taken)
            self.blue_units.append(make_unit(f"B{i}", "blue", pos[0], pos[1]))
# Region for red team.
        red_spawns = [(9, 9), (9, 8), (8, 9), (8, 8), (9, 6), (7, 9)]
        self.red_units = []
# Create units.
        for i in range(self.num_red):
            pos = self._find_spawn(red_spawns, taken)
            self.red_units.append(make_unit(f"R{i}", "red", pos[0], pos[1]))
# Refresh grids.
        self._rebuild_unit_map()
        self._update_fog()
# Package initial look.
        obs = self._get_obs()
        return obs, self._get_info()
# Form the observation dictionary.
    def _get_obs(self):
        return {
            "terrain_map": self.terrain_map.copy(),
            "unit_map":    self.unit_map.copy(),
            "fog_map":     self.fog_map.copy(),
            "elevation":   self.elevation.copy(),
        }
# Form the metadata information dictionary.
    def _get_info(self):
        return {
            "blue_units":    copy.deepcopy(self.blue_units),
            "red_units":     copy.deepcopy(self.red_units),
            "step":          self.current_step,
            "combat_log":    list(self.combat_log),
            "supply_drops":  copy.deepcopy(self.supply_drops),
        }
# Increment simulation time by one step.
    def step(self, action, unit_index=0):
# Docstring.
        """
        Execute an action for blue_units[unit_index].
        Returns:  obs, reward, terminated, truncated, info
        """
# Increment step counter.
        self.current_step += 1
# Wipe step combat log.
        self.combat_log = []
# Default reward.
        reward = 0.0
# Unit validity check.
        blue = self.blue_units[unit_index] if unit_index < len(self.blue_units) else None
        if blue is None or not blue["alive"]:
            return self._get_obs(), 0, False, False, self._get_info()
# Execute move actions.
        # Movement (0-4)
        if action <= 4:
            reward += self._do_move(blue, action)
# Execute ranged actions.
        # Ranged (5)
        elif action == 5:
            reward += self._do_ranged_attack(blue)
# Manage crate pickups and generation.
        # Supply check & spawn
        self._check_supply_pickup(blue)
        self._maybe_spawn_supply()
# Passive morale penalty for health.
        # Morale low-HP tick
        for u in self.blue_units + self.red_units:
            if u["alive"] and u["hp"] < 30:
                self._adjust_morale(u, MORALE_LOW_HP)
# Refresh grids.
        self._rebuild_unit_map()
        self._update_fog()
# Game over conditions.
        # Win / loss
        terminated = False
# Victory logic.
        if not self._alive_red():
            reward += 50
            terminated = True
            self.combat_log.append("VICTORY: All Red units eliminated!")
# Defeat logic.
        elif not self._alive_blue():
            reward -= 50
            terminated = True
            self.combat_log.append("DEFEAT: All Blue units eliminated!")
# Timeout logic.
        truncated = self.current_step >= self.max_steps
# Return result.
        return self._get_obs(), reward, terminated, truncated, self._get_info()
# Empty line.

# ─── movement logic ───────────────────────
# Execute coordinate shift for a unit.
    def _do_move(self, unit, action):
# Stay command logic.
        if action == 0:
            return 0.0
# Vector map for UDLR keys.
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
# Calculate new spot.
        nr, nc = unit["pos"][0] + dr, unit["pos"][1] + dc
# Block check.
        if not self._passable(nr, nc):
            return -0.5
# Calculate fuel requirements.
        terrain = self.terrain_map[nr, nc]
        fuel_cost = BASE_FUEL_COST * TERRAIN_COST[terrain]
# Fuel check.
        if unit["fuel"] < fuel_cost:
            self.combat_log.append(f"{unit['id']}: Out of fuel!")
            return -1.0
# Collision/Combat check.
        occupant = self._occupied_by(nr, nc, exclude_uid=unit["id"])
        if occupant is not None:
# Block friendlies.
            if occupant["team"] == unit["team"]:
                return -0.5
# Engage enemies.
            else:
                return self._do_melee(unit, occupant, nr, nc, fuel_cost)
# Movement success logic.
        unit["pos"] = [nr, nc]
        unit["fuel"] -= fuel_cost
# Logic to check for loot crates upon stepping on a tile.
        # CONSUME SUPPLY: If a unit hits a crate, reload and heal
        for i, drop in enumerate(self.supply_drops):
            if drop["pos"] == (nr, nc):
                unit["hp"] = min(100, unit["hp"] + 30)
                unit["ammo"] = min(50, unit["ammo"] + 15)
                unit["fuel"] = min(100, unit["fuel"] + 50)
                self.supply_drops.pop(i)
                self.combat_log.append(f"{unit['id']}: Collected SUPPLY 📦")
                break
        return 0.0
# Execute melee instance.
    def _do_melee(self, attacker, defender, nr, nc, fuel_cost):
# Subtract fuel.
        attacker["fuel"] -= fuel_cost
# Calculate damage modifier from tile traits.
        cover = TERRAIN_COVER[self.terrain_map[defender["pos"][0], defender["pos"][1]]]
# Advantage from height.
        elev  = self._elevation_bonus(attacker, defender)
# Modifier from psychological state.
        morale_mult = self._morale_multiplier(attacker)
# Calculated damage total.
        damage = int(MELEE_DAMAGE * (1.0 - cover) * elev * morale_mult)
# Apply damage.
        defender["hp"] -= damage
# Log.
        self.combat_log.append(
            f"MELEE: {attacker['id']}→{defender['id']} {damage}dmg "
            f"(cover {cover:.0%}, elev ×{elev:.2f}). HP:{defender['hp']}"
        )
# Trigger blood/spark effects at enemy location.
        # Particle FX
        cx = defender["pos"][1] * self.cell_size + self.cell_size // 2
        cy = defender["pos"][0] * self.cell_size + self.cell_size // 2
        self.particles.emit(cx, cy, (255, 200, 50), count=12, speed=4.0)
# Default hit reward.
        reward = 2.0
# Fatality logic.
        if defender["hp"] <= 0:
            defender["alive"] = False
# Occupy the dead enemy's tile.
            attacker["pos"] = [nr, nc]
            attacker["kills"] += 1
# Global morale boost/drop.
            self._broadcast_morale(defender)
            self.combat_log.append(f"KILL: {defender['id']} eliminated!")
            reward += 10.0
# Defensive counter logic.
        else:
            counter_cover = TERRAIN_COVER[self.terrain_map[attacker["pos"][0], attacker["pos"][1]]]
            counter_dmg = int(MELEE_DAMAGE * 0.5 * (1.0 - counter_cover))
            attacker["hp"] -= counter_dmg
            self.combat_log.append(
                f"COUNTER: {defender['id']}→{attacker['id']} {counter_dmg}dmg. HP:{attacker['hp']}"
            )
# Possible attacker fatality.
            if attacker["hp"] <= 0:
                attacker["alive"] = False
# Global morale drop.
                self._broadcast_morale(attacker)
                self.combat_log.append(f"KILL: {attacker['id']} eliminated!")
                reward -= 15.0
# Complete combat return.
        return reward
# Execute ranged fire instance.
    def _do_ranged_attack(self, unit):
# Ammo check.
        if unit["ammo"] < RANGED_AMMO:
            self.combat_log.append(f"{unit['id']}: Not enough ammo!")
            return -1.0
# Identify alive targets on the other team.
        enemies = self._alive_red() if unit["team"] == "blue" else self._alive_blue()
# Find nearest eligible target.
        best_enemy, best_dist = None, float("inf")
        for e in enemies:
            d = abs(unit["pos"][0] - e["pos"][0]) + abs(unit["pos"][1] - e["pos"][1])
# Check distance limit.
            if d <= RANGED_RANGE and d < best_dist:
                best_dist, best_enemy = d, e
# Fail logic.
        if best_enemy is None:
            self.combat_log.append(f"{unit['id']}: No targets in range ({RANGED_RANGE}).")
            return -0.5
# Consume resources.
        unit["ammo"] -= RANGED_AMMO
# Map modifiers.
        cover = TERRAIN_COVER[self.terrain_map[best_enemy["pos"][0], best_enemy["pos"][1]]]
        elev  = self._elevation_bonus(unit, best_enemy)
# Psychology modifiers.
        morale_mult = self._morale_multiplier(unit)
# Damager calculation.
        damage = int(RANGED_DAMAGE * (1.0 - cover) * elev * morale_mult)
        best_enemy["hp"] -= damage
# Log.
        self.combat_log.append(
            f"RANGED: {unit['id']}→{best_enemy['id']} {damage}dmg "
            f"(rng {best_dist}, cover {cover:.0%}, elev ×{elev:.2f}). HP:{best_enemy['hp']}"
        )
# Tracer visual effect logic.
        # Tracer particle from shooter to target
        sx = unit["pos"][1] * self.cell_size + self.cell_size // 2
        sy = unit["pos"][0] * self.cell_size + self.cell_size // 2
        tx = best_enemy["pos"][1] * self.cell_size + self.cell_size // 2
        ty = best_enemy["pos"][0] * self.cell_size + self.cell_size // 2
# Impact sparks.
        self.particles.emit(tx, ty, (255, 100, 30), count=10, speed=3.5)
# Muzzle flash sparks.
        self.particles.emit(sx, sy, (200, 200, 255), count=4, speed=1.5, life=8)
# Hit success reward.
        reward = 2.0
# Kill logic.
        if best_enemy["hp"] <= 0:
            best_enemy["alive"] = False
            unit["kills"] += 1
            self._broadcast_morale(best_enemy)
            self.combat_log.append(f"KILL: {best_enemy['id']} eliminated!")
            reward += 10.0
# Finish.
        return reward
# Empty line.

# ─── Red convenience ───────────────────────
# Manual unit override for red bots.
    def move_red(self, unit_index, action):
# Docstring.
        """Move a red unit (used by simulation / red_strategy)."""
# Bound check.
        if unit_index >= len(self.red_units):
            return 0.0
        red = self.red_units[unit_index]
# Life check.
        if not red["alive"]:
            return 0.0
# Call standard move logic.
        if action <= 4:
            self._do_move(red, action)
# Call standard combat logic.
        elif action == 5:
            self._do_ranged_attack(red)
# Refresh logic.
        self._check_supply_pickup(red)
        self._rebuild_unit_map()
        self._update_fog()
        return 0.0
# Name lookup helper.
    def get_unit_terrain(self, unit):
        return TERRAIN_NAMES.get(self.terrain_map[unit["pos"][0], unit["pos"][1]], "Unknown")
# Empty line.

# ─── Backward-compatible properties ────────
# Legacy support for single-unit index lookups.
    @property
    def blue_pos(self):
        alive = self._alive_blue()
        return alive[0]["pos"] if alive else None
# Legacy support.
    @property
    def red_pos(self):
        alive = self._alive_red()
        return alive[0]["pos"] if alive else None
# Legacy support to find impassable coordinates.
    @property
    def obstacles(self):
        result = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if TERRAIN_COST[self.terrain_map[r, c]] is None:
                    result.append([r, c])
        return result
# Empty line.

# ─── Rendering ─────────────────────────────
# Master visual call.
    def render(self):
        if self.render_mode == "ansi":
            self._render_ansi()
        elif self.render_mode == "human":
            self._render_pygame()
# Execute text-based terminal visualization.
    def _render_ansi(self):
# Map IDs to characters.
        symbols = {
            TERRAIN_PLAINS: ".", TERRAIN_WALL: "#", TERRAIN_FOREST: "T",
            TERRAIN_WATER:  "~", TERRAIN_URBAN: "U", TERRAIN_ROAD:   "=",
        }
# Print header.
        print(f"\n{'─'*30} Step {self.current_step} {'─'*30}")
        header = "    " + " ".join(f"{c:>2}" for c in range(self.grid_size))
        print(header)
# Scan rows.
        for r in range(self.grid_size):
            row_str = f"{r:>2}  "
# Scan columns.
            for c in range(self.grid_size):
                cell = None
# Visibility mask logic.
                if self.fog_enabled and self.fog_map[r, c] == 0:
                    cell = " ?"
                else:
# Scan for Blue units.
                    for u in self.blue_units:
                        if u["alive"] and u["pos"] == [r, c]:
                            cell = f"\033[94mB{u['id'][-1]}\033[0m"
                            break
# Scan for Red units.
                    if cell is None:
                        for u in self.red_units:
                            if u["alive"] and u["pos"] == [r, c]:
                                cell = f"\033[91mR{u['id'][-1]}\033[0m"
                                break
# Scan for crates.
                    if cell is None:
                        for s in self.supply_drops:
                            if s["pos"] == [r, c]:
                                cell = f"\033[93m+S\033[0m"
                                break
# Final fallback to terrain characters.
                    if cell is None:
                        cell = f" {symbols[self.terrain_map[r, c]]}"
# Assemble string.
                row_str += cell + " "
            print(row_str)
# Force summary.
        print("\n\033[94m── Blue Forces ──\033[0m")
        for u in self.blue_units:
            status = "ALIVE" if u["alive"] else "DEAD"
            print(f"  {u['id']}: pos={u['pos']} HP={u['hp']} A={u['ammo']} "
                  f"F={u['fuel']:.0f} M={u['morale']} K={u['kills']} [{status}]")
# Force summary.
        print("\033[91m── Red Forces ──\033[0m")
        for u in self.red_units:
            status = "ALIVE" if u["alive"] else "DEAD"
            print(f"  {u['id']}: pos={u['pos']} HP={u['hp']} A={u['ammo']} "
                  f"F={u['fuel']:.0f} M={u['morale']} K={u['kills']} [{status}]")
# Log output.
        if self.combat_log:
            print("\n\033[93m── Combat Log ──\033[0m")
            for msg in self.combat_log:
                print(f"  ⚔ {msg}")
# Execute Pygame graphical visualization.
    def _render_pygame(self):
# Surface initialization check.
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("TDSS — Enhanced Tactical Battlefield")
            self.screen = pygame.display.set_mode((self.window_w, self.window_h))
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("consolas", 12)
            self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)
# Color mappings dictionary.
        # ── Colour palette ──
        TC = {
            TERRAIN_PLAINS: (50, 60, 45),   TERRAIN_WALL: (80, 80, 80),
            TERRAIN_FOREST: (20, 80, 30),   TERRAIN_WATER: (30, 60, 120),
            TERRAIN_URBAN:  (90, 80, 70),   TERRAIN_ROAD:  (100, 95, 75),
        }
# Elevation shading tints.
        ELEV_TINT  = [(0, 0, 0), (15, 15, 10), (30, 30, 20)]
# Force colors.
        BLUE_CLR   = (50, 140, 255)
        RED_CLR    = (240, 60, 60)
        SUPPLY_CLR = (255, 220, 50)
        GRID_LINE  = (40, 40, 40)
        HUD_BG     = (18, 18, 22)
        TXT        = (200, 200, 200)
        FOG_CLR    = (10, 10, 15)
# Fill backdrop.
        self.screen.fill(HUD_BG)
# Dimensions.
        cs = self.cell_size
        grid_px = self.grid_size * cs
# Draw terrain grid.
        # ── Terrain + elevation ──
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                terrain = self.terrain_map[r, c]
                base = TC[terrain]
# Tint squares according to height.
                tint = ELEV_TINT[min(self.elevation[r, c], 2)]
                color = tuple(min(255, base[i] + tint[i]) for i in range(3))
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)
# Visual marker for hilltops.
                # Elevation indicator (small triangle for high ground)
                if self.elevation[r, c] >= 2:
                    cx, cy = rect.centerx, rect.y + 4
                    pygame.draw.polygon(self.screen, (180, 180, 120),
                        [(cx - 4, cy + 6), (cx + 4, cy + 6), (cx, cy)])
# Execute Fog of War alpha layer.
        # ── Fog overlay ──
        if self.fog_enabled:
            fog_surf = pygame.Surface((grid_px, grid_px), pygame.SRCALPHA)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.fog_map[r, c] == 0:
                        fog_rect = pygame.Rect(c * cs, r * cs, cs, cs)
# 180 alpha = semi-transparent black.
                        pygame.draw.rect(fog_surf, (0, 0, 0, 180), fog_rect)
            self.screen.blit(fog_surf, (0, 0))
# Draw loot graphics.
        # ── Supply crates ──
        for s in self.supply_drops:
            sr, sc = s["pos"]
# Visibility check.
            if not self.fog_enabled or self.fog_map[sr, sc]:
                cx = sc * cs + cs // 2
                cy = sr * cs + cs // 2
# Draw yellow square.
                pygame.draw.rect(self.screen, SUPPLY_CLR,
                    (cx - 6, cy - 6, 12, 12))
                pygame.draw.rect(self.screen, (180, 150, 20),
                    (cx - 6, cy - 6, 12, 12), 1)
# Draw plus sign.
                lbl = self.font.render("+", True, (0, 0, 0))
                self.screen.blit(lbl, (cx - 3, cy - 6))
# Draw Blue units.
        # ── Units ──
        for u in self.blue_units:
            if u["alive"]:
                self._draw_unit(u, BLUE_CLR)
# Draw Red units.
        for u in self.red_units:
            if u["alive"]:
# Visible check.
                visible = (not self.fog_enabled) or self.fog_map[u["pos"][0], u["pos"][1]]
                if visible:
                    self._draw_unit(u, RED_CLR)
# Update effect sprites.
        # ── Particles ──
        self.particles.update()
        self.particles.draw(self.screen)
# Draw sidebar information module.
        # ── Minimap (right side) ──
        mm_x = grid_px + 10
        mm_y = 10
        mm_cs = self.minimap_size // self.grid_size
# Draw minimap border.
        pygame.draw.rect(self.screen, (30, 30, 35),
            (mm_x - 2, mm_y - 2, self.minimap_size + 4, self.minimap_size + 4))
# Loop for terrain pixels.
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(mm_x + c * mm_cs, mm_y + r * mm_cs, mm_cs, mm_cs)
                terrain = self.terrain_map[r, c]
                pygame.draw.rect(self.screen, TC[terrain], rect)
# Overlay units on minimap.
        # Units on minimap
        for u in self._alive_blue():
            pygame.draw.rect(self.screen, BLUE_CLR,
                (mm_x + u["pos"][1]*mm_cs, mm_y + u["pos"][0]*mm_cs, mm_cs, mm_cs))
        for u in self._alive_red():
            pygame.draw.rect(self.screen, RED_CLR,
                (mm_x + u["pos"][1]*mm_cs, mm_y + u["pos"][0]*mm_cs, mm_cs, mm_cs))
        for s in self.supply_drops:
            pygame.draw.rect(self.screen, SUPPLY_CLR,
                (mm_x + s["pos"][1]*mm_cs, mm_y + s["pos"][0]*mm_cs, mm_cs, mm_cs))
# Minimap text label.
        lbl = self.font_lg.render("MINIMAP", True, TXT)
        self.screen.blit(lbl, (mm_x, mm_y + self.minimap_size + 4))
# Draw bottom panel.
        # ── HUD Panel ──
        hud_y = grid_px + 5
# Render step stats.
        step_text = self.font_lg.render(
            f"Step: {self.current_step}/{self.max_steps}   "
            f"Supplies: {len(self.supply_drops)}", True, TXT)
        self.screen.blit(step_text, (10, hud_y))
# Display Blue health list.
        # Blue status
        bx, by = 10, hud_y + 22
        for u in self.blue_units:
# Color dead units grey.
            color = BLUE_CLR if u["alive"] else (60, 60, 60)
            txt = (f"{u['id']} HP:{u['hp']:>3} A:{u['ammo']:>2} "
                   f"F:{u['fuel']:>3.0f} M:{u['morale']:>3} K:{u['kills']}")
            surf = self.font.render(txt, True, color)
            self.screen.blit(surf, (bx, by))
            by += 16
# Display Red health list.
        # Red status
        rx, ry = self.window_w // 2, hud_y + 22
        for u in self.red_units:
# Color dead units grey.
            color = RED_CLR if u["alive"] else (60, 60, 60)
            txt = (f"{u['id']} HP:{u['hp']:>3} A:{u['ammo']:>2} "
                   f"F:{u['fuel']:>3.0f} M:{u['morale']:>3} K:{u['kills']}")
            surf = self.font.render(txt, True, color)
            self.screen.blit(surf, (rx, ry))
            ry += 16
# Event log block.
        # Combat log
        log_y = hud_y + 90
# Display last 3 events.
        for msg in self.combat_log[-3:]:
            surf = self.font.render(f"⚔ {msg[:80]}", True, (255, 200, 80))
            self.screen.blit(surf, (10, log_y))
            log_y += 14
# Render current frame.
        pygame.display.flip()
# Set target FPS.
        self.clock.tick(15)
# Sub-routine to draw unit sprite and health bars.
    def _draw_unit(self, u, color):
# Docstring.
        """Draw a single unit circle + HP bar + morale bar + label."""
# Scale settings.
        cs = self.cell_size
        cx = u["pos"][1] * cs + cs // 2
        cy = u["pos"][0] * cs + cs // 2
        radius = cs // 3
# Elite unit glow logic.
        # Morale glow ring
        if u["morale"] >= 120:
            pygame.draw.circle(self.screen, (255, 255, 150), (cx, cy), radius + 3, 2)
# Draw main solid color unit body.
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
# HP visualiztion block.
        # HP bar
        bar_w, bar_h = cs // 2, 4
        bx = cx - bar_w // 2
        by = cy + radius + 3
# Draw dark backdrop bar.
        pygame.draw.rect(self.screen, (40, 40, 40), (bx, by, bar_w, bar_h))
# Calculate current health bar width and color.
        fill_w = max(0, int(bar_w * u["hp"] / 100))
        fill_c = (50, 200, 50) if u["hp"] > 50 else (200, 200, 0) if u["hp"] > 25 else (200, 50, 50)
# Draw the colored foreground bar.
        pygame.draw.rect(self.screen, fill_c, (bx, by, fill_w, bar_h))
# Morale visualization block.
        # Morale bar (thin, below HP)
        my = by + 5
# Backdrop bar.
        pygame.draw.rect(self.screen, (40, 40, 40), (bx, my, bar_w, 2))
# Morale width and color.
        m_fill = max(0, int(bar_w * u["morale"] / MORALE_MAX))
        m_c = (100, 150, 255) if u["morale"] >= 80 else (200, 100, 50)
# Draw.
        pygame.draw.rect(self.screen, m_c, (bx, my, m_fill, 2))
# Text label identification.
        # Label
        label = self.font.render(u["id"], True, (255, 255, 255))
        self.screen.blit(label, (cx - 8, cy - 6))
# Cleanup environment call.
    def close(self):
# Verification check.
        if self.screen is not None:
# Shutdown pygame services.
            pygame.quit()
# Nullify surface.
            self.screen = None
# Empty line.
# Empty line.


# ──────────────────────────────────────────────
# Visual divider for experimental manual control.
# Interactive Demo
# ──────────────────────────────────────────────
# Check if executing script directly.
if __name__ == "__main__":
# Initialize demo environment.
    env = BattleEnv(render_mode="human", num_blue=2, num_red=2, fog_enabled=True)
# Perform reset.
    obs, info = env.reset()
# Initial render draw.
    env.render()
# UI Instruction header.
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  TDSS Enhanced Battlefield v2 — Interactive         ║")
    print("║  Actions: 0=Stay 1=Up 2=Down 3=Left 4=Right        ║")
    print("║           5=Ranged Attack                           ║")
    print("║  Format:  <unit_index> <action>                     ║")
    print("║  Example: 0 1  (move unit B0 up)                    ║")
    print("║  NEW: Fog of War · Supply Drops · Elevation · Morale║")
    print("╚══════════════════════════════════════════════════════╝")
# Main loop control.
    running = True
# Game loop.
    while running:
# Scan for system events like closing window.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
# Handle terminal console input.
        try:
# Get typed command from user.
            raw = input("Command> ").strip()
# Blank enter check.
            if not raw:
                continue
# Split components.
            parts = raw.split()
# Single action input logic.
            if len(parts) == 1:
                uid, act = 0, int(parts[0])
# Standard index + action logic.
            else:
                uid, act = int(parts[0]), int(parts[1])
# Logic check for valid action index.
            if act not in range(NUM_ACTIONS):
                print(f"Invalid action. Use 0-{NUM_ACTIONS-1}.")
                continue
# Apply chosen action to the simulation.
            obs, reward, terminated, truncated, info = env.step(act, unit_index=uid)
# Redraw screen.
            env.render()
# Notify of score changes.
            if reward != 0:
                print(f"  Reward: {reward:+.1f}")
# Automatic loop restart logic for terminal mode.
            if terminated:
                print("Episode finished. Resetting...")
                env.reset()
                env.render()
            elif truncated:
                print("Max steps reached. Resetting...")
                env.reset()
                env.render()
# Catch typing errors.
        except ValueError:
            print("Enter: <unit_index> <action>  or just <action> for unit 0.")
# Catch break commands.
        except (KeyboardInterrupt, EOFError):
            break
# Final shutdown call.
    env.close()
