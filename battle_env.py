# Core simulation engine for the TDASS project, implementing high-fidelity tactical logic using Gymnasium.
"""
# This module leverages the Gymnasium API to create a standardized interface for reinforcement learning agents.
battle_env.py — Enhanced Battlefield Environment (Phase A+)
==========================================================
# The simulation is designed to provide a multi-modal observation space (spatial, resource, psychological).
A Gymnasium environment for multi-unit tactical simulation with:
# Movement is non-uniform; different terrains (Forest, Urban, etc.) have varying deployment costs.
  - 6 terrain types with movement costs
# Logistics management: Units must balance tactical aggression against their HP, Ammo, and Fuel reserves.
  - Multiple Blue / Red units with individual resources (HP, ammo, fuel)
# Deterministic combat resolution: Ranged fire is safer but high-cost; Melee is high-impact but carries health risks.
  - Combat mechanics (ranged + melee)
# Information Asymmetry: Implementation of a visibility mask (Fog of War) to simulate realistic tactical uncertainty.
  - Fog of War — units have limited vision radius
# Dynamic map events: Supply crates introduce stochastic resource replenishment, forcing units to pivot goals.
  - Supply Drops — resource crates spawn on the map
# Tactical Geometry: High-ground bonuses (Elevation) encourage AI agents to secure strategic terrain.
  - Elevation layer — high ground grants attack bonus
# Psychological Modeling: Success or failure affects a unit's 'Morale', modifying its accuracy and damage.
  - Morale system — unit effectiveness scales with events
# High-performance visualization: Integrated Pygame engine for monitoring simulation health and unit trajectories.
  - Rich Pygame renderer with terrain, fog overlay, minimap, particles
# OODA Loop support: Full state exposure via the 'info' dictionary to facilitate external predictive analysis.
  - Full OODA-loop data exposure via the info dict

# Standardized terrain IDs used for lookup in cost and cover dictionaries.
Grid Terrain Codes (terrain_map layer):
# Neutral, low-cost movement area.
  0 = Plains   (cost 1.0)
# Structural obstacles that block all movement and line-of-sight.
  1 = Wall      (impassable)
# Strategic cover: Slows movement but drastically reduces incoming small-arms damage.
  2 = Forest    (cost 1.5, provides cover)
# Natural barrier: Blocks movement but allows for line-of-sight analysis.
  3 = Water     (impassable)
# High-density cover: The most difficult terrain to traverse, offering maximum defensive bonuses.
  4 = Urban     (cost 2.0, provides cover)
# Optimized logistics path: Minimal movement cost for fast unit redeployment.
  5 = Road      (cost 0.5, fast movement)

# Identity mapping for the occupancy grid layer.
Unit Layer (unit_map):
# Ready for movement.
  0 = Empty
# Primary agent team (Friendly).
  1 = Blue unit
# Target agent team (Adversary).
  2 = Red unit
"""

# The industry-standard library for developing and comparing reinforcement learning algorithms.
import gymnasium as gym
# Action and Observation space definitions required by the Gymnasium interface.
from gymnasium import spaces
# NumPy is utilized for efficient grid-based matrix operations and spatial calculations.
import numpy as np
# Pygame provides the real-time graphical rendering context for the 2D battlefield.
import pygame
# Deep-copying unit states ensures that the 'info' metadata doesn't mutate across simulation steps.
import copy
# Psuedo-random number generation for supply spawn probabilities and terrain variation.
import random
# Standard math library for calculating Euclidean/Manhattan distances and trigonometry.
import math

# ──────────────────────────────────────────────
# Global Environment Constants & Tactical Parameters
# ──────────────────────────────────────────────

# Semantic aliases for terrain IDs, improving code readability and maintainability.
TERRAIN_PLAINS = 0
TERRAIN_WALL   = 1
TERRAIN_FOREST = 2
TERRAIN_WATER  = 3
TERRAIN_URBAN = 4
TERRAIN_ROAD   = 5

# Human-readable mapping for UI display and console logging.
TERRAIN_NAMES = {0: "Plains", 1: "Wall", 2: "Forest", 3: "Water", 4: "Urban", 5: "Road"}

# Economic cost of movement: Defines the fuel depletion for entering a specific tile type.
TERRAIN_COST = {
    TERRAIN_PLAINS: 1.0,
    TERRAIN_WALL:   None, # Impassable
    TERRAIN_FOREST: 1.5,
    TERRAIN_WATER:  None, # Impassable
    TERRAIN_URBAN:  2.0,
    TERRAIN_ROAD:   0.5,
}

# Defensive modifiers: The percentage of damage absorbed by the environment itself.
TERRAIN_COVER = {
    TERRAIN_PLAINS: 0.0,
    TERRAIN_WALL:   0.0,
    TERRAIN_FOREST: 0.3, # 30% Damage Reduction
    TERRAIN_WATER:  0.0,
    TERRAIN_URBAN:  0.4, # 40% Damage Reduction
    TERRAIN_ROAD:   0.0,
}

# Longitudinal logistics: Base fuel consumption for standard movement on Plains.
BASE_FUEL_COST = 2.0

# Combat Calibration: These values define the lethal efficiency of the units.
# Melee is high risk/reward, whereas Ranged allows for stand-off engagement.
MELEE_DAMAGE   = 30
MELEE_AMMO     = 0 # Melee combat does not consume ammunition.
RANGED_DAMAGE  = 20
RANGED_AMMO    = 5 # High ammo cost to penalize spamming fire from distance.
RANGED_RANGE   = 3 # Effective reach (Manhattan distance) for projectile attacks.

# Situational awareness logic: Defines the visibility radius for each unit.
FOG_VISION_RADIUS = 4 

# Logistics replenishment constants for supply crates.
SUPPLY_HP    = 25
SUPPLY_AMMO  = 15
SUPPLY_FUEL  = 30
# Probability of a resource drop spawning in a random valid tile at the end of a step.
SUPPLY_SPAWN_CHANCE = 0.08 

# Geometric combat advantage: Scalar bonus applied when attacking from a higher altitude.
ELEVATION_DAMAGE_BONUS = 0.25 

# Psychological throughput parameters: Determinates unit effectiveness trends.
MORALE_DEFAULT   = 100
# Penalty for team losses.
MORALE_KILL_ALLY = -20   
# Bonus for successful adversary elimination.
MORALE_KILL_FOE  = 15    
# Passive drain when critical damage is sustained.
MORALE_LOW_HP    = -10   
# Operational limits for the morale scalar.
MORALE_MIN       = 20
MORALE_MAX       = 150

# Action Space: Stay (0), Up (1), Down (2), Left (3), Right (4), RangedAttack (5).
NUM_ACTIONS = 6

# ──────────────────────────────────────────────
# Engineering Components: Data Structures & Helpers
# ──────────────────────────────────────────────

# Object factory for unit entities, ensuring a consistent schema for all actors in the world.
def make_unit(uid, team, row, col, hp=100, ammo=50, fuel=100):
    """Initializes the tactical state dictionary for a single unit entity."""
    return {
        # Unique identifier used for event logging and historical tracking.
        "id":      uid,
        # Team affiliation determines friend/foe identification in the unit_map.
        "team":    team,
        # Absolute grid coordinates [Row, Column].
        "pos":     [row, col],
        # Structural health points (0-100).
        "hp":      hp,
        # Combat resource used for projectile-based engagements.
        "ammo":    ammo,
        # Movement energy required for traversal (reduced by terrain cost).
        "fuel":    fuel,
        # Boolean flag indicating if the unit is still an active participant in the match.
        "alive":   True,
        # Performance modifier dynamic scalar.
        "morale":  MORALE_DEFAULT,
        # Count of adversaries successfully eliminated by this unit.
        "kills":   0,
    }

# Map generator utility for creating a tactical multi-terrain grid.
def default_terrain():
    """Generates a handcrafted 10x10 map with complex tactical features."""
    # Start with a baseline of open Plains (ID: 0).
    t = np.zeros((10, 10), dtype=int)
    # Define a forested flank on the western side of the map.
    t[3, 0:5] = TERRAIN_FOREST
    t[4, 0:3] = TERRAIN_FOREST
    # Establish a water barrier create a central choke point.
    t[2:8, 7] = TERRAIN_WATER
    # Central Urban hub (High cover, high traversal cost).
    t[4:6, 4:6] = TERRAIN_URBAN
    # High-speed transport corridor along the southern perimeter.
    t[8, :]    = TERRAIN_ROAD
    # Structural Walls positioned to block direct sightlines and movement.
    t[1, 5]    = TERRAIN_WALL
    t[2, 5]    = TERRAIN_WALL
    t[6, 2]    = TERRAIN_WALL
    t[6, 3]    = TERRAIN_WALL
    return t

# Strategic verticality generator.
def default_elevation(grid_size=10):
    """Assigns height levels to the grid: 0 (Low), 1 (Mid), 2 (High)."""
    e = np.zeros((grid_size, grid_size), dtype=int)
    # The Urban center is located on a defensible high-ground plateau.
    e[4:6, 4:6] = 2   
    # Natural inclines surrounding the primary deployment zones.
    e[0:2, 0:2] = 1   
    e[8:10, 8:10] = 1  
    # Strategic sniper nest overlooking the central choke point.
    e[3, 7] = 2        
    return e

# ──────────────────────────────────────────────
# Visual FX Engine: Particle Physics system
# ──────────────────────────────────────────────

# Individual component of a visual explosion or impact effect.
class _Particle:
    """Represents a physics-based visual point with a finite lifespan."""
    # Optimization: __slots__ reduces memory overhead for high-frequency particle spawning.
    __slots__ = ("x", "y", "vx", "vy", "life", "color", "size")
    def __init__(self, x, y, vx, vy, life, color, size=3):
        # Pixel-level coordinates and velocity vectors.
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        # Survival countdown (measured in frames).
        self.life = life
        # Visual styling properties.
        self.color = color
        self.size = size

# Orchestrator for managing groups of particles during combat events.
class _ParticleSystem:
    """Emitter logic that handles the spawning and aging of combat visual effects."""
    def __init__(self):
        # Buffer of currently active particle objects.
        self.particles: list[_Particle] = []
    
    # Trigger a burst of particles from a specific epicenter.
    def emit(self, x, y, color, count=8, speed=3.0, life=15):
        """Generates a radially symmetric outward burst of sparks."""
        for _ in range(count):
            # Randomize direction and velocity to simulate organic shrapnel flow.
            angle = random.uniform(0, 2 * math.pi)
            spd   = random.uniform(0.5, speed)
            self.particles.append(_Particle(
                x, y,
                math.cos(angle) * spd, math.sin(angle) * spd,
                random.randint(life // 2, life),
                color, random.randint(2, 4)
            ))

    # Frame-by-frame physics integration.
    def update(self):
        """Moves particles and cleans up objects that have reached the end of their lifespan."""
        alive = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    # Render pass: Blits the particles to the Pygame surface.
    def draw(self, surface):
        """Draws circles to the screen with transparency based on remaining life."""
        for p in self.particles:
            alpha = max(0, min(255, int(255 * p.life / 15)))
            # Color is tinted based on alpha for a fade-out effect.
            c = (min(255, p.color[0]), min(255, p.color[1]), min(255, p.color[2]))
            pygame.draw.circle(surface, c, (int(p.x), int(p.y)), p.size)

# ──────────────────────────────────────────────
# Primary Environment Logic
# ──────────────────────────────────────────────

# The main tactical simulation class, strictly adhering to the Gymnasium Env API.
class BattleEnv(gym.Env):
    """
    Enhanced Battlefield Gymnasium Environment.
    Handles the state transitions, rewards, and multi-modal observations.
    """
    # Registered render modes for compatibility with external evaluation tools.
    metadata = {"render_modes": ["human", "ansi"]}

    # Constructor: Configures the simulation grid and initialization parameters.
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
        # Invoke base Gymnasium initialization.
        super().__init__()
        self.grid_size   = grid_size
        self.render_mode = render_mode
        self.num_blue    = num_blue
        self.num_red     = num_red
        self.max_steps   = max_steps
        self.current_step = 0
        self.fog_enabled  = fog_enabled

        # Construct the static world layers (Terrain and Elevation).
        self.terrain_map = np.array(terrain_map, dtype=int) if terrain_map is not None else default_terrain()
        self.elevation = np.array(elevation_map, dtype=int) if elevation_map is not None else default_elevation(grid_size)
        
        # Situational awareness mask and supply loot tracking.
        self.fog_map = np.ones((grid_size, grid_size), dtype=int)
        self.supply_drops: list[dict] = []

        # Define the Discrete Action Space [0-5] and the multi-layered Observation Dictionary.
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "terrain_map": spaces.Box(0, 5, shape=(grid_size, grid_size), dtype=int),
            "unit_map":    spaces.Box(0, 2, shape=(grid_size, grid_size), dtype=int),
        })

        # Graphical rendering layout constants.
        self.cell_size   = 64
        self.hud_height  = 140
        self.minimap_size = 120
        self.window_w    = self.grid_size * self.cell_size + self.minimap_size + 20
        self.window_h    = self.grid_size * self.cell_size + self.hud_height
        
        # Visual assets and performance monitoring placeholders.
        self.screen      = None
        self.clock       = None
        self.font        = None
        self.particles   = _ParticleSystem()

        # Operational state buffers.
        self.blue_units: list[dict] = []
        self.red_units:  list[dict] = []
        self.unit_map    = np.zeros((grid_size, grid_size), dtype=int)
        self.combat_log: list[str] = []

    # ─── Tactical Helpers ───────────────────────────────

    # Spatial check to determine if a specific coordinate is traversable.
    def _passable(self, row, col):
        """Validates grid boundaries and terrain-specific movement cost rules."""
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return False
        return TERRAIN_COST[self.terrain_map[row, col]] is not None

    # Occupational check to determine if another unit is present at a coordinate.
    def _occupied_by(self, row, col, exclude_uid=None):
        """Returns the unit object currently standing on the cell, if any."""
        for u in self.blue_units + self.red_units:
            if u["alive"] and u["pos"] == [row, col] and u["id"] != exclude_uid:
                return u
        return None

    # Synchonization utility that updates the numeric grid layer from the unit list objects.
    def _rebuild_unit_map(self):
        """Refreshes the internal unit occupancy grid for fast spatial queries."""
        self.unit_map.fill(0)
        for u in self.blue_units:
            if u["alive"]:
                self.unit_map[u["pos"][0], u["pos"][1]] = 1
        for u in self.red_units:
            if u["alive"]:
                self.unit_map[u["pos"][0], u["pos"][1]] = 2

    # Deployment logic for placing units near their base zones at the start of a match.
    def _find_spawn(self, preferred_positions, taken):
        """Attempts to find a valid, un-occupied spawn point near the team's start area."""
        for pos in preferred_positions:
            r, c = pos
            if self._passable(r, c) and (r, c) not in taken:
                taken.add((r, c))
                return [r, c]
        # Failsafe: Brute-force random coordinate generation if preferred zones are blocked.
        for _ in range(200):
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if self._passable(r, c) and (r, c) not in taken:
                taken.add((r, c))
                return [r, c]
        return [0, 0]

    # Filtering shortcuts for active units.
    def _alive_blue(self):
        return [u for u in self.blue_units if u["alive"]]
    def _alive_red(self):
        return [u for u in self.red_units if u["alive"]]

    # ─── Fog of War Logic ────────────────────────────

    # situational awareness update triggered after every unit movement.
    def _update_fog(self):
        """Recomputes which grid squares are visible to the Blue team based on current unit positions."""
        if not self.fog_enabled:
            # If fog is disabled globally, the entire map is automatically revealed.
            self.fog_map.fill(1)
            return
        # Reset the mask to total obscurity before calculating new vision spheres.
        self.fog_map.fill(0)
        for u in self._alive_blue():
            r0, c0 = u["pos"]
            # Perform a Manhattan distance scan centered on each friendly unit.
            for dr in range(-FOG_VISION_RADIUS, FOG_VISION_RADIUS + 1):
                for dc in range(-FOG_VISION_RADIUS, FOG_VISION_RADIUS + 1):
                    if abs(dr) + abs(dc) <= FOG_VISION_RADIUS:
                        nr, nc = r0 + dr, c0 + dc
                        # Reveal the cell if it falls within the unit's sight radius.
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            self.fog_map[nr, nc] = 1

    # ─── Supply Logistics ──────────────────────────

    # Stochastic event generator for resource drops.
    def _maybe_spawn_supply(self):
        """Randomly places a supply crate on an empty traversable cell."""
        if random.random() < SUPPLY_SPAWN_CHANCE:
            # Search for a valid, unoccupied coordinate for the drop.
            for _ in range(50):
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                if self._passable(r, c) and self._occupied_by(r, c) is None:
                    # Prevent stacking multiple crates on a single tile.
                    if not any(s["pos"] == [r, c] for s in self.supply_drops):
                        self.supply_drops.append({
                            "pos": [r, c], "hp": SUPPLY_HP,
                            "ammo": SUPPLY_AMMO, "fuel": SUPPLY_FUEL,
                        })
                        self.combat_log.append(f"SUPPLY: Crate dropped at ({r},{c})")
                        return

    # Claim logic for units standing on a crate.
    def _check_supply_pickup(self, unit):
        """Consumes a supply crate if the unit coordinate matches the crate coordinate."""
        for s in self.supply_drops:
            if s["pos"] == unit["pos"]:
                # Replenish resources up to their respective hard-caps.
                unit["hp"]   = min(100, unit["hp"]   + s["hp"])
                unit["ammo"] = min(50,  unit["ammo"] + s["ammo"])
                unit["fuel"] = min(100, unit["fuel"]  + s["fuel"])
                self.supply_drops.remove(s)
                self.combat_log.append(
                    f"PICKUP: {unit['id']} collected supply (+{s['hp']}HP +{s['ammo']}A +{s['fuel']}F)"
                )
                return

    # ─── Behavioral Dynamics (Morale & Elevation) ────────────────────────

    # Updates the unit's psych scalar within established operational bounds.
    def _adjust_morale(self, unit, delta):
        unit["morale"] = max(MORALE_MIN, min(MORALE_MAX, unit["morale"] + delta))

    # Calculates the combat effectiveness modifier based on current morale.
    def _morale_multiplier(self, unit):
        """Linear scaling: 100 morale = 1.0x damage, 50 = 0.75x, 150 = 1.25x."""
        return 0.5 + 0.5 * (unit["morale"] / MORALE_DEFAULT)

    # Broadcasts psychological shifts to everyone when a fatality occurs.
    def _broadcast_morale(self, dead_unit):
        """Applies a 'loss penalty' to the same team and a 'morale boost' to the opposition."""
        for u in self.blue_units + self.red_units:
            if not u["alive"]:
                continue
            if u["team"] == dead_unit["team"]:
                self._adjust_morale(u, MORALE_KILL_ALLY)
            else:
                self._adjust_morale(u, MORALE_KILL_FOE)

    # Calculates the damage modifier based on vertical height difference between units.
    def _elevation_bonus(self, attacker, defender):
        """Grants a 25% damage bonus to attackers firing from high-ground toward low-ground."""
        a_elev = self.elevation[attacker["pos"][0], attacker["pos"][1]]
        d_elev = self.elevation[defender["pos"][0], defender["pos"][1]]
        if a_elev > d_elev:
            return 1.0 + ELEVATION_DAMAGE_BONUS
        return 1.0

    # ─── Gymnasium API Implementation ─────────────────────────

    # Standard Gym reset: Clears all world state and prepares a fresh match.
    def reset(self, seed=None, options=None):
        """Reboots the battlefield to T=0 state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.combat_log   = []
        self.supply_drops = []
        taken = set()

        # Step 1: Deploy Blue Team units in their respective spawn zones.
        blue_spawns = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
        self.blue_units = []
        for i in range(self.num_blue):
            pos = self._find_spawn(blue_spawns, taken)
            self.blue_units.append(make_unit(f"B{i}", "blue", pos[0], pos[1]))

        # Step 2: Deploy Red Team units in opposing spawn zones.
        red_spawns = [(9, 9), (9, 8), (8, 9), (8, 8), (9, 6), (7, 9)]
        self.red_units = []
        for i in range(self.num_red):
            pos = self._find_spawn(red_spawns, taken)
            self.red_units.append(make_unit(f"R{i}", "red", pos[0], pos[1]))

        # Step 3: Formalize world matrices.
        self._rebuild_unit_map()
        self._update_fog()
        return self._get_obs(), self._get_info()

    # Returns the current state representation (the input for the policy/brain).
    def _get_obs(self):
        """Constructs a copy of the grid matrices for external consumption."""
        return {
            "terrain_map": self.terrain_map.copy(),
            "unit_map":    self.unit_map.copy(),
            "fog_map":     self.fog_map.copy(),
            "elevation":   self.elevation.copy(),
        }

    # Returns additional metadata that shouldn't be parsed directly as features.
    def _get_info(self):
        """Deep-copies the list states of all units to prevent external mutation."""
        return {
            "blue_units":    copy.deepcopy(self.blue_units),
            "red_units":     copy.deepcopy(self.red_units),
            "step":          self.current_step,
            "combat_log":    list(self.combat_log),
            "supply_drops":  copy.deepcopy(self.supply_drops),
        }

    # The master transition function: Ingests an action and advances time by one tick.
    def step(self, action, unit_index=0):
        """Executes a single frame of simulation logic for the active Blue unit."""
        self.current_step += 1
        self.combat_log = []
        reward = 0.0

        # Step 1: Validate targeting - Ensure the chosen unit index exists and is alive.
        blue = self.blue_units[unit_index] if unit_index < len(self.blue_units) else None
        if blue is None or not blue["alive"]:
            # If invalid, return a null step to maintain chronological sync.
            return self._get_obs(), 0, False, False, self._get_info()

        # Step 2: Core Branching logic based on Action ID.
        # Action mappings: 0-4 handle movement (including stationary).
        if action <= 4:
            reward += self._do_move(blue, action)
        # Action index 5 triggers a ranged projectile attack.
        elif action == 5:
            reward += self._do_ranged_attack(blue)

        # Step 3: Passive world events and maintenance.
        self._check_supply_pickup(blue)
        self._maybe_spawn_supply()

        # Periodic morale attrition for units with high damage (HP < 30).
        for u in self.blue_units + self.red_units:
            if u["alive"] and u["hp"] < 30:
                self._adjust_morale(u, MORALE_LOW_HP)

        # Step 4: Refresh state matrices.
        self._rebuild_unit_map()
        self._update_fog()

        # Step 5: Termination analysis (Win/Loss/Timeout).
        terminated = False
        # Victory condition: Red annihilation.
        if not self._alive_red():
            reward += 50
            terminated = True
            self.combat_log.append("VICTORY: All Red units eliminated!")
        # Failure condition: Blue annihilation.
        elif not self._alive_blue():
            reward -= 50
            terminated = True
            self.combat_log.append("DEFEAT: All Blue units eliminated!")

        # Truncation: Environment reaching the max_steps limit (200 by default).
        truncated = self.current_step >= self.max_steps

        # Return the standard 5-tuple result for Gymnasium execution.
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ─── Interaction Execution (Movement & Combat) ───────────────────────

    # Handles unit displacement and the 'auto-melee' trigger logic.
    def _do_move(self, unit, action):
        """Moves a unit by one grid unit in UDLR directions, consuming fuel."""
        if action == 0: # 'Stay' command results in zero movement and zero cost.
            return 0.0
        # Map IDs 1-4 to specific row/column delta shifts.
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        nr, nc = unit["pos"][0] + dr, unit["pos"][1] + dc

        # Check for map boundary/terrain impassability.
        if not self._passable(nr, nc):
            return -1.0 # Significant negative reward for impacting obstacles.

        # Calculate logistics: Fuel cost scales with terrain type.
        terrain = self.terrain_map[nr, nc]
        fuel_cost = BASE_FUEL_COST * TERRAIN_COST[terrain]
        if unit["fuel"] < fuel_cost:
            self.combat_log.append(f"{unit['id']}: Out of fuel!")
            return -1.5 # Heavy penalty for attempts to move without resources.

        # Spatial collision logic: Check if another unit is present at the target tile.
        occupant = self._occupied_by(nr, nc, exclude_uid=unit["id"])
        if occupant is not None:
            if occupant["team"] == unit["team"]:
                # Friendly units act as soft-barriers (no melee against teammates).
                return -0.5
            else:
                # AUTO-MELEE: Moving into an enemy cell automatically initiates close-quarters combat.
                return self._do_melee(unit, occupant, nr, nc, fuel_cost)

        # Finalize successful movement event.
        unit["pos"] = [nr, nc]
        unit["fuel"] -= fuel_cost
        return 0.0

    # Close-quarters combat resolver.
    def _do_melee(self, attacker, defender, nr, nc, fuel_cost):
        """Resolves high-impact combat with automatic defensive counters."""
        attacker["fuel"] -= fuel_cost
        # Environment modifiers: Check if defender is in cover (e.g., Forest).
        cover = TERRAIN_COVER[self.terrain_map[defender["pos"][0], defender["pos"][1]]]
        # Vertical modifiers: Check if attacker has high-ground advantage.
        elev  = self._elevation_bonus(attacker, defender)
        # Psychological modifiers: Attacker status affects damage output.
        morale_mult = self._morale_multiplier(attacker)

        # Apply damage calculation: Base * Cover_Penalty * Elevation_Bonus * Morale.
        damage = int(MELEE_DAMAGE * (1.0 - cover) * elev * morale_mult)
        defender["hp"] -= damage

        # Emit visual feedback: Spark particle burst at impact point.
        cx = defender["pos"][1] * self.cell_size + self.cell_size // 2
        cy = defender["pos"][0] * self.cell_size + self.cell_size // 2
        self.particles.emit(cx, cy, (255, 200, 50), count=12, speed=4.0)

        reward = 2.0 # Reward for successful landing of an attack.
        if defender["hp"] <= 0:
            # FATALITY: If enemy is eliminated, attacker occupies the square.
            defender["alive"] = False
            attacker["pos"] = [nr, nc]
            attacker["kills"] += 1
            self._broadcast_morale(defender)
            self.combat_log.append(f"KILL: {defender['id']} eliminated!")
            reward += 10.0 # Major reward for tactical elimination.
        else:
            # DEFENSIVE COUNTER-STRIKE: Survivor automatically retaliates at 50% damage reduction.
            counter_cover = TERRAIN_COVER[self.terrain_map[attacker["pos"][0], attacker["pos"][1]]]
            counter_dmg = int(MELEE_DAMAGE * 0.5 * (1.0 - counter_cover))
            attacker["hp"] -= counter_dmg
            self.combat_log.append(f"COUNTER: {defender['id']}→{attacker['id']} {counter_dmg}dmg.")
            if attacker["hp"] <= 0:
                attacker["alive"] = False
                self._broadcast_morale(attacker)
                reward -= 15.0 # Significant penalty for losing a unit during a melee charge.
        return reward

    # Long-range projectile resolver.
    def _do_ranged_attack(self, unit):
        """Performs a stand-off attack consuming ammunition on the nearest target within range."""
        if unit["ammo"] < RANGED_AMMO:
            self.combat_log.append(f"{unit['id']}: Insufficient ammo!")
            return -1.0 # Penalty for inefficient resource usage.

        # Target Identification Logic.
        enemies = self._alive_red() if unit["team"] == "blue" else self._alive_blue()
        best_enemy, best_dist = None, float("inf")
        for e in enemies:
            # Calculate Manhattan distance to each potential target.
            d = abs(unit["pos"][0] - e["pos"][0]) + abs(unit["pos"][1] - e["pos"][1])
            if d <= RANGED_RANGE and d < best_dist:
                best_dist, best_enemy = d, e

        if best_enemy is None:
            self.combat_log.append(f"{unit['id']}: No targets in range {RANGED_RANGE}.")
            return -0.5 # Small penalty for firing into empty air.

        # Resource consumption.
        unit["ammo"] -= RANGED_AMMO
        # Environmental and situational modifiers.
        cover = TERRAIN_COVER[self.terrain_map[best_enemy["pos"][0], best_enemy["pos"][1]]]
        elev  = self._elevation_bonus(unit, best_enemy)
        morale_mult = self._morale_multiplier(unit)

        # Final combat calculation.
        damage = int(RANGED_DAMAGE * (1.0 - cover) * elev * morale_mult)
        best_enemy["hp"] -= damage

        # Visual FX: Impact sparks at the target point and muzzle flash at shooter.
        tx = best_enemy["pos"][1] * self.cell_size + self.cell_size // 2
        ty = best_enemy["pos"][0] * self.cell_size + self.cell_size // 2
        self.particles.emit(tx, ty, (255, 100, 30), count=10)
        
        reward = 2.0 
        if best_enemy["hp"] <= 0:
            best_enemy["alive"] = False
            unit["kills"] += 1
            self._broadcast_morale(best_enemy)
            reward += 10.0
        return reward

    # Automated override for driving the Red Team adversaries (bots).
    def move_red(self, unit_index, action):
        """Allows external scripts to commandRed units using the same world logic as Blue."""
        if unit_index >= len(self.red_units): return 0.0
        red = self.red_units[unit_index]
        if not red["alive"]: return 0.0
        # Standard movement/combat branching.
        if action <= 4: self._do_move(red, action)
        elif action == 5: self._do_ranged_attack(red)
        # World state sync.
        self._check_supply_pickup(red)
        self._rebuild_unit_map()
        self._update_fog()
        return 0.0

    # Lookup helper for identifying the ground type under a unit.
    def get_unit_terrain(self, unit):
        return TERRAIN_NAMES.get(self.terrain_map[unit["pos"][0], unit["pos"][1]], "Unknown")

    # ─── Legacy Support Properties ─────────────────────────

    @property
    def blue_pos(self):
        """Returns coordinate of lead Blue unit for basic tracking."""
        alive = self._alive_blue()
        return alive[0]["pos"] if alive else None

    @property
    def red_pos(self):
        """Returns coordinate of lead Red unit for simple distance monitoring."""
        alive = self._alive_red()
        return alive[0]["pos"] if alive else None

    @property
    def obstacles(self):
        """Returns list of all impassable coordinate pairs currently on map."""
        result = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if TERRAIN_COST[self.terrain_map[r, c]] is None:
                    result.append([r, c])
        return result

    # ─── High-Fidelity Rendering System ─────────────────────────────

    # Master render call used to push world state to the user's display.
    def render(self):
        """Selects and executes the requested visual mode (Terminal or Graphite)."""
        if self.render_mode == "ansi":
            self._render_ansi()
        elif self.render_mode == "human":
            self._render_pygame()

    # Lightweight visualizer that prints the battlefield as colored text in the console.
    def _render_ansi(self):
        """Terminal-based grid renderer using ANSI escape codes for color."""
        symbols = {
            TERRAIN_PLAINS: ".", TERRAIN_WALL: "#", TERRAIN_FOREST: "T",
            TERRAIN_WATER:  "~", TERRAIN_URBAN: "U", TERRAIN_ROAD:   "=",
        }
        print(f"\n{'─'*30} Step {self.current_step} {'─'*30}")
        header = "    " + " ".join(f"{c:>2}" for c in range(self.grid_size))
        print(header)
        for r in range(self.grid_size):
            row_str = f"{r:>2}  "
            for c in range(self.grid_size):
                cell = None
                # Strategic visibility check for the console human player.
                if self.fog_enabled and self.fog_map[r, c] == 0:
                    cell = " ?"
                else:
                    # Render friendly units (Blue).
                    for u in self.blue_units:
                        if u["alive"] and u["pos"] == [r, c]:
                            cell = f"\033[94mB{u['id'][-1]}\033[0m"
                            break
                    # Render adversarial units (Red).
                    if cell is None:
                        for u in self.red_units:
                            if u["alive"] and u["pos"] == [r, c]:
                                cell = f"\033[91mR{u['id'][-1]}\033[0m"
                                break
                    # Render loot objects.
                    if cell is None:
                        for s in self.supply_drops:
                            if s["pos"] == [r, c]:
                                cell = f"\033[93m+S\033[0m"
                                break
                    # Fallback to terrain icons.
                    if cell is None:
                        cell = f" {symbols[self.terrain_map[r, c]]}"
                row_str += cell + " "
            print(row_str)

        # Output detailed status tables for both teams.
        print("\n\033[94m── Blue Forces Status ──\033[0m")
        for u in self.blue_units:
            print(f"  {u['id']}: {u['hp']}HP, {u['ammo']}A, {u['fuel']:.1f}F, {u['morale']}M")
        print("\033[91m── Red Forces Status ──\033[0m")
        for u in self.red_units:
            print(f"  {u['id']}: {u['hp']}HP, {u['ammo']}A, {u['fuel']:.1f}F, {u['morale']}M")

    # Premium visual engine using Pygame for fluid interaction and better situational context.
    def _render_pygame(self):
        """Graphical renderer with terrain shading, particles, and interactive elements."""
        if self.screen is None:
            # First-run initialization of the Pygame window context.
            pygame.init()
            pygame.display.set_caption("TDSS — Enhanced Tactical Battlefield")
            self.screen = pygame.display.set_mode((self.window_w, self.window_h))
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("consolas", 12)
            self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)

        # Tactical Color Palette: Harmonious HSL colors for a professional simulation feel.
        TC = {
            TERRAIN_PLAINS: (50, 60, 45),   TERRAIN_WALL: (80, 80, 80),
            TERRAIN_FOREST: (20, 80, 30),   TERRAIN_WATER: (30, 60, 120),
            TERRAIN_URBAN:  (90, 80, 70),   TERRAIN_ROAD:  (100, 95, 75),
        }
        # Tints used to distinguish different heights on the elevation layer.
        ELEV_TINT  = [(0, 0, 0), (15, 15, 10), (30, 30, 20)]
        
        self.screen.fill((18, 18, 22)) # Dark theme backdrop.
        cs = self.cell_size

        # ── Step 1: Draw the Terrain and Elevation mesh ──
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                terrain = self.terrain_map[r, c]
                base_color = TC[terrain]
                tint = ELEV_TINT[min(self.elevation[r, c], 2)]
                final_color = tuple(min(255, base_color[i] + tint[i]) for i in range(3))
                pygame.draw.rect(self.screen, final_color, rect)
                pygame.draw.rect(self.screen, (40, 40, 40), rect, 1) # Cell borders.

        # ── Step 2: Atmospheric Fog of War overlay ──
        if self.fog_enabled:
            # Render fog on a specialized transparent surface.
            fog_surf = pygame.Surface((self.grid_size * cs, self.grid_size * cs), pygame.SRCALPHA)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.fog_map[r, c] == 0:
                        pygame.draw.rect(fog_surf, (0, 0, 0, 180), (c * cs, r * cs, cs, cs))
            self.screen.blit(fog_surf, (0, 0))

        # ── Step 3: Game Entities (Crates and Units) ──
        for s in self.supply_drops:
            sr, sc = s["pos"]
            if not self.fog_enabled or self.fog_map[sr, sc]:
                # Draw yellow supply crate with visual indicator.
                self._draw_entity_indicator(sc * cs + cs // 2, sr * cs + cs // 2, (255, 220, 50), "S")

        for u in self.blue_units:
            if u["alive"]: self._draw_unit_premium(u, (50, 140, 255))
        for u in self.red_units:
            # Adversary visibility check: Only show red units if they aren't in the fog.
            if u["alive"] and ((not self.fog_enabled) or self.fog_map[u["pos"][0], u["pos"][1]]):
                self._draw_unit_premium(u, (240, 60, 60))

        # ── Step 4: Dynamics and HUD elements ──
        self.particles.update()
        self.particles.draw(self.screen)
        self._draw_hud_dashboard()

        # Update the hardware display buffer.
        pygame.display.flip()
        self.clock.tick(20) # Maintain consistent simulation speed.

    # Specialized internal plotter for unit entities.
    def _draw_unit_premium(self, u, color):
        """Draws a unit circle accompanied by HP, Morale, and ID indicators."""
        cx = u["pos"][1] * self.cell_size + self.cell_size // 2
        cy = u["pos"][0] * self.cell_size + self.cell_size // 2
        rad = self.cell_size // 3
        
        # High-Morale unit aura (gold ring).
        if u["morale"] >= 120:
            pygame.draw.circle(self.screen, (255, 255, 150), (cx, cy), rad + 3, 2)
        
        # Core unit body.
        pygame.draw.circle(self.screen, color, (cx, cy), rad)
        # ID text.
        label = self.font.render(u["id"], True, (255, 255, 255))
        self.screen.blit(label, (cx - 8, cy - 6))

        # Stat-bars alignment.
        bx = cx - (self.cell_size // 4)
        by = cy + rad + 4
        # HP bar (Green to Red).
        self._draw_stat_bar(bx, by, self.cell_size // 2, 4, u["hp"] / 100, (50, 200, 50))
        # Morale bar (Blue/Cyan).
        self._draw_stat_bar(bx, by + 6, self.cell_size // 2, 2, u["morale"] / MORALE_MAX, (100, 150, 255))

    # Generic bar rendering utility for stats.
    def _draw_stat_bar(self, x, y, w, h, ratio, color):
        pygame.draw.rect(self.screen, (30, 30, 30), (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * max(0, ratio)), h))

    # Generic indicator rendering utility for drops.
    def _draw_entity_indicator(self, x, y, color, char):
        pygame.draw.rect(self.screen, color, (x - 6, y - 6, 12, 12))
        lbl = self.font.render(char, True, (0, 0, 0))
        self.screen.blit(lbl, (x - 3, y - 6))

    # Console and HUD panel overlay.
    def _draw_hud_dashboard(self):
        """Renders the statistics and event logs at the bottom of the window."""
        hud_y = self.grid_size * self.cell_size + 10
        # Global stats.
        header = self.font_lg.render(f"Step: {self.current_step}  |  Operational Status: ACTIVE", True, (220, 220, 220))
        self.screen.blit(header, (20, hud_y))

        # Team columns.
        for i, u in enumerate(self.blue_units):
            txt = self.font.render(f"{u['id']}: HP {u['hp']:>3} AMMO {u['ammo']:>2} F {u['fuel']:>3.0f} M {u['morale']:>3}", True, (50, 140, 255) if u["alive"] else (80, 80, 80))
            self.screen.blit(txt, (20, hud_y + 30 + (i*16)))

        for i, u in enumerate(self.red_units):
            txt = self.font.render(f"{u['id']}: HP {u['hp']:>3} AMMO {u['ammo']:>2} F {u['fuel']:>3.0f} M {u['morale']:>3}", True, (240, 60, 60) if u["alive"] else (80, 80, 80))
            self.screen.blit(txt, (self.window_w // 2, hud_y + 30 + (i*16)))

    # Graceful shutdown of the graphic engine resources.
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# ──────────────────────────────────────────────
# Interactive Runtime Environment
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize the manual play environment.
    env = BattleEnv(render_mode="human", fog_enabled=True)
    env.reset()
    env.render()
    
    print("\n[TDASS Engine] Simulation ready for manual operator commands.")
    print("Commands help: <unit_idx> <action_id> (e.g., '0 4' moves unit B0 Right)")
    
    # Simple interaction loop.
    try:
        while True:
            # Poll Pygame events for window close events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
            
            cmd = input("Tactical Command> ").split()
            if not cmd: continue
            
            idx, act = (int(cmd[0]), int(cmd[1])) if len(cmd) > 1 else (0, int(cmd[0]))
            _, rew, done, _, info = env.step(act, unit_index=idx)
            env.render()
            
            if done:
                print("\n[EXFIL COMPLETE] Match termination reached.")
                break
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Sequence initiated.")
    finally:
        env.close()
