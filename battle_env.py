"""
battle_env.py — Enhanced Battlefield Environment (Phase A+)
==========================================================
A Gymnasium environment for multi-unit tactical simulation with:
  - 6 terrain types with movement costs
  - Multiple Blue / Red units with individual resources (HP, ammo, fuel)
  - Combat mechanics (ranged + melee)
  - Fog of War — units have limited vision radius
  - Supply Drops — resource crates spawn on the map
  - Elevation layer — high ground grants attack bonus
  - Morale system — unit effectiveness scales with events
  - Rich Pygame renderer with terrain, fog overlay, minimap, particles
  - Full OODA-loop data exposure via the info dict

Grid Terrain Codes (terrain_map layer):
  0 = Plains   (cost 1.0)
  1 = Wall      (impassable)
  2 = Forest    (cost 1.5, provides cover)
  3 = Water     (impassable)
  4 = Urban     (cost 2.0, provides cover)
  5 = Road      (cost 0.5, fast movement)

Unit Layer (unit_map):
  0 = Empty
  1 = Blue unit
  2 = Red unit
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy
import random
import math

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TERRAIN_PLAINS = 0
TERRAIN_WALL   = 1
TERRAIN_FOREST = 2
TERRAIN_WATER  = 3
TERRAIN_URBAN  = 4
TERRAIN_ROAD   = 5

TERRAIN_NAMES = {0: "Plains", 1: "Wall", 2: "Forest", 3: "Water", 4: "Urban", 5: "Road"}

# Movement cost multiplier per terrain; None = impassable
TERRAIN_COST = {
    TERRAIN_PLAINS: 1.0,
    TERRAIN_WALL:   None,
    TERRAIN_FOREST: 1.5,
    TERRAIN_WATER:  None,
    TERRAIN_URBAN:  2.0,
    TERRAIN_ROAD:   0.5,
}

# Cover bonus — reduces incoming damage by this fraction
TERRAIN_COVER = {
    TERRAIN_PLAINS: 0.0,
    TERRAIN_WALL:   0.0,
    TERRAIN_FOREST: 0.3,
    TERRAIN_WATER:  0.0,
    TERRAIN_URBAN:  0.4,
    TERRAIN_ROAD:   0.0,
}

# Base fuel cost per step on Plains
BASE_FUEL_COST = 2.0

# Combat constants
MELEE_DAMAGE   = 30
MELEE_AMMO     = 0
RANGED_DAMAGE  = 20
RANGED_AMMO    = 5
RANGED_RANGE   = 3

# Fog of War
FOG_VISION_RADIUS = 4  # Manhattan distance each unit can see

# Supply drop constants
SUPPLY_HP    = 25
SUPPLY_AMMO  = 15
SUPPLY_FUEL  = 30
SUPPLY_SPAWN_CHANCE = 0.08  # probability per step of a new crate

# Elevation
ELEVATION_DAMAGE_BONUS = 0.25  # +25% damage from high ground

# Morale
MORALE_DEFAULT   = 100
MORALE_KILL_ALLY = -20   # morale lost when friendly dies
MORALE_KILL_FOE  = 15    # morale gained when enemy killed
MORALE_LOW_HP    = -10   # morale penalty when HP < 30
MORALE_MIN       = 20
MORALE_MAX       = 150

# Actions: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right, 5=RangedAttack
NUM_ACTIONS = 6

# ──────────────────────────────────────────────
# Unit helper
# ──────────────────────────────────────────────
def make_unit(uid, team, row, col, hp=100, ammo=50, fuel=100):
    """Create a unit state dictionary."""
    return {
        "id":      uid,
        "team":    team,
        "pos":     [row, col],
        "hp":      hp,
        "ammo":    ammo,
        "fuel":    fuel,
        "alive":   True,
        "morale":  MORALE_DEFAULT,
        "kills":   0,
    }


# ──────────────────────────────────────────────
# Default 10×10 terrain & elevation maps
# ──────────────────────────────────────────────
def default_terrain():
    """Hand-crafted 10×10 terrain with tactical features."""
    t = np.zeros((10, 10), dtype=int)
    t[3, 0:5] = TERRAIN_FOREST
    t[4, 0:3] = TERRAIN_FOREST
    t[2:8, 7] = TERRAIN_WATER
    t[4:6, 4:6] = TERRAIN_URBAN
    t[8, :]    = TERRAIN_ROAD
    t[1, 5]    = TERRAIN_WALL
    t[2, 5]    = TERRAIN_WALL
    t[6, 2]    = TERRAIN_WALL
    t[6, 3]    = TERRAIN_WALL
    return t


def default_elevation(grid_size=10):
    """Elevation layer: 0=low, 1=mid, 2=high.  Hills in centre & corners."""
    e = np.zeros((grid_size, grid_size), dtype=int)
    e[4:6, 4:6] = 2   # urban hilltop
    e[0:2, 0:2] = 1   # blue spawn ridge
    e[8:10, 8:10] = 1  # red spawn ridge
    e[3, 7] = 2        # sniper bluff
    return e


# ──────────────────────────────────────────────
# Particle system (for renderer)
# ──────────────────────────────────────────────
class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size=3):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.color = color
        self.size = size


class _ParticleSystem:
    """Lightweight particle emitter for combat FX."""

    def __init__(self):
        self.particles: list[_Particle] = []

    def emit(self, x, y, color, count=8, speed=3.0, life=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            spd   = random.uniform(0.5, speed)
            self.particles.append(_Particle(
                x, y,
                math.cos(angle) * spd, math.sin(angle) * spd,
                random.randint(life // 2, life),
                color, random.randint(2, 4)
            ))

    def update(self):
        alive = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    def draw(self, surface):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * p.life / 15)))
            c = (min(255, p.color[0]), min(255, p.color[1]), min(255, p.color[2]))
            pygame.draw.circle(surface, c, (int(p.x), int(p.y)), p.size)


# ──────────────────────────────────────────────
# BattleEnv
# ──────────────────────────────────────────────
class BattleEnv(gym.Env):
    """
    Enhanced Battlefield Gymnasium Environment.

    Observation: dict with keys
      - "terrain_map": (H, W) int   — terrain type per cell
      - "unit_map":    (H, W) int   — 0=empty, 1=blue, 2=red
      - "fog_map":     (H, W) int   — 1=visible to blue, 0=fogged
      - "elevation":   (H, W) int   — elevation level per cell

    Action: int 0-5  (applied to the *active* blue unit via step)
    """

    metadata = {"render_modes": ["human", "ansi"]}

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
        super().__init__()
        self.grid_size   = grid_size
        self.render_mode = render_mode
        self.num_blue    = num_blue
        self.num_red     = num_red
        self.max_steps   = max_steps
        self.current_step = 0
        self.fog_enabled  = fog_enabled

        # Terrain
        self.terrain_map = np.array(terrain_map, dtype=int) if terrain_map is not None else default_terrain()
        # Elevation
        self.elevation = np.array(elevation_map, dtype=int) if elevation_map is not None else default_elevation(grid_size)

        # Fog of war (blue's perspective)
        self.fog_map = np.ones((grid_size, grid_size), dtype=int)

        # Supply drops: list of {"pos": [r,c], "hp", "ammo", "fuel"}
        self.supply_drops: list[dict] = []

        # Action & observation spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "terrain_map": spaces.Box(0, 5, shape=(grid_size, grid_size), dtype=int),
            "unit_map":    spaces.Box(0, 2, shape=(grid_size, grid_size), dtype=int),
        })

        # Pygame rendering state
        self.cell_size   = 64
        self.hud_height  = 140
        self.minimap_size = 120
        self.window_w    = self.grid_size * self.cell_size + self.minimap_size + 20
        self.window_h    = self.grid_size * self.cell_size + self.hud_height
        self.screen      = None
        self.clock       = None
        self.font        = None
        self.particles   = _ParticleSystem()

        # Units
        self.blue_units: list[dict] = []
        self.red_units:  list[dict] = []
        self.unit_map    = np.zeros((grid_size, grid_size), dtype=int)

        # Combat log
        self.combat_log: list[str] = []

    # ─── helpers ───────────────────────────────

    def _passable(self, row, col):
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return False
        return TERRAIN_COST[self.terrain_map[row, col]] is not None

    def _occupied_by(self, row, col, exclude_uid=None):
        for u in self.blue_units + self.red_units:
            if u["alive"] and u["pos"] == [row, col] and u["id"] != exclude_uid:
                return u
        return None

    def _rebuild_unit_map(self):
        self.unit_map.fill(0)
        for u in self.blue_units:
            if u["alive"]:
                self.unit_map[u["pos"][0], u["pos"][1]] = 1
        for u in self.red_units:
            if u["alive"]:
                self.unit_map[u["pos"][0], u["pos"][1]] = 2

    def _find_spawn(self, preferred_positions, taken):
        for pos in preferred_positions:
            r, c = pos
            if self._passable(r, c) and (r, c) not in taken:
                taken.add((r, c))
                return [r, c]
        for _ in range(200):
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if self._passable(r, c) and (r, c) not in taken:
                taken.add((r, c))
                return [r, c]
        return [0, 0]

    def _alive_blue(self):
        return [u for u in self.blue_units if u["alive"]]

    def _alive_red(self):
        return [u for u in self.red_units if u["alive"]]

    # ─── Fog of War ────────────────────────────

    def _update_fog(self):
        """Recompute fog map from blue unit positions."""
        if not self.fog_enabled:
            self.fog_map.fill(1)
            return
        self.fog_map.fill(0)
        for u in self._alive_blue():
            r0, c0 = u["pos"]
            for dr in range(-FOG_VISION_RADIUS, FOG_VISION_RADIUS + 1):
                for dc in range(-FOG_VISION_RADIUS, FOG_VISION_RADIUS + 1):
                    if abs(dr) + abs(dc) <= FOG_VISION_RADIUS:
                        nr, nc = r0 + dr, c0 + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            self.fog_map[nr, nc] = 1

    # ─── Supply Drops ─────────────────────────

    def _maybe_spawn_supply(self):
        """Randomly place a supply crate on an empty passable cell."""
        if random.random() < SUPPLY_SPAWN_CHANCE:
            for _ in range(50):
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                if self._passable(r, c) and self._occupied_by(r, c) is None:
                    # Make sure there's no supply already there
                    if not any(s["pos"] == [r, c] for s in self.supply_drops):
                        self.supply_drops.append({
                            "pos": [r, c], "hp": SUPPLY_HP,
                            "ammo": SUPPLY_AMMO, "fuel": SUPPLY_FUEL,
                        })
                        self.combat_log.append(f"SUPPLY: Crate dropped at ({r},{c})")
                        return

    def _check_supply_pickup(self, unit):
        """If unit is standing on a supply crate, consume it."""
        for s in self.supply_drops:
            if s["pos"] == unit["pos"]:
                unit["hp"]   = min(100, unit["hp"]   + s["hp"])
                unit["ammo"] = min(50,  unit["ammo"] + s["ammo"])
                unit["fuel"] = min(100, unit["fuel"]  + s["fuel"])
                self.supply_drops.remove(s)
                self.combat_log.append(
                    f"PICKUP: {unit['id']} collected supply (+{s['hp']}HP +{s['ammo']}A +{s['fuel']}F)"
                )
                return

    # ─── Morale helpers ────────────────────────

    def _adjust_morale(self, unit, delta):
        unit["morale"] = max(MORALE_MIN, min(MORALE_MAX, unit["morale"] + delta))

    def _morale_multiplier(self, unit):
        """Morale affects damage output: 100 = 1.0×, 50 = 0.75×, 150 = 1.25×."""
        return 0.5 + 0.5 * (unit["morale"] / MORALE_DEFAULT)

    def _broadcast_morale(self, dead_unit):
        """Update morale for all units when someone dies."""
        for u in self.blue_units + self.red_units:
            if not u["alive"]:
                continue
            if u["team"] == dead_unit["team"]:
                self._adjust_morale(u, MORALE_KILL_ALLY)
            else:
                self._adjust_morale(u, MORALE_KILL_FOE)

    # ─── Elevation helper ─────────────────────

    def _elevation_bonus(self, attacker, defender):
        """Return damage multiplier based on elevation difference."""
        a_elev = self.elevation[attacker["pos"][0], attacker["pos"][1]]
        d_elev = self.elevation[defender["pos"][0], defender["pos"][1]]
        if a_elev > d_elev:
            return 1.0 + ELEVATION_DAMAGE_BONUS
        return 1.0

    # ─── gym interface ─────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.combat_log   = []
        self.supply_drops = []

        taken = set()
        blue_spawns = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
        self.blue_units = []
        for i in range(self.num_blue):
            pos = self._find_spawn(blue_spawns, taken)
            self.blue_units.append(make_unit(f"B{i}", "blue", pos[0], pos[1]))

        red_spawns = [(9, 9), (9, 8), (8, 9), (8, 8), (9, 6), (7, 9)]
        self.red_units = []
        for i in range(self.num_red):
            pos = self._find_spawn(red_spawns, taken)
            self.red_units.append(make_unit(f"R{i}", "red", pos[0], pos[1]))

        self._rebuild_unit_map()
        self._update_fog()
        obs = self._get_obs()
        return obs, self._get_info()

    def _get_obs(self):
        return {
            "terrain_map": self.terrain_map.copy(),
            "unit_map":    self.unit_map.copy(),
            "fog_map":     self.fog_map.copy(),
            "elevation":   self.elevation.copy(),
        }

    def _get_info(self):
        return {
            "blue_units":    copy.deepcopy(self.blue_units),
            "red_units":     copy.deepcopy(self.red_units),
            "step":          self.current_step,
            "combat_log":    list(self.combat_log),
            "supply_drops":  copy.deepcopy(self.supply_drops),
        }

    def step(self, action, unit_index=0):
        """
        Execute an action for blue_units[unit_index].
        Returns:  obs, reward, terminated, truncated, info
        """
        self.current_step += 1
        self.combat_log = []
        reward = 0.0

        blue = self.blue_units[unit_index] if unit_index < len(self.blue_units) else None
        if blue is None or not blue["alive"]:
            return self._get_obs(), 0, False, False, self._get_info()

        # Movement (0-4)
        if action <= 4:
            reward += self._do_move(blue, action)
        # Ranged (5)
        elif action == 5:
            reward += self._do_ranged_attack(blue)

        # Supply check & spawn
        self._check_supply_pickup(blue)
        self._maybe_spawn_supply()

        # Morale low-HP tick
        for u in self.blue_units + self.red_units:
            if u["alive"] and u["hp"] < 30:
                self._adjust_morale(u, MORALE_LOW_HP)

        self._rebuild_unit_map()
        self._update_fog()

        # Win / loss
        terminated = False
        if not self._alive_red():
            reward += 50
            terminated = True
            self.combat_log.append("VICTORY: All Red units eliminated!")
        elif not self._alive_blue():
            reward -= 50
            terminated = True
            self.combat_log.append("DEFEAT: All Blue units eliminated!")

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ─── movement logic ───────────────────────

    def _do_move(self, unit, action):
        if action == 0:
            return 0.0

        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        nr, nc = unit["pos"][0] + dr, unit["pos"][1] + dc

        if not self._passable(nr, nc):
            return -0.5

        terrain = self.terrain_map[nr, nc]
        fuel_cost = BASE_FUEL_COST * TERRAIN_COST[terrain]
        if unit["fuel"] < fuel_cost:
            self.combat_log.append(f"{unit['id']}: Out of fuel!")
            return -1.0

        occupant = self._occupied_by(nr, nc, exclude_uid=unit["id"])
        if occupant is not None:
            if occupant["team"] == unit["team"]:
                return -0.5
            else:
                return self._do_melee(unit, occupant, nr, nc, fuel_cost)

        unit["pos"] = [nr, nc]
        unit["fuel"] -= fuel_cost
        return 0.0

    def _do_melee(self, attacker, defender, nr, nc, fuel_cost):
        attacker["fuel"] -= fuel_cost

        cover = TERRAIN_COVER[self.terrain_map[defender["pos"][0], defender["pos"][1]]]
        elev  = self._elevation_bonus(attacker, defender)
        morale_mult = self._morale_multiplier(attacker)
        damage = int(MELEE_DAMAGE * (1.0 - cover) * elev * morale_mult)
        defender["hp"] -= damage

        self.combat_log.append(
            f"MELEE: {attacker['id']}→{defender['id']} {damage}dmg "
            f"(cover {cover:.0%}, elev ×{elev:.2f}). HP:{defender['hp']}"
        )

        # Particle FX
        cx = defender["pos"][1] * self.cell_size + self.cell_size // 2
        cy = defender["pos"][0] * self.cell_size + self.cell_size // 2
        self.particles.emit(cx, cy, (255, 200, 50), count=12, speed=4.0)

        reward = 2.0

        if defender["hp"] <= 0:
            defender["alive"] = False
            attacker["pos"] = [nr, nc]
            attacker["kills"] += 1
            self._broadcast_morale(defender)
            self.combat_log.append(f"KILL: {defender['id']} eliminated!")
            reward += 10.0
        else:
            counter_cover = TERRAIN_COVER[self.terrain_map[attacker["pos"][0], attacker["pos"][1]]]
            counter_dmg = int(MELEE_DAMAGE * 0.5 * (1.0 - counter_cover))
            attacker["hp"] -= counter_dmg
            self.combat_log.append(
                f"COUNTER: {defender['id']}→{attacker['id']} {counter_dmg}dmg. HP:{attacker['hp']}"
            )
            if attacker["hp"] <= 0:
                attacker["alive"] = False
                self._broadcast_morale(attacker)
                self.combat_log.append(f"KILL: {attacker['id']} eliminated!")
                reward -= 15.0

        return reward

    def _do_ranged_attack(self, unit):
        if unit["ammo"] < RANGED_AMMO:
            self.combat_log.append(f"{unit['id']}: Not enough ammo!")
            return -1.0

        enemies = self._alive_red() if unit["team"] == "blue" else self._alive_blue()
        best_enemy, best_dist = None, float("inf")
        for e in enemies:
            d = abs(unit["pos"][0] - e["pos"][0]) + abs(unit["pos"][1] - e["pos"][1])
            if d <= RANGED_RANGE and d < best_dist:
                best_dist, best_enemy = d, e

        if best_enemy is None:
            self.combat_log.append(f"{unit['id']}: No targets in range ({RANGED_RANGE}).")
            return -0.5

        unit["ammo"] -= RANGED_AMMO
        cover = TERRAIN_COVER[self.terrain_map[best_enemy["pos"][0], best_enemy["pos"][1]]]
        elev  = self._elevation_bonus(unit, best_enemy)
        morale_mult = self._morale_multiplier(unit)
        damage = int(RANGED_DAMAGE * (1.0 - cover) * elev * morale_mult)
        best_enemy["hp"] -= damage

        self.combat_log.append(
            f"RANGED: {unit['id']}→{best_enemy['id']} {damage}dmg "
            f"(rng {best_dist}, cover {cover:.0%}, elev ×{elev:.2f}). HP:{best_enemy['hp']}"
        )

        # Tracer particle from shooter to target
        sx = unit["pos"][1] * self.cell_size + self.cell_size // 2
        sy = unit["pos"][0] * self.cell_size + self.cell_size // 2
        tx = best_enemy["pos"][1] * self.cell_size + self.cell_size // 2
        ty = best_enemy["pos"][0] * self.cell_size + self.cell_size // 2
        self.particles.emit(tx, ty, (255, 100, 30), count=10, speed=3.5)
        self.particles.emit(sx, sy, (200, 200, 255), count=4, speed=1.5, life=8)

        reward = 2.0
        if best_enemy["hp"] <= 0:
            best_enemy["alive"] = False
            unit["kills"] += 1
            self._broadcast_morale(best_enemy)
            self.combat_log.append(f"KILL: {best_enemy['id']} eliminated!")
            reward += 10.0

        return reward

    # ─── Red convenience ───────────────────────

    def move_red(self, unit_index, action):
        """Move a red unit (used by simulation / red_strategy)."""
        if unit_index >= len(self.red_units):
            return 0.0
        red = self.red_units[unit_index]
        if not red["alive"]:
            return 0.0

        if action <= 4:
            self._do_move(red, action)
        elif action == 5:
            self._do_ranged_attack(red)

        self._check_supply_pickup(red)
        self._rebuild_unit_map()
        self._update_fog()
        return 0.0

    def get_unit_terrain(self, unit):
        return TERRAIN_NAMES.get(self.terrain_map[unit["pos"][0], unit["pos"][1]], "Unknown")

    # ─── Backward-compatible properties ────────

    @property
    def blue_pos(self):
        alive = self._alive_blue()
        return alive[0]["pos"] if alive else None

    @property
    def red_pos(self):
        alive = self._alive_red()
        return alive[0]["pos"] if alive else None

    @property
    def obstacles(self):
        result = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if TERRAIN_COST[self.terrain_map[r, c]] is None:
                    result.append([r, c])
        return result

    # ─── Rendering ─────────────────────────────

    def render(self):
        if self.render_mode == "ansi":
            self._render_ansi()
        elif self.render_mode == "human":
            self._render_pygame()

    def _render_ansi(self):
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
                if self.fog_enabled and self.fog_map[r, c] == 0:
                    cell = " ?"
                else:
                    for u in self.blue_units:
                        if u["alive"] and u["pos"] == [r, c]:
                            cell = f"\033[94mB{u['id'][-1]}\033[0m"
                            break
                    if cell is None:
                        for u in self.red_units:
                            if u["alive"] and u["pos"] == [r, c]:
                                cell = f"\033[91mR{u['id'][-1]}\033[0m"
                                break
                    # Supply crate
                    if cell is None:
                        for s in self.supply_drops:
                            if s["pos"] == [r, c]:
                                cell = f"\033[93m+S\033[0m"
                                break
                    if cell is None:
                        cell = f" {symbols[self.terrain_map[r, c]]}"
                row_str += cell + " "
            print(row_str)

        print("\n\033[94m── Blue Forces ──\033[0m")
        for u in self.blue_units:
            status = "ALIVE" if u["alive"] else "DEAD"
            print(f"  {u['id']}: pos={u['pos']} HP={u['hp']} A={u['ammo']} "
                  f"F={u['fuel']:.0f} M={u['morale']} K={u['kills']} [{status}]")
        print("\033[91m── Red Forces ──\033[0m")
        for u in self.red_units:
            status = "ALIVE" if u["alive"] else "DEAD"
            print(f"  {u['id']}: pos={u['pos']} HP={u['hp']} A={u['ammo']} "
                  f"F={u['fuel']:.0f} M={u['morale']} K={u['kills']} [{status}]")

        if self.combat_log:
            print("\n\033[93m── Combat Log ──\033[0m")
            for msg in self.combat_log:
                print(f"  ⚔ {msg}")

    def _render_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("TDSS — Enhanced Tactical Battlefield")
            self.screen = pygame.display.set_mode((self.window_w, self.window_h))
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("consolas", 12)
            self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)

        # ── Colour palette ──
        TC = {
            TERRAIN_PLAINS: (50, 60, 45),   TERRAIN_WALL: (80, 80, 80),
            TERRAIN_FOREST: (20, 80, 30),   TERRAIN_WATER: (30, 60, 120),
            TERRAIN_URBAN:  (90, 80, 70),   TERRAIN_ROAD:  (100, 95, 75),
        }
        ELEV_TINT  = [(0, 0, 0), (15, 15, 10), (30, 30, 20)]
        BLUE_CLR   = (50, 140, 255)
        RED_CLR    = (240, 60, 60)
        SUPPLY_CLR = (255, 220, 50)
        GRID_LINE  = (40, 40, 40)
        HUD_BG     = (18, 18, 22)
        TXT        = (200, 200, 200)
        FOG_CLR    = (10, 10, 15)

        self.screen.fill(HUD_BG)
        cs = self.cell_size
        grid_px = self.grid_size * cs

        # ── Terrain + elevation ──
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                terrain = self.terrain_map[r, c]
                base = TC[terrain]
                tint = ELEV_TINT[min(self.elevation[r, c], 2)]
                color = tuple(min(255, base[i] + tint[i]) for i in range(3))
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

                # Elevation indicator (small triangle for high ground)
                if self.elevation[r, c] >= 2:
                    cx, cy = rect.centerx, rect.y + 4
                    pygame.draw.polygon(self.screen, (180, 180, 120),
                        [(cx - 4, cy + 6), (cx + 4, cy + 6), (cx, cy)])

        # ── Fog overlay ──
        if self.fog_enabled:
            fog_surf = pygame.Surface((grid_px, grid_px), pygame.SRCALPHA)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.fog_map[r, c] == 0:
                        fog_rect = pygame.Rect(c * cs, r * cs, cs, cs)
                        pygame.draw.rect(fog_surf, (0, 0, 0, 180), fog_rect)
            self.screen.blit(fog_surf, (0, 0))

        # ── Supply crates ──
        for s in self.supply_drops:
            sr, sc = s["pos"]
            if not self.fog_enabled or self.fog_map[sr, sc]:
                cx = sc * cs + cs // 2
                cy = sr * cs + cs // 2
                pygame.draw.rect(self.screen, SUPPLY_CLR,
                    (cx - 6, cy - 6, 12, 12))
                pygame.draw.rect(self.screen, (180, 150, 20),
                    (cx - 6, cy - 6, 12, 12), 1)
                lbl = self.font.render("+", True, (0, 0, 0))
                self.screen.blit(lbl, (cx - 3, cy - 6))

        # ── Units ──
        for u in self.blue_units:
            if u["alive"]:
                self._draw_unit(u, BLUE_CLR)
        for u in self.red_units:
            if u["alive"]:
                visible = (not self.fog_enabled) or self.fog_map[u["pos"][0], u["pos"][1]]
                if visible:
                    self._draw_unit(u, RED_CLR)

        # ── Particles ──
        self.particles.update()
        self.particles.draw(self.screen)

        # ── Minimap (right side) ──
        mm_x = grid_px + 10
        mm_y = 10
        mm_cs = self.minimap_size // self.grid_size
        pygame.draw.rect(self.screen, (30, 30, 35),
            (mm_x - 2, mm_y - 2, self.minimap_size + 4, self.minimap_size + 4))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(mm_x + c * mm_cs, mm_y + r * mm_cs, mm_cs, mm_cs)
                terrain = self.terrain_map[r, c]
                pygame.draw.rect(self.screen, TC[terrain], rect)
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

        lbl = self.font_lg.render("MINIMAP", True, TXT)
        self.screen.blit(lbl, (mm_x, mm_y + self.minimap_size + 4))

        # ── HUD Panel ──
        hud_y = grid_px + 5
        step_text = self.font_lg.render(
            f"Step: {self.current_step}/{self.max_steps}   "
            f"Supplies: {len(self.supply_drops)}", True, TXT)
        self.screen.blit(step_text, (10, hud_y))

        # Blue status
        bx, by = 10, hud_y + 22
        for u in self.blue_units:
            color = BLUE_CLR if u["alive"] else (60, 60, 60)
            txt = (f"{u['id']} HP:{u['hp']:>3} A:{u['ammo']:>2} "
                   f"F:{u['fuel']:>3.0f} M:{u['morale']:>3} K:{u['kills']}")
            surf = self.font.render(txt, True, color)
            self.screen.blit(surf, (bx, by))
            by += 16

        # Red status
        rx, ry = self.window_w // 2, hud_y + 22
        for u in self.red_units:
            color = RED_CLR if u["alive"] else (60, 60, 60)
            txt = (f"{u['id']} HP:{u['hp']:>3} A:{u['ammo']:>2} "
                   f"F:{u['fuel']:>3.0f} M:{u['morale']:>3} K:{u['kills']}")
            surf = self.font.render(txt, True, color)
            self.screen.blit(surf, (rx, ry))
            ry += 16

        # Combat log
        log_y = hud_y + 90
        for msg in self.combat_log[-3:]:
            surf = self.font.render(f"⚔ {msg[:80]}", True, (255, 200, 80))
            self.screen.blit(surf, (10, log_y))
            log_y += 14

        pygame.display.flip()
        self.clock.tick(15)

    def _draw_unit(self, u, color):
        """Draw a single unit circle + HP bar + morale bar + label."""
        cs = self.cell_size
        cx = u["pos"][1] * cs + cs // 2
        cy = u["pos"][0] * cs + cs // 2
        radius = cs // 3

        # Morale glow ring
        if u["morale"] >= 120:
            pygame.draw.circle(self.screen, (255, 255, 150), (cx, cy), radius + 3, 2)

        pygame.draw.circle(self.screen, color, (cx, cy), radius)

        # HP bar
        bar_w, bar_h = cs // 2, 4
        bx = cx - bar_w // 2
        by = cy + radius + 3
        pygame.draw.rect(self.screen, (40, 40, 40), (bx, by, bar_w, bar_h))
        fill_w = max(0, int(bar_w * u["hp"] / 100))
        fill_c = (50, 200, 50) if u["hp"] > 50 else (200, 200, 0) if u["hp"] > 25 else (200, 50, 50)
        pygame.draw.rect(self.screen, fill_c, (bx, by, fill_w, bar_h))

        # Morale bar (thin, below HP)
        my = by + 5
        pygame.draw.rect(self.screen, (40, 40, 40), (bx, my, bar_w, 2))
        m_fill = max(0, int(bar_w * u["morale"] / MORALE_MAX))
        m_c = (100, 150, 255) if u["morale"] >= 80 else (200, 100, 50)
        pygame.draw.rect(self.screen, m_c, (bx, my, m_fill, 2))

        # Label
        label = self.font.render(u["id"], True, (255, 255, 255))
        self.screen.blit(label, (cx - 8, cy - 6))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ──────────────────────────────────────────────
# Interactive Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    env = BattleEnv(render_mode="human", num_blue=2, num_red=2, fog_enabled=True)
    obs, info = env.reset()
    env.render()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  TDSS Enhanced Battlefield v2 — Interactive         ║")
    print("║  Actions: 0=Stay 1=Up 2=Down 3=Left 4=Right        ║")
    print("║           5=Ranged Attack                           ║")
    print("║  Format:  <unit_index> <action>                     ║")
    print("║  Example: 0 1  (move unit B0 up)                    ║")
    print("║  NEW: Fog of War · Supply Drops · Elevation · Morale║")
    print("╚══════════════════════════════════════════════════════╝")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        try:
            raw = input("Command> ").strip()
            if not raw:
                continue

            parts = raw.split()
            if len(parts) == 1:
                uid, act = 0, int(parts[0])
            else:
                uid, act = int(parts[0]), int(parts[1])

            if act not in range(NUM_ACTIONS):
                print(f"Invalid action. Use 0-{NUM_ACTIONS-1}.")
                continue

            obs, reward, terminated, truncated, info = env.step(act, unit_index=uid)
            env.render()

            if reward != 0:
                print(f"  Reward: {reward:+.1f}")

            if terminated:
                print("Episode finished. Resetting...")
                env.reset()
                env.render()
            elif truncated:
                print("Max steps reached. Resetting...")
                env.reset()
                env.render()

        except ValueError:
            print("Enter: <unit_index> <action>  or just <action> for unit 0.")
        except (KeyboardInterrupt, EOFError):
            break

    env.close()
