"""
simulation.py — Enhanced Data Generation Pipeline (Phase A5+)
==============================================================
Runs N automated matches between Red bots and Blue random movers.
Logs rich battlefield state per step to CSV for Oracle training.

New columns (v2):
  morale, elevation, fog_visible, nearby_supplies, kills
"""

import csv
import random
import time
from battle_env import BattleEnv, TERRAIN_NAMES, FOG_VISION_RADIUS
from Element_Logic.red_strategy import get_red_action


def run_simulation(matches=5000, target_rows=100000, output_file="battle_data.csv"):
    """
    Run automated matches and log enhanced battle data.

    Args:
        matches:      max number of episodes to run
        target_rows:  stop once this many data rows are collected
        output_file:  path to output CSV
    """
    env = BattleEnv(render_mode="ansi", num_blue=2, num_red=2,
                    max_steps=200, fog_enabled=True)

    print(f"╔═══════════════════════════════════════════════════╗")
    print(f"║  TDSS Data Generation Pipeline v2                ║")
    print(f"║  Matches: {matches:<6}  Target rows: {target_rows:<8}       ║")
    print(f"║  New: Morale · Elevation · Fog · Supplies        ║")
    print(f"╚═══════════════════════════════════════════════════╝")

    start_time = time.time()
    total_rows = 0
    total_blue_wins = 0
    total_red_wins  = 0
    total_draws     = 0

    # Extended header
    header = [
        "step", "match_id",
        "blue_id", "blue_x", "blue_y", "blue_hp", "blue_ammo", "blue_fuel",
        "blue_terrain", "blue_morale", "blue_elevation", "blue_kills",
        "red_id", "red_x", "red_y", "red_hp", "red_ammo", "red_fuel",
        "red_terrain", "red_morale", "red_elevation", "red_kills",
        "red_posture", "red_prev_action", "red_next_action",
        "distance", "fog_visible", "nearby_supplies", "outcome",
    ]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for match in range(matches):
            obs, info = env.reset()
            terminated = False
            truncated  = False
            step_num   = 0
            outcome    = "ongoing"

            red_prev_actions = {u["id"]: 0 for u in env.red_units}

            while not terminated and not truncated:
                step_num += 1

                alive_blues = [u for u in env.blue_units if u["alive"]]
                alive_reds  = [u for u in env.red_units  if u["alive"]]

                if not alive_blues or not alive_reds:
                    break

                # ── Red decisions + logging ──
                red_actions = {}
                for red in alive_reds:
                    action, posture = get_red_action(
                        red, alive_blues, env.grid_size,
                        env.terrain_map, env.red_units
                    )
                    red_actions[red["id"]] = (action, posture)

                    # Nearest blue
                    nearest_blue = min(
                        alive_blues,
                        key=lambda b: abs(red["pos"][0] - b["pos"][0]) +
                                      abs(red["pos"][1] - b["pos"][1])
                    )
                    dist = (abs(red["pos"][0] - nearest_blue["pos"][0]) +
                            abs(red["pos"][1] - nearest_blue["pos"][1]))

                    # Terrain names
                    r_terrain = TERRAIN_NAMES.get(
                        env.terrain_map[red["pos"][0], red["pos"][1]], "Plains")
                    b_terrain = TERRAIN_NAMES.get(
                        env.terrain_map[nearest_blue["pos"][0], nearest_blue["pos"][1]], "Plains")

                    # Elevation
                    r_elev = int(env.elevation[red["pos"][0], red["pos"][1]])
                    b_elev = int(env.elevation[nearest_blue["pos"][0], nearest_blue["pos"][1]])

                    # Fog visibility (can blue see this red?)
                    fog_vis = int(env.fog_map[red["pos"][0], red["pos"][1]])

                    # Nearby supply count (within 3 Manhattan distance of red)
                    nearby_sup = sum(
                        1 for s in env.supply_drops
                        if abs(s["pos"][0] - red["pos"][0]) +
                           abs(s["pos"][1] - red["pos"][1]) <= 3
                    )

                    writer.writerow([
                        step_num, match,
                        nearest_blue["id"], nearest_blue["pos"][0], nearest_blue["pos"][1],
                        nearest_blue["hp"], nearest_blue["ammo"], nearest_blue["fuel"],
                        b_terrain, nearest_blue.get("morale", 100), b_elev,
                        nearest_blue.get("kills", 0),
                        red["id"], red["pos"][0], red["pos"][1],
                        red["hp"], red["ammo"], red["fuel"],
                        r_terrain, red.get("morale", 100), r_elev,
                        red.get("kills", 0),
                        posture, red_prev_actions.get(red["id"], 0), action,
                        dist, fog_vis, nearby_sup, outcome,
                    ])
                    total_rows += 1

                # ── Execute Blue actions (random) ──
                for i, blue in enumerate(env.blue_units):
                    if blue["alive"]:
                        blue_action = random.randint(0, 5)  # now includes ranged
                        obs, reward, terminated, truncated, info = env.step(
                            blue_action, unit_index=i)
                        if terminated or truncated:
                            break

                if terminated or truncated:
                    break

                # ── Execute Red actions ──
                for i, red in enumerate(env.red_units):
                    if red["alive"] and red["id"] in red_actions:
                        action, posture = red_actions[red["id"]]
                        env.move_red(i, action)
                        red_prev_actions[red["id"]] = action

                # Check end conditions after red moves
                if not env._alive_blue():
                    terminated = True
                elif not env._alive_red():
                    terminated = True

            # Match outcome
            blue_alive = len(env._alive_blue())
            red_alive  = len(env._alive_red())
            if blue_alive > 0 and red_alive == 0:
                total_blue_wins += 1
            elif red_alive > 0 and blue_alive == 0:
                total_red_wins += 1
            else:
                total_draws += 1

            # Progress
            if (match + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(f"  Match {match+1:>5}/{matches}  |  Rows: {total_rows:>7}  |  "
                      f"Rate: {rate:.0f} rows/s  |  B:{total_blue_wins} R:{total_red_wins} D:{total_draws}")

            if total_rows >= target_rows:
                print(f"\n  ✔ Target of {target_rows} rows reached at match {match+1}.")
                break

    elapsed = time.time() - start_time
    print(f"\n{'═'*55}")
    print(f"  Simulation Complete!")
    print(f"  Total rows:    {total_rows:,}")
    print(f"  Total matches: {match+1}")
    print(f"  Blue wins:     {total_blue_wins}")
    print(f"  Red wins:      {total_red_wins}")
    print(f"  Draws:         {total_draws}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Output:        {output_file}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    run_simulation(matches=5000, target_rows=100000)
