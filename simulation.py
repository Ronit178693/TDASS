# Automated Tactical Data Generation Pipeline for synthesizing behavior datasets.
"""
# This script orchestrates massive-scale simulations to generate high-fidelity behavioral data for the Oracle.
simulation.py — Enhanced Data Generation Pipeline (Phase A5+)
==============================================================
# It runs thousands of autonomous matches between strategic Red bots and stochastic Blue agents.
# Each specific step in the timeline is captured as a CSV record to train the LSTM-based intelligence.

# Version 2.0 Feature Set:
# Captures secondary tactical signals like morale, elevation advantages, and fog-of-war visibility.
  Morale, Elevation, Fog_Visible, Nearby_Supplies, Kills
"""

# Import standard library for writing structured tabular data.
import csv
# Import random for stochastic blue team behavior and initial coordinate variance.
import random
# Import time for monitoring pipeline performance and rows-per-second throughput.
import time
# Lifecycle and constants from our battlefield environment.
from battle_env import BattleEnv, TERRAIN_NAMES, FOG_VISION_RADIUS
# Strategic logic for the Red adversary; this acts as the 'Teacher' for the Oracle.
from Element_Logic.red_strategy import get_red_action


def run_simulation(matches=5000, target_rows=100000, output_file="battle_data.csv"):
    """
    Executes an automated engagement loop to collect behavioral training data.

    Args:
        matches: The upper limit on the number of individual combat scenarios to run.
        target_rows: The primary stopping condition; halts once enough history is recorded for large-scale training.
        output_file: The destination CSV path where the temporal features will be stored.
    """
    # Instantiate an instance of the tactical environment with specific rules (Fog, multi-unit).
    # Render mode 'ansi' is chosen here for performance, minimizing overhead compared to Pygame.
    env = BattleEnv(render_mode="ansi", num_blue=2, num_red=2,
                    max_steps=200, fog_enabled=True)

    # ── Console Interface Header ──
    print(f"╔═══════════════════════════════════════════════════╗")
    print(f"║  TDSS Data Generation Pipeline v2.0              ║")
    print(f"║  Objective: Synthetic Tactical Dataset Synthesis   ║")
    print(f"║  Target: {target_rows:<8} rows across {matches:<6} matches   ║")
    print(f"╚═══════════════════════════════════════════════════╝")

    # Benchmarking metrics initialization.
    start_time = time.time()
    total_rows = 0
    total_blue_wins = 0
    total_red_wins  = 0
    total_draws     = 0

    # Define the high-dimensional feature schema for the dataset.
    # We capture both Friendly (Blue) and Adversary (Red) states to give the Oracle full context.
    header = [
        "step", "match_id",
        # Friendly State Block
        "blue_id", "blue_x", "blue_y", "blue_hp", "blue_ammo", "blue_fuel",
        "blue_terrain", "blue_morale", "blue_elevation", "blue_kills",
        # Adversary State Block (Targets for Prediction)
        "red_id", "red_x", "red_y", "red_hp", "red_ammo", "red_fuel",
        "red_terrain", "red_morale", "red_elevation", "red_kills",
        # Tactical Metadata (Labels for the Intent Classifier and Action Predictor)
        "red_posture", "red_prev_action", "red_next_action",
        # Spatial relationship and environmental visibility.
        "distance", "fog_visible", "nearby_supplies", "outcome",
    ]

    # Open the CSV file and write the header row to establish the data structure.
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # ── Main Match Loop ──
        for match in range(matches):
            # Reset the environment to a fresh state for each new engagement.
            obs, info = env.reset()
            terminated = False
            truncated  = False
            step_num   = 0
            outcome    = "ongoing"

            # Track the previous action of each red unit to provide temporal context in the CSV.
            red_prev_actions = {u["id"]: 0 for u in env.red_units}

            # ── Step Loop: Continue until one team is eliminated or time runs out ──
            while not terminated and not truncated:
                step_num += 1

                # Filter for units currently operational on the battlefield.
                alive_blues = [u for u in env.blue_units if u["alive"]]
                alive_reds  = [u for u in env.red_units  if u["alive"]]

                # If either team is wiped, the current match is functionally over.
                if not alive_blues or not alive_reds:
                    break

                # ── Step A: Red Decision Logic & Data Capture ──
                # This is the most critical part: we ask the 'Heuristic Teacher' what it wants to do,
                # then we RECORD that intent as the target label for our future neural network.
                red_actions = {}
                for red in alive_reds:
                    # Ingest logic from the red_strategy module.
                    action, posture = get_red_action(
                        red, alive_blues, env.grid_size,
                        env.terrain_map, env.red_units
                    )
                    red_actions[red["id"]] = (action, posture)

                    # Calculate spatial features: Manhattan distance to the closest threat.
                    nearest_blue = min(
                        alive_blues,
                        key=lambda b: abs(red["pos"][0] - b["pos"][0]) +
                                      abs(red["pos"][1] - b["pos"][1])
                    )
                    dist = (abs(red["pos"][0] - nearest_blue["pos"][0]) +
                            abs(red["pos"][1] - nearest_blue["pos"][1]))

                    # Extract discrete grid features for this specific coordinate.
                    r_terrain = TERRAIN_NAMES.get(
                        env.terrain_map[red["pos"][0], red["pos"][1]], "Plains")
                    b_terrain = TERRAIN_NAMES.get(
                        env.terrain_map[nearest_blue["pos"][0], nearest_blue["pos"][1]], "Plains")

                    # Extract verticality.
                    r_elev = int(env.elevation[red["pos"][0], red["pos"][1]])
                    b_elev = int(env.elevation[nearest_blue["pos"][0], nearest_blue["pos"][1]])

                    # Record visibility: Can the Blue team see this Red unit according to the Fog of War?
                    fog_vis = int(env.fog_map[red["pos"][0], red["pos"][1]])

                    # Identify local resource density around the Red agent.
                    nearby_sup = sum(
                        1 for s in env.supply_drops
                        if abs(s["pos"][0] - red["pos"][0]) +
                           abs(s["pos"][1] - red["pos"][1]) <= 3
                    )

                    # COMMIT: Log the entire state-action pair to the dataset.
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
                        # Here, 'action' is the NEXT action (the target label for prediction).
                        posture, red_prev_actions.get(red["id"], 0), action,
                        dist, fog_vis, nearby_sup, outcome,
                    ])
                    total_rows += 1

                # ── Step B: Execute Blue Team Actions (Random Explorers) ──
                # Blue acts as the 'environment probe', moving and firing randomly to create diverse scenarios.
                for i, blue in enumerate(env.blue_units):
                    if blue["alive"]:
                        # Randomly sample from the action space [0-5].
                        blue_action = random.randint(0, 5)  
                        obs, reward, terminated, truncated, info = env.step(
                            blue_action, unit_index=i)
                        # Check if a Blue move ended the match (e.g., killed the last Red).
                        if terminated or truncated:
                            break

                if terminated or truncated:
                    break

                # ── Step C: Execute Red Team Actions (Strategic Teacher) ──
                # Now that Red's planned actions are already logged, we actually perform them in the environment.
                for i, red in enumerate(env.red_units):
                    if red["alive"] and red["id"] in red_actions:
                        action, posture = red_actions[red["id"]]
                        env.move_red(i, action)
                        # Store as 'previous' for the next step's record.
                        red_prev_actions[red["id"]] = action

                # Synchronization Check: Update termination status after Red's turn.
                if not env._alive_blue() or not env._alive_red():
                    terminated = True

            # ── Match Conclusion Analytics ──
            blue_alive = len(env._alive_blue())
            red_alive  = len(env._alive_red())
            if blue_alive > 0 and red_alive == 0:
                total_blue_wins += 1
            elif red_alive > 0 and blue_alive == 0:
                total_red_wins += 1
            else:
                total_draws += 1

            # Real-time Telemetry: Print performance metrics every 200 matches.
            if (match + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(f"  Match {match+1:>5}/{matches}  |  Rows: {total_rows:>7}  |  "
                      f"Rate: {rate:.0f} rows/s  |  B:{total_blue_wins} R:{total_red_wins} D:{total_draws}")

            # Early Exit: Stop simulation once the target volume of training data is secured.
            if total_rows >= target_rows:
                print(f"\n  [✔] Optimization: Target dataset volume reached at match {match+1}.")
                break

    # Final Summary Report.
    elapsed = time.time() - start_time
    print(f"\n{'═'*55}")
    print(f"  Synthetic Dataset Generation Success!")
    print(f"  Total Data Samples: {total_rows:,}")
    print(f"  Matches Simulated:  {match+1}")
    print(f"  Combat Outcomes:    Blue Wins: {total_blue_wins} | Red Wins: {total_red_wins} | Draws: {total_draws}")
    print(f"  Pipeline Latency:   {elapsed:.1f}s total")
    print(f"  Artifact Created:   {output_file}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    # Start the simulation with a target of 100,000 behavior records.
    run_simulation(matches=5000, target_rows=100000)
