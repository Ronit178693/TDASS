import csv
from battle_env import BattleEnv
from Element_Logic.red_strategy import get_red_action
import random

def run_simulation(matches=5000):
    env = BattleEnv(render_mode="ansi")
    dataset_file = "battle_data.csv"
    
    with open(dataset_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["Blue_X", "Blue_Y", "Red_X", "Red_Y", "Red_Previous_Action", "Red_Next_Move"])
        
        count = 0
        for match in range(matches):
            obs, info = env.reset()
            terminated = False
            
            # Initialize Red's previous action (None or 0 for initialization)
            red_prev_action = 0
            
            while not terminated and count < 100000: # Safety break at 100k steps
                # 1. State current positions
                b_x, b_y = env.blue_pos
                r_x, r_y = env.red_pos
                
                # 2. Red decides its action (NEXT MOVE)
                red_next_action = get_red_action([r_x, r_y], [b_x, b_y], env.grid_size, env.obstacles)
                
                # 3. Log current state and Red's next move
                writer.writerow([b_x, b_y, r_x, r_y, red_prev_action, red_next_action])
                count += 1
                
                # 4. Update state for next step:
                # Blue (Friendly) moves randomly
                blue_action = random.randint(0, 4)
                obs, reward, terminated, truncated, info = env.step(blue_action)
                
                # If Blue captured Red, episode ends!
                if terminated:
                    break
                    
                # Red moves manually because current BattleEnv.step only moves Blue
                # Let's manually Update Red's position in the environment
                rx, ry = env.red_pos
                nx, ny = rx, ry
                if red_next_action == 1: nx -= 1
                elif red_next_action == 2: nx += 1
                elif red_next_action == 3: ny -= 1
                elif red_next_action == 4: ny += 1
                
                # Update Red position (Check OOB and Obstacles)
                if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size and 
                    [nx, ny] not in env.obstacles):
                    # Check if Red moves into Blue (Red capturing Blue? Not defined, but we update)
                    env.red_pos = [nx, ny]
                    env._update_grid()
                
                # Store Red's action as previous for the next row
                red_prev_action = red_next_action
                
                # In case 5,000 matches isn't enough, we could stop when count hits 50,000
                if count >= 50000:
                    print(f"Reached 50,000 steps. Closing simulation.")
                    return

            if (match + 1) % 100 == 0:
                print(f"Finished {match + 1} matches...")

    print(f"Simulation complete. Data saved to {dataset_file}.")

if __name__ == "__main__":
    run_simulation(matches=5000)
