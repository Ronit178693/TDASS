import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# ==========================================
# PART 1: THE GAME (The Environment)
# ==========================================
class SimpleGame(gym.Env):
    def __init__(self):
        # We define a 5x5 grid (smaller is easier to read)
        self.grid_size = 5
        
        # ACTIONS: We have 4 buttons: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # OBSERVATION: The screen is a 5x5 grid of numbers
        self.observation_space = spaces.Box(low=0, high=2, shape=(5, 5), dtype=int)
        
        # The Player starts at top-left [0,0]
        self.player_pos = [0, 0]
        # The Goal is at bottom-right [4,4]
        self.goal_pos = [4, 4]

    def reset(self, seed=None, options=None):
        # Reset player to start
        self.player_pos = [0, 0]
        
        # create the empty grid (0 = empty)
        grid = np.zeros((5, 5), dtype=int)
        
        # Place player (1) and goal (2)
        grid[self.player_pos[0], self.player_pos[1]] = 1
        grid[self.goal_pos[0], self.goal_pos[1]] = 2
        
        return grid, {}

    def step(self, action):
        # 1. Update Player Position based on button press
        # Remember: [Row, Column]
        if action == 0: # Up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 1: # Down
            self.player_pos[0] = min(4, self.player_pos[0] + 1)
        elif action == 2: # Left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif action == 3: # Right
            self.player_pos[1] = min(4, self.player_pos[1] + 1)

        # 2. Check if we won
        # Are we standing on the goal?
        terminated = (self.player_pos == self.goal_pos)
        
        # 3. Create the grid to show the user
        grid = np.zeros((5, 5), dtype=int)
        grid[self.player_pos[0], self.player_pos[1]] = 1 # Player
        grid[self.goal_pos[0], self.goal_pos[1]] = 2     # Goal
        
        return grid, 0, terminated, False, {}

    def render(self):
        # This just prints the grid to the console so we can see it
        grid, _, _, _, _ = self.step(action=-1) # Just get current state
        print("\n" + "-"*10)
        for row in grid:
            # 0 is ., 1 is P (Player), 2 is G (Goal)
            row_str = " ".join(str(x) for x in row)
            print(row_str.replace("0", ".").replace("1", "P").replace("2", "G"))
        print("-" * 10)


# ==========================================
# PART 2: THE PLAYER (Running the Code)
# ==========================================

# 1. Turn on the game
env = SimpleGame()
env.reset()

print("GAME START! (P = Player, G = Goal)")
env.render()

# 2. Let's take 10 steps
for step in range(50):
    # Pick a RANDOM button (0, 1, 2, or 3)
    # In a real AI, the AI would choose this.
    action = env.action_space.sample() 
    
    # Send the button press to the game
    obs, reward, terminated, _, _ = env.step(action)
    
    # Print what happened
    move_name = ["Up", "Down", "Left", "Right"][action]
    print(f"Step {step+1}: Player moved {move_name}")
    env.render()
    
    if terminated:
        print("YOU WON! Reached the Goal!")
        break
    
    time.sleep(0) # Pause so we can read the output