import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
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
        self.observation_space = spaces.MultiDiscrete(5, 5, 5, 5)
        
        # The Player starts at top-left [0,0]
        self.player_pos = [0, 0]
        # The Goal is at bottom-right [4,4]
        self.goal_pos = [4, 4]
        # The Obstacle is at [2,2]
        self.obs_pos = [2,2]

        # Pygame Setup 
        # Define colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        # Screen dimensions
        self.window_size = 500
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):
        # Reset player to start
        self.player_pos = [0, 0]
        
        # create the empty grid (0 = empty)
        grid = np.zeros((5, 5), dtype=int)
        
        # Place player (1) and goal (2) and obstacle (3)
        grid[self.player_pos[0], self.player_pos[1]] = 1
        grid[self.goal_pos[0], self.goal_pos[1]] = 2
        grid[self.obs_pos[0], self.obs_pos[1]] = 3  
        
        return grid, {}

    def _get_obs(self):
        # Data Flattening
        # Not add obs as it is static and adding it increases noise the ai learns it pos automatically
        return np.array([*self.player_pos, *self.goal_pos], dtype=np.int64)

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
        grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Goal
        grid[self.obs_pos[0], self.obs_pos[1]] = 3  # Obstacle
        
        return grid, 0, terminated, False, {}

    def render(self):
        # This just prints the grid to the console so we can see it
        grid, _, _, _, _ = self.step(action=-1) # Just get current state
        print("\n" + "-"*10)
        for row in grid:
            # 0 is ., 1 is P (Player), 2 is G (Goal)
            row_str = " ".join(str(x) for x in row)
            print(row_str.replace("0", ".").replace("1", "B").replace("2", "R").replace("3", "O"))
        print("-" * 10)


# ==========================================
# PART 2: THE PLAYER (Running the Code)
# ==========================================

# 1. Turn on the game
env = SimpleGame()
env.reset()

print("GAME START! (B = Player, R = Enemy, O = Obstacle)")
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
        print("Captured Red")
        break
    
    time.sleep(0) # Pause so we can read the output
    
env.close() #Closing the environment
    
    
# Remember to trun the target to enemy and add obstacles for more complex scenarios! and increase the grid size!