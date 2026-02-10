import gymnasium as gym
from Element_Logic.red_logic import Red_logic
from gymnasium import spaces
import numpy as np
import pygame
import time

# ==========================================
# PART 1: THE GAME (The Environment)
# ==========================================
class SimpleGame(gym.Env):
    def __init__(self, render_mode="human"):
        # We define a 5x5 grid (smaller is easier to read)
        self.grid_size = 5
        self.render_mode = render_mode
        # ACTIONS: We have 4 buttons: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # OBSERVATION: The screen is a 5x5 grid of numbers
        self.observation_space = spaces.MultiDiscrete([5, 5, 5, 5])
        
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

#  As we are only playing a single game our reset function is nt helping us but when playing more then one game it helps

    # def reset(self):
    #     # Reset player to start
    #     self.player_pos = [0, 0]
        
    #     # create the empty grid (0 = empty)
    #     grid = np.zeros((5, 5), dtype=int)
        
    #     # Place player (1) and goal (2) and obstacle (3)
    #     grid[self.player_pos[0], self.player_pos[1]] = 1
    #     grid[self.goal_pos[0], self.goal_pos[1]] = 2
    #     grid[self.obs_pos[0], self.obs_pos[1]] = 3  
        
    #     return grid, {}

    def _get_obs(self):
        # Data Flattening
        # Not add obs as it is static and adding it increases noise the ai learns it pos automatically
        return np.array([*self.player_pos, *self.goal_pos], dtype=np.int64)

    def step(self, action):

        # The core logic:
        # 1. Apply the action
        # 2. Check for collisions
        # 3. Calculate rewards
        # 4. Check if the game is over

        grid = np.zeros((5, 5), dtype=int)
        grid[self.player_pos[0], self.player_pos[1]] = 1 # Player
        grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Goal
        grid[self.obs_pos[0], self.obs_pos[1]] = 3  # Obstacle

        # 1. Update Player Position based on button press   
        # Remember: [Row, Column]
        # Boundry check
        if action == 0: # Up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 1: # Down
            self.player_pos[0] = min(4, self.player_pos[0] + 1)
        elif action == 2: # Left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif action == 3: # Right
            self.player_pos[1] = min(4, self.player_pos[1] + 1)
            
        # Update Red entity position
        Red_logic(self)

        # Default values
        # With every step the agent takes it loses 1 point
        reward = -0.1
        terminated = False

        # Wall collision check
        if self.player_pos == self.obs_pos:
            # Penalty for hitting the wall
            reward = -0.5
            # Note: Player currently overlaps obstacle.

        # 2. Check if we won
        if self.player_pos == self.goal_pos:
            reward = 2
            terminated = True
        else:
            print("Cant reach target")
        
        # 3. Get Observation (Vector)
        observation = self._get_obs()

        # 4. Render if human
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill(self.WHITE)
            
            # Draw Grid
            for x in range(0, self.window_size, self.cell_size):
                pygame.draw.line(canvas, self.BLACK, (x, 0), (x, self.window_size))
            for y in range(0, self.window_size, self.cell_size):
                pygame.draw.line(canvas, self.BLACK, (0, y), (self.window_size, y))
            
            # Helper to draw rect
            def draw_cell(pos, color):
                pygame.draw.rect(canvas, color, pygame.Rect(
                    pos[1] * self.cell_size, 
                    pos[0] * self.cell_size, 
                    self.cell_size, self.cell_size
                ))
            
            # Draw Elements
            draw_cell(self.goal_pos, self.RED)      # Goal
            draw_cell(self.obs_pos, self.BLACK)     # Obstacle (using Black for obstacle as configured)
            draw_cell(self.player_pos, self.BLUE)   # Player
            
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(10) # 10 FPS
        else:
            # Console Render
            # Manually create grid for visualization
            grid = np.zeros((5, 5), dtype=int)
            grid[self.player_pos[0], self.player_pos[1]] = 1
            grid[self.goal_pos[0], self.goal_pos[1]] = 2
            grid[self.obs_pos[0], self.obs_pos[1]] = 3
            
            print("\n" + "-"*10)
            for row in grid:
                row_str = " ".join(str(x) for x in row)
                print(row_str.replace("0", ".").replace("1", "B").replace("2", "R").replace("3", "O"))
            print("-" * 10)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


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
    observation, reward, terminated, _, _ = env.step(action)
    
    # Print what happened
    move_name = ["Up", "Down", "Left", "Right"][action]
    print(f"Step {step+1}: Player moved {move_name}")
    env.render()
    
    if terminated:
        print("Captured Red")
        break
    
    time.sleep(0) # Pause so we can read the output
    
env.close() #Closing the environment
