import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys

class BattleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode="ansi"):
        super(BattleEnv, self).__init__()
        self.grid_size = 10
        self.cell_size = 50
        self.window_size = self.grid_size * self.cell_size
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Grid setup: 0=Empty, 1=Blue, 2=Red, 3=Obstacle
        self.blue_pos = [0, 0]
        self.red_pos = [9, 9]
        self.obstacles = [[5, 5], [5, 6], [6, 5], [6, 6]]
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self.action_space = spaces.Discrete(5) # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=int)
        
        self._update_grid()

    def _update_grid(self):
        self.grid.fill(0)
        # Set Obstacles
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = 3
        # Set Units
        if self.red_pos is not None:
             self.grid[self.red_pos[0], self.red_pos[1]] = 2
        self.grid[self.blue_pos[0], self.blue_pos[1]] = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blue_pos = [0, 0]
        self.red_pos = [9, 9]
        self._update_grid()
        return self.grid, {}

    def step(self, action):
        # Action Mapping: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        if action == 0: # Stay
             return self.grid, 0, False, False, {}

        new_pos = list(self.blue_pos)
        if action == 1:   # Up
            new_pos[0] -= 1
        elif action == 2: # Down
            new_pos[0] += 1
        elif action == 3: # Left
            new_pos[1] -= 1
        elif action == 4: # Right
            new_pos[1] += 1
            
        # 1. Out of Bounds Check
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
             return self.grid, 0, False, False, {"msg": "OOB"}

        # 2. Collision Detection: Prevent walking into walls (3)
        if self.grid[new_pos[0], new_pos[1]] == 3:
             return self.grid, 0, False, False, {"msg": "Collision with Wall"}

        # 3. Battle Logic: If Blue moves into Red's square -> Capture
        if self.red_pos is not None and new_pos == self.red_pos:
             print("BATTLE: Blue captured Red!")
             self.red_pos = None # Capture
             self.blue_pos = new_pos
             self._update_grid()
             return self.grid, 10, True, False, {"msg": "Enemy Captured"}

        # Standard Move
        self.blue_pos = new_pos
        self._update_grid()
        return self.grid, 0, False, False, {}

    def render(self):
        if self.render_mode == "ansi":
            print("\n--- Current Grid ---")
            print(self.grid)
            return

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.set_caption("Battlefield Environment")
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()

            # Predefined colors
            BG_COLOR = (30, 30, 30)
            BLUE_COLOR = (0, 120, 255)
            RED_COLOR = (255, 50, 50)
            WALL_COLOR = (100, 100, 100)
            GRID_COLOR = (60, 60, 60)

            self.screen.fill(BG_COLOR)

            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                    val = self.grid[r, c]
                    
                    # Draw grid cell
                    pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

                    if val == 1: # Blue
                        pygame.draw.circle(self.screen, BLUE_COLOR, rect.center, self.cell_size // 2.5)
                    elif val == 2: # Red
                        pygame.draw.circle(self.screen, RED_COLOR, rect.center, self.cell_size // 2.5)
                    elif val == 3: # Wall
                        pygame.draw.rect(self.screen, WALL_COLOR, rect.inflate(-4, -4))

            pygame.display.flip()
            self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    # Create environment with Pygame rendering
    env = BattleEnv(render_mode="human")
    obs, info = env.reset()
    env.render()
    
    print("Week 2 Deliverable: Control Blue unit (0-4). Close window or Ctrl+C to exit.")
    
    running = True
    while running:
        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        try:
            # Note: input() blocks the pygame window from updating if not handled well,
            # but for this deliverable it's acceptable as long as we render after move.
            action_input = input("Move (0=Stay, 1=U, 2=D, 3=L, 4=R): ")
            if not action_input: continue
            action = int(action_input)
            
            if action in range(5):
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if terminated:
                    print("Game over! Enemy captured. Resetting...")
                    env.reset()
                    env.render()
        except ValueError:
             print("Please enter 0-4.")
        except KeyboardInterrupt:
            break
            
    env.close()
