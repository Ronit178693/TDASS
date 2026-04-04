import random
import numpy as np

def get_red_action(red_pos, blue_pos, grid_size, obstacles):
    """
    Red Bot logic:
    If Blue is visible (Manhattan distance <= 3), move toward Blue.
    Else, move random.
    Avoid obstacles.
    """
    if red_pos is None or blue_pos is None:
        return 0 # Stay
        
    rx, ry = red_pos
    bx, by = blue_pos
    
    visible_range = 3
    distance = abs(rx - bx) + abs(ry - by)
    
    possible_actions = []
    
    # Check each action for feasibility (collision with obstacles or OOB)
    # 0: Stay, 1: Up (-x), 2: Down (+x), 3: Left (-y), 4: Right (+y)
    for action in range(1, 5):
        nx, ny = rx, ry
        if action == 1: nx -= 1
        elif action == 2: nx += 1
        elif action == 3: ny -= 1
        elif action == 4: ny += 1
        
        # Check OOB
        if not (0 <= nx < grid_size and 0 <= ny < grid_size):
            continue
        # Check Obstacles
        if [nx, ny] in obstacles:
            continue
        possible_actions.append(action)
        
    if distance <= visible_range:
        # Move toward Blue
        # Prioritize actions that reduce distance
        best_action = 0
        min_dist = distance
        
        # Try to find the action among possible ones that minimizes distance
        for action in possible_actions:
            nx, ny = rx, ry
            if action == 1: nx -= 1
            elif action == 2: nx += 1
            elif action == 3: ny -= 1
            elif action == 4: ny += 1
            
            d = abs(nx - bx) + abs(ny - by)
            if d < min_dist:
                min_dist = d
                best_action = action
        
        if best_action != 0:
            return best_action
            
    # Random movement if not visible or no distance-reducing action found
    if possible_actions:
        return random.choice(possible_actions)
    
    return 0 # Stay if no moves possible
