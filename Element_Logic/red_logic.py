from random import randint

def Red_logic(self):
    # We want red to move towards blue once in the vision range until it moves randomly
    # Red must avoid collision with the wall and bounds

    enemy_x, enemy_y = self.goal_pos
    player_x, player_y = self.player_pos

    visible_range = 3

    # Once the visible range is set we will calculate the distance b/w the 2 elements 
    # using absolute difference for Manhattan distance
    distance = abs(enemy_x - player_x) + abs(enemy_y - player_y) 
    
    if distance <= visible_range:
        print("Enemy is visible")
        # Move towards player (simple chase logic)
        # Note: This allows diagonal movement if both x and y change
        if player_x > enemy_x:
            enemy_x += 1
        elif player_x < enemy_x:
            enemy_x -= 1
        
        if player_y > enemy_y:
            enemy_y += 1
        elif player_y < enemy_y:
            enemy_y -= 1
    # Enemy moves randomly
    else:
        # Range is not visible, move randomly
        move = randint(0, 3)
        # 0 = up, 1 = down, 2 = left, 3 = right
        
        if move == 0: # Up (decrease row)
            enemy_x = max(0, enemy_x - 1)
        elif move == 1: # Down (increase row)
            enemy_x = min(4, enemy_x + 1)
        elif move == 2: # Left (decrease col)
            enemy_y = max(0, enemy_y - 1)
        elif move == 3: # Right (increase col)
            enemy_y = min(4, enemy_y + 1)

    # Update the position in the game state
    self.goal_pos = [enemy_x, enemy_y]


            





    

