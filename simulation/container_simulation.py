import random
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class ContainerSimulation:
    
    def __init__(self, size: int = 10):
        self.size = size
        self.container = [[0 for _ in range(size)] for _ in range(size)]
        self.action_history = []
        self.homogeneity_history = []
        
    def reset(self):
        self.container = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.action_history = []
        self.homogeneity_history = []
        logger.info("Container reset")
    
    def add_ball(self, ball_type: int, position: Tuple[int, int] = None):
        
        if position is None:
            empty_positions = []
            for i in range(self.size):
                for j in range(self.size):
                    if self.container[i][j] == 0:
                        empty_positions.append((i, j))
            
            if empty_positions:
                pos = random.choice(empty_positions)
                self.container[pos[0]][pos[1]] = ball_type
            else:
                logger.warning("Container is full, cannot add more balls")
                return False
        else:
            if self.container[position[0]][position[1]] == 0:
                self.container[position[0]][position[1]] = ball_type
            else:
                return False
        
        return True
    
    def add_multiple_balls(self, ball_type: int, count: int):
        for _ in range(count):
            if not self.add_ball(ball_type):
                break
    
    def shake(self, duration: int = 10):

        num_swaps = duration * (self.size ** 2) // 20
        
        for _ in range(num_swaps):
            # Picking here 2 random adjacent positions
            i1, j1 = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            
            # Choose direction of movement of ball (up, down, left, right)
            direction = random.choice(['up', 'down', 'left', 'right'])
            
            i2, j2 = i1, j1
            if direction == 'up' and i1 > 0:
                i2 = i1 - 1
            elif direction == 'down' and i1 < self.size - 1:
                i2 = i1 + 1
            elif direction == 'left' and j1 > 0:
                j2 = j1 - 1
            elif direction == 'right' and j1 < self.size - 1:
                j2 = j1 + 1
            else:
                continue
            
            ball1 = self.container[i1][j1]
            ball2 = self.container[i2][j2]
            
            # Probability of swap based on weight difference
            if ball1 > 0 and ball2 > 0:
                # Heavier balls have higher chance to move down
                if ball1 > ball2 and direction == 'down':
                    prob = 0.7
                elif ball2 > ball1 and direction == 'up':
                    prob = 0.7
                else:
                    prob = 0.3
                
                if random.random() < prob:
                    self.container[i1][j1], self.container[i2][j2] = ball2, ball1
            elif ball1 > 0 and ball2 == 0:
                if random.random() < 0.8:
                    self.container[i1][j1], self.container[i2][j2] = ball2, ball1
            elif ball2 > 0 and ball1 == 0:
                if random.random() < 0.8:
                    self.container[i1][j1], self.container[i2][j2] = ball2, ball1
    
    def calculate_homogeneity(self) -> float:

        if self._get_ball_count() == 0:
            return 0.0
        
        total_diversity = 0
        neighbor_count = 0
        
        for i in range(self.size):
            for j in range(self.size):
                if self.container[i][j] > 0:
                    neighbors = []
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            neighbors.append(self.container[ni][nj])
                    
                    unique_neighbors = set(n for n in neighbors if n > 0)
                    if unique_neighbors:
                        diversity = len(unique_neighbors) / 3 
                        total_diversity += diversity
                        neighbor_count += 1
        
        if neighbor_count == 0:
            return 0.0
        
        return total_diversity / neighbor_count
    
    def _get_ball_count(self) -> int:
        count = 0
        for row in self.container:
            for cell in row:
                if cell > 0:
                    count += 1
        return count
    
    def get_state(self) -> List[List[int]]:
        return self.container
    
    def get_ball_distribution(self) -> Dict[str, int]:
        distribution = {"LIGHT": 0, "NORMAL": 0, "HEAVY": 0}
        for row in self.container:
            for cell in row:
                if cell == 1:
                    distribution["LIGHT"] += 1
                elif cell == 2:
                    distribution["NORMAL"] += 1
                elif cell == 3:
                    distribution["HEAVY"] += 1
        return distribution
    
    def execute_action(self, action: str, parameters: Dict) -> Tuple[bool, str]:
        try:
            if action == "ADD_LIGHT":
                count = parameters.get("count", 1)
                self.add_multiple_balls(1, count)
                return True, f"Added {count} light ball(s)"
                
            elif action == "ADD_NORMAL":
                count = parameters.get("count", 1)
                self.add_multiple_balls(2, count)
                return True, f"Added {count} normal ball(s)"
                
            elif action == "ADD_HEAVY":
                count = parameters.get("count", 1)
                self.add_multiple_balls(3, count)
                return True, f"Added {count} heavy ball(s)"
                
            elif action == "SHAKE":
                duration = parameters.get("duration", 10)
                self.shake(duration)
                return True, f"Shook container for {duration} seconds"
                
            elif action == "RESET":
                self.reset()
                return True, "Container reset"
                
            else:
                return False, f"Unknown action: {action}"
                
        except Exception as e:
            return False, f"Error executing action: {str(e)}"
    
    def display(self):
        from config.prompts import BALL_SYMBOLS
        
        print("\n" + "=" * 50)
        print("CONTAINER STATE")
        print("=" * 50)
        
        for i, row in enumerate(self.container):
            row_str = ""
            for cell in row:
                row_str += f" {BALL_SYMBOLS.get(cell, '?')}"
            print(f"{i:2} |{row_str}")
        
        print("-" * 50)
        print(f"Homogeneity Score: {self.calculate_homogeneity():.3f}")
        print(f"Distribution: {self.get_ball_distribution()}")
        print("=" * 50)