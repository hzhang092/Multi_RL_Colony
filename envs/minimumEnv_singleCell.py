# minimum colony simulation
# hard coded rewards 
import numpy as np
import random

class minimumColonyEnv:
    def __init__(self) -> None:
        self.nutrient_levels = [0, 1, 2]
        self.pressure_levels = [0, 1, 2]
        self.state = None
        
    def reset(self):
        self.state = (random.choice(self.nutrient_levels),random.choice(self.pressure_levels))
        return self.state
    
    def step(self, action):
        nutrient, presure = self.state
        reward = 0
        
        if action == 0: # grow
            reward = 2 if nutrient > 0 else -1
            if presure == 2: reward = -1
            
        elif action == 1: # divide
            if nutrient == 2 and presure == 0:
                reward = 3
            else:
                reward = -2
                
        elif action == 2: # dormant
            reward = 0
            
        # update environment state randomly
        next_state = (random.choice(self.nutrient_levels),
                      random.choice(self.pressure_levels))

        done = False  # no terminal state for now
        self.state = next_state
        return self.state, reward, done