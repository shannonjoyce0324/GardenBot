import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GardenBotEnv(gym.Env):
    def __init__(self):
        super(GardenBotEnv, self).__init__()

        # Action space: 0 = Do nothing, 1 = Water, 2 = Fertilize, 3 = Pesticide
        self.action_space = spaces.Discrete(4)

        # Observation space: soil_moisture, nutrient_level, pest_threat, plant_health
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([100, 100, 1, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([50.0, 50.0, 0.0, 80.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        soil_moisture, nutrient_level, pest_threat, plant_health = self.state

        # Natural decay and random events
        soil_moisture -= 5
        nutrient_level -= 2
        if random.random() < 0.1:
            pest_threat = 1

        # Apply actions
        if action == 1:  # Water
            soil_moisture += 20
        elif action == 2:  # Fertilize
            nutrient_level += 15
        elif action == 3:  # Pesticide
            pest_threat = 0

        # Clamp values to [0, 100]
        soil_moisture = np.clip(soil_moisture, 0, 100)
        nutrient_level = np.clip(nutrient_level, 0, 100)

        # Update plant health
        if soil_moisture < 30 or nutrient_level < 30 or pest_threat == 1:
            plant_health -= 10
        else:
            plant_health += 5

        plant_health = np.clip(plant_health, 0, 100)

        # Calculate reward
        reward = 0
        if plant_health >= 80:
            reward += 2
        if 50 <= plant_health <= 90:
            if soil_moisture > 90:
                reward -= 1  # Overwatering
            if nutrient_level > 90:
                reward -= 1  # Over fertilizing
        if plant_health < 50:
            reward -= 2  # Neglect

        # Done if plant dies
        terminated = bool(plant_health <= 0)

        self.state = np.array([soil_moisture, nutrient_level, pest_threat, plant_health], dtype=np.float32)
        return self.state, reward, terminated, False, {}

    def render(self):
        print(f"State: {self.state}")
