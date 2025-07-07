import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

class SocAlertEnv(gym.Env):
    def __init__(self, params=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.max_steps = 5
        self.steps = 0

        self.params = {
            "compromise_multiplier": 10,
            "ignore_bias_penalty": 0.5,
            "alert_reward": 3,
            "alert_penalty": -1,
            "isolate_reward_high": 8,
            "isolate_reward_mid": 4,
            "isolate_penalty_low": -0.5,
            "isolate_penalty_uncertain": -1.5
        }

        if params is None and os.path.exists("current_params.json"):
            with open("current_params.json", "r") as f:
                self.params.update(json.load(f))
        elif isinstance(params, dict):
            self.params.update(params)

        self.state = self._generate_state()

    def _generate_state(self):
        if np.random.rand() < 0.3:
            # Force risky case (encourages isolation)
            threat_score = np.random.uniform(0.8, 1.0)
            confidence = np.random.uniform(0.4, 0.7)  # Uncertain range
            criticality = np.random.uniform(0.8, 1.0)
        else:
            # Normal case
            threat_score = np.random.uniform(0.0, 1.0)
            confidence = np.random.uniform(0.0, 1.0)
            criticality = np.random.uniform(0.2, 1.0)

        return np.array([
            threat_score,
            confidence,
            criticality,
            np.random.uniform(0.0, 1.0),       # recurrence
            np.random.uniform(0.0, 1.0),       # isolation cost
        ], dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = self._generate_state()
        return self.state, {}

    def step(self, action):
        s = self.state
        threat_score, threat_confidence, asset_criticality = s[0], s[1], s[2]
        alert_recurrence, isolation_cost = s[3], s[4]
        risk = threat_score * asset_criticality
        p = self.params

        if action == 0:  # Ignore
            if risk > 0.6:
                compromise_cost = p["compromise_multiplier"] * asset_criticality * (1 + alert_recurrence)
                reward = -threat_confidence * compromise_cost
            else:
                reward = 0.2
            reward -= p["ignore_bias_penalty"]

        elif action == 1:  # Alert
            reward = p["alert_reward"] * threat_confidence if risk > 0.6 else p["alert_penalty"]

        elif action == 2:  # Isolate
            if risk > 0.9:
                reward = p["isolate_reward_high"] * threat_confidence - isolation_cost
            elif risk > 0.6:
                reward = p["isolate_reward_mid"] * threat_confidence - isolation_cost
            else:
                penalty = p["isolate_penalty_uncertain"]
                reward = penalty - isolation_cost


        self.state = self._generate_state()
        self.steps += 1
        return self.state, reward, self.steps >= self.max_steps, False, {}

    def render(self):
        print(f"Current state: {self.state}")

class SocFlatEnv(gym.Env):
    def __init__(self, params=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.max_steps = 5
        self.steps = 0

        self.state = self._generate_state()

    def _generate_state(self):
        if np.random.rand() < 0.3:
            # Risky case
            threat_score = np.random.uniform(0.8, 1.0)
            confidence = np.random.uniform(0.4, 0.7)
            criticality = np.random.uniform(0.8, 1.0)
        else:
            # Normal case
            threat_score = np.random.uniform(0.0, 1.0)
            confidence = np.random.uniform(0.0, 1.0)
            criticality = np.random.uniform(0.2, 1.0)

        return np.array([
            threat_score,
            confidence,
            criticality,
            np.random.uniform(0.0, 1.0),  # recurrence
            np.random.uniform(0.0, 1.0),  # isolation cost
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = self._generate_state()
        return self.state, {}

    def step(self, action):
        s = self.state
        theta, gamma, kappa = s[0], s[1], s[2]
        risk = theta * kappa
        is_malicious = risk > 0.6

        if action == 0:  # Ignore
            reward = -1 if is_malicious else 0.5
        elif action == 1:  # Alert
            reward = 1 if is_malicious else -0.5
        elif action == 2:  # Isolate
            reward = 1 if is_malicious else -1

        self.state = self._generate_state()
        self.steps += 1
        return self.state, reward, self.steps >= self.max_steps, False, {}

    def render(self):
        print(f"Current state: {self.state}")
