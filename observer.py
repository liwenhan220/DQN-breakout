import numpy as np

class Observer:
    def __init__(self):
        self.rewards = []

    def store(self, name = 'reward_list'):
        np.save(name, self.rewards)

    def add(self, value):
        self.rewards.append(value)
