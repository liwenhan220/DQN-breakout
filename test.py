import gym
import time

env = gym.make('BreakoutDeterministic-v4')
env.reset()

for i in range(10):
    env.step(1)
    env.render()
    time.sleep(0.1)
input('press key to terminate')