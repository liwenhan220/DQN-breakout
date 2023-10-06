from dqn import DQN
from Epsilon import Epsilon
from game import Game
import time
import numpy as np

NET_DIR = 'networks'
GAME_NAME = 'BreakoutDeterministic-v4'
MODEL_NAME = 'generation-18-eval-reward-67.73.model'
DEFAULT = False
if DEFAULT:
    MODEL_DIR = 'Breakout_v2_ai'
else:
    MODEL_DIR = NET_DIR + '/' + MODEL_NAME

TRAIN = False
RENDER = False
AGENT_HIST = 4
GAMMA = 0.99
EXPLORE_FRAMES = 1_000_000
REPLAY_START_SIZE = 50_000
DEF_EPS = 0.0
LEARNING_RATE = 0.00025
SHOW_EVERY = 1
UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 10_000
NO_OP_MAX = 10
TOTAL_FRAMES = 30_000_000
BATCH_SIZE = 32
INIT_EPS = 1.0
FINAL_EPS = 0.1
REPLAY_SIZE = 1_000_000

def run():
    frame_count = 0
    env = Game(key=GAME_NAME)
    net = DQN(learning_rate=LEARNING_RATE, action_space=env.actions_n(),gamma=GAMMA, batch_size=BATCH_SIZE)
    if MODEL_NAME is not None:
        net.load(MODEL_DIR)
    epsilon = Epsilon()
    while frame_count < TOTAL_FRAMES:
        state, frame = env.reset()
        terminal = False
        for _ in range(np.random.randint(1, NO_OP_MAX)):
            env.step(1)
        while not terminal:
            action = epsilon.self_act(state, net, DEF_EPS, render = RENDER)
            new_state, reward, terminal, new_frame, life_lost = env.step(action)
            env.render()
            if frame_count % UPDATE_FREQ == 0:
                time.sleep(0.007)
            # env.show()
            state = new_state
            frame_count += 1
            epsilon.step()
            if life_lost:
                for _ in range(np.random.randint(1, NO_OP_MAX)):
                    env.step(1)
                
        # print('epsilon: {}, rewards: {}'.format(epsilon.get(), env.reward_total))

import tensorflow as tf
with tf.device('/cpu:0'):
    run()
