import gym
import numpy as np
from fp import preprocess
import pdb

class Game:
    def __init__(self, key = 'BreakoutDeterministic-v4', hist = 4):
        self.env = gym.make(key)
        self.frames = []
        self.origs = []
        self.hist = hist
        self.height, self.width = preprocess(self.env.reset()).shape
        self.reward_total = 0
        _, _, _, info = self.env.step(0)
        self.initial_lives = info['ale.lives']
        self.last_lives = self.initial_lives

    def actions_n(self):
        return self.env.action_space.n

    def resetFrame(self, frame):
        self.frames = [frame for _ in range(self.hist)]

    def reset(self):
        frame = self.env.reset()
        self.origs = [frame]

        frame = preprocess(frame)
        self.last_lives = self.initial_lives
        self.resetFrame(frame)
        self.reward_total = 0
        return self.get_state(), frame

    def get_state(self):
        return np.array(self.frames[len(self.frames) - self.hist: len(self.frames)], dtype=np.uint8).reshape((1, self.height, self.width, self.hist))

    def step(self, action):
        state, reward, is_terminal, info = self.env.step(action)
        self.reward_total += reward
        reward = np.sign(reward)

        self.origs.append(state)

        frame = preprocess(state)
        self.frames.append(frame)
        if info['ale.lives'] < self.last_lives:
            self.last_lives = info['ale.lives']
            self.resetFrame(frame)
            return self.get_state(), -1, is_terminal, frame, True
        return self.get_state(), reward, is_terminal, frame, is_terminal

    def getFrames(self):
        return self.origs

    def randA(self):
        return np.random.randint(0, self.env.action_space.n)

    def render(self):
        return self.env.render()

    def show(self): #for test only
        import cv2
        x = np.zeros((self.height, self.width))
        state = self.get_state().reshape((4, self.height, self.width))
        for i in range(len(state)):
            x += state[i]
        x /= 4
        cv2.imshow('state', cv2.resize(x, (500, 500)))
        cv2.waitKey(1)

    def show_debug(self, state): #for test only
        import cv2
        x = np.zeros((self.height, self.width))
        state = state.reshape((4, self.height, self.width))
        for i in range(len(state)):
            x += state[i]
        x /= 4
        cv2.imshow('state', cv2.resize(x, (500, 500)))
        cv2.waitKey(1)

def test():
    g = Game()
    g.reset()
    g.step(1)
    g.step(3)

    g.step(3)

    g.step(3)
    g.step(3)
    g.show()
    input()


# test()
