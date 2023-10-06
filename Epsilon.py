import numpy as np
import matplotlib.pyplot as plt


class Epsilon:
    def __init__(self, initial_eps = 1.0, final_eps = 0.1, explore_frames = 1_000_000, total_frames = 30_000_000, action_space_n = 4):
        self.epsilon = initial_eps
        self.slope = (final_eps - initial_eps) / explore_frames
        self.action_space = action_space_n
        self.final_eps = final_eps
        self.second_slope = (-final_eps) / (total_frames - explore_frames)

    def step(self):
        if self.epsilon > self.final_eps:
            self.epsilon += self.slope
        elif self.epsilon > 0:
            self.epsilon += self.second_slope
        return self

    def gen_action(self, state, dqn, render = False):
        epsilon = self.epsilon
        if np.random.random() > epsilon:
            qs = dqn.predict(state/255.0)
            if render:
                print('Q values: {}'.format(qs))
            return np.argmax(qs)
        return np.random.randint(0, self.action_space)

    def show_Qs(self, qs):
        plt.clf()
        plt.title('Q-values')
        plt.ylim(-1, 9)
        plt.bar(np.arange(len(qs)), qs)
        plt.draw()
        plt.pause(1e-17)

    def self_act(self, state, dqn, epsilon, render = False):
        if np.random.random() > epsilon:
            qs = dqn.predict(state / 255.0)
            if render:
                self.show_Qs(qs)
                print('Q-values: {}'.format(qs))
            return np.argmax(qs)
        return np.random.randint(0, self.action_space)

    def get(self):
        return self.epsilon

    def show(self):
        print(self.epsilon)
        return self


def test():
    e = Epsilon()
    for i in range(2_000_000):
        e.step().show()

# test()

