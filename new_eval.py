from game import Game
import numpy as np
from Epsilon import Epsilon

def evaluate(net, frames, no_op_max, epsilon, debug=False, show_percentage = 0.1):
    print('Evaluation Started!!!!!!!!!!!!!!!!!!!!!!!!')
    show_every = int(show_percentage * frames)
    frame_count = 0
    eps_count = 0
    total_score = 0
    env = Game()
    actor = Epsilon()
    while frame_count < frames:
        state, _ = env.reset()
        done = False
        for _ in range(np.random.randint(0, no_op_max)):
            state, _, done, _, _ = env.step(1)
        while not done:
            a = actor.self_act(state, net, epsilon)
            state, _, done, _, _ = env.step(a)
            if debug:
                env.render()
            frame_count += 1
            if frame_count % show_every == 0:
                print('Evaluating Network: {}% complete!!!, Average: {}'.format(int(frame_count / frames * 100), int(total_score / eps_count * 100) / 100))
        total_score += env.reward_total
        eps_count += 1
    print('Evaluation Complete! Average score: %.2f' % (total_score / eps_count))
    return total_score / eps_count


def test():
    from dqn import DQN
    net = DQN()
    net.load('test_net')
    print(evaluate(net, 135000, 30, 0.05, False))

