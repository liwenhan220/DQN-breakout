from dqn import DQN
from Epsilon import Epsilon
from game import Game
from RM import ReplayMemory
from observer import Observer
import numpy as np
import os
from new_eval import evaluate

STATE_DEBUG = False
DEBUG = False
TRAIN_DEBUG = False
GAME_NAME = 'BreakoutDeterministic-v4'
MODEL_NAME = None
SAVE_NAME = 'Breakout_v3_ai'
TRAIN = True
AGENT_HIST = 4
GAMMA = 0.99
EXPLORE_FRAMES = 1_000_000
REPLAY_START_SIZE = 50_000
DEF_EPS = 0.05
LEARNING_RATE = 0.00025
SHOW_EVERY = 10000
UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 10_000
NO_OP_MAX = 30
TOTAL_FRAMES = 50_000_000
BATCH_SIZE = 32
INIT_EPS = 1.0
FINAL_EPS = 0.1
REPLAY_SIZE = 1_000_000
EVAL_STEPS = 250_000
EVAL_DUR_FRAMES = 135_000
NET_DIR = 'networks'
LOG_DIR = 'vlog'
SAMPN_DIR = 'sample_net'
SAMPLE_DIR = 'sample_v'

if not os.path.exists(NET_DIR):
    os.makedirs(NET_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# def save(net, frames, gen_count, rewards):
#     net.save(NET_DIR + '/' + 'generation-{}-reward-{}'.format(gen_count, rewards))
#     np.save(LOG_DIR + '/' + 'generation-{}-reward-{}'.format(gen_count, rewards), frames)
#

def run():
    max_rewards = 0
    gen_count = 0
    max_eval = 0
    frame_count = 0
    eps_count = 0
    env = Game(key=GAME_NAME)
    rm = ReplayMemory(size = REPLAY_SIZE, agent_history_length=AGENT_HIST, batch_size=32)
    net = DQN(learning_rate=LEARNING_RATE, action_space=env.actions_n(),gamma=GAMMA, batch_size=BATCH_SIZE)
    o = Observer()
    eval_o = Observer()
    if MODEL_NAME is not None:
        net.load(MODEL_NAME)
    epsilon = Epsilon(total_frames=TOTAL_FRAMES)
    while frame_count < TOTAL_FRAMES:
        state, frame = env.reset()
        terminal = False
        while not terminal:
            if TRAIN:
                if STATE_DEBUG:
                    env.show_debug(state)
                action = epsilon.gen_action(state, net, DEBUG)
            else:
                action = epsilon.self_act(state, net, DEF_EPS)
            new_state, reward, terminal, new_frame, life_lost = env.step(action)
            if TRAIN:
                rm.add_experience(action=action, frame=frame[:, :], reward=reward, terminal=life_lost)
                if frame_count >= REPLAY_START_SIZE and frame_count % UPDATE_FREQ == 0:
                    net.learn(rm.get_minibatch(), TRAIN_DEBUG)

                if (frame_count + 1) % TARGET_UPDATE_FREQ == 0:
                    net.update()
                    net.save(SAVE_NAME)
                    o.store(name='train_rewards')
                    print('TARGET UPDATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                if (frame_count + 1) % EVAL_STEPS == 0:
                    val = evaluate(net, EVAL_DUR_FRAMES, NO_OP_MAX, DEF_EPS)
                    eval_o.add(val)
                    if val >= max_eval:
                        net.save(NET_DIR + '/' + 'generation-{}-eval-reward-{}.model'.format(gen_count, int(val * 100) / 100.0))
                        gen_count += 1
                        max_eval = val
                    eval_o.store(name = 'eval_rewards')

            if eps_count % SHOW_EVERY == 0:
                env.render()

            state = new_state
            frame = new_frame
            frame_count += 1
            epsilon.step()
        eps_count += 1
        o.add(env.reward_total)
        if env.reward_total >= max_rewards:
            max_rewards = env.reward_total
            np.save(LOG_DIR + '/' + 'episode-{}-reward-{}'.format(eps_count, max_rewards), env.getFrames())

        print('epsilon: {}, rewards: {}, frame_num: {}, episode: {}'.format(epsilon.get(), env.reward_total, frame_count, eps_count))


if __name__ == '__main__':
    run()
