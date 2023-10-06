
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Add
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.backend import mean
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, learning_rate = 0.00001, action_space = 4, gamma = 0.99, batch_size = 32):
        self.lr = learning_rate
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update()

    def load(self, name):
        self.model = load_model(name)
        self.update()

    def create_model(self):
        inputs = Input(shape=(84, 84, 4))
        first = Conv2D(filters = 32, kernel_size = (8, 8),
                       strides = 2, use_bias = False, padding = 'valid',
                       kernel_initializer = VarianceScaling(scale = 2),
                       activation = 'relu')(inputs)
        second = Conv2D(filters = 64, kernel_size = (4, 4),
                        strides = 2, use_bias = False, padding = 'valid',
                        kernel_initializer = VarianceScaling(scale = 2),
                        activation = 'relu')(first)
        third = Conv2D(filters = 64, kernel_size= (3, 3),
                       strides = 1, use_bias = False, padding = 'valid',
                       kernel_initializer = VarianceScaling(scale = 2),
                       activation = 'relu')(second)

        flatten = Flatten()(third)

        value = Dense(512, use_bias=False, kernel_initializer = VarianceScaling(scale = 2), activation = 'relu')(flatten)
        value = Dense(1, kernel_initializer = VarianceScaling(scale = 2))(value)

        advantage = Dense(512, use_bias=False, kernel_initializer = VarianceScaling(scale = 2), activation = 'relu')(flatten)
        advantage = Dense(self.action_space, kernel_initializer=VarianceScaling(scale=2))(advantage)

        advantage = Lambda(lambda x: x - mean(x))(advantage)

        outputs = Add()([advantage, value])

        model = Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer=Adam(learning_rate = self.lr), loss = Huber(delta=1.0, reduction="auto", name="huber_loss"))

        return model

    def predict(self, state):
        return self.model(state).numpy()[0]

    def learn(self, transitions, verbose = 0):
        states, actions, rewards, new_states, terminals = transitions
        future_qs = (self.target_model(new_states / 255.0)).numpy()

        expected_qs = (self.model(states / 255.0)).numpy()
        for i in range(len(expected_qs)):
            if terminals[i]:
                expected_qs[i][actions[i]] = rewards[i]
            else:
                expected_qs[i][actions[i]] = rewards[i] + self.gamma * max(future_qs[i])
        self.model.fit(states / 255.0, expected_qs, verbose=verbose)

    def update(self):
        self.model.save('intermediate', save_format='h5')
        self.target_model = load_model('intermediate')

    def save(self, name):
        self.model.save(name, save_format='h5')


def test_model():
    from game import Game
    import numpy as np
    env = Game()
    state = env.reset()
    dqn = DQN()
    print(dqn.model(np.array([[state for _ in range(4)]]).reshape((1, 84, 84, 4))/255.0))


def test_train():
    from game import Game
    from RM import ReplayMemory
    import time
    g = Game()
    rm = ReplayMemory()
    for i in range(32):
        done = False
        _, frame = g.reset()
        while not done:
            _, reward, done, new_frame = g.step(1)
            rm.add_experience(1, frame, reward, done)
            frame = new_frame
    dqn = DQN()
    for _ in range(100):
        last = time.time()
        transitions = rm.get_minibatch()
        dqn.learn(transitions)
        print('Time cost: {}'.format(time.time() - last))
    transitions = rm.get_minibatch()

    dqn.update()
    dqn.save('simple_test')
    print(dqn.model(transitions[0][0].reshape(1, 84, 84, 4) / 255.0))
    dqn.model = load_model('simple_test')
    print(dqn.model(transitions[0][0].reshape(1, 84, 84, 4) / 255.0))

    for _ in range(100):
        last = time.time()
        transitions = rm.get_minibatch()
        dqn.learn(transitions)
        print('Time cost: {}'.format(time.time() - last))
    dqn.update()


