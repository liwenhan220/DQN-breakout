import numpy as np


class ReplayMemory:
    def __init__(self, size = 1_000_000, frame_height = 84, frame_width = 84, agent_history_length = 4, batch_size = 32):
        self.size = size + agent_history_length
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.current_length = 0
        self.current_count = 0
        self.end_activate = False

        self.actions = np.empty(self.size, dtype = np.uint8)
        self.rewards = np.empty(self.size, dtype = np.int8)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminals = np.empty(self.size, dtype = np.bool)

        self.states = np.empty((self.batch_size, self.frame_height, self.frame_width, self.agent_history_length), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.frame_height, self.frame_width, self.agent_history_length), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, np.uint32)

    def get_state(self, index):
        count = index
        state = []
        for i in range(self.agent_history_length):
            state.append(self.frames[count])
            if not self.terminals[count - 1]:
                if count - 1 < 0 and (not self.end_activate):
                    continue
                count -= 1
        return np.array(state[::-1], dtype = np.uint8).reshape((self.frame_height, self.frame_width, self.agent_history_length))

    def get_indices(self):
        for i in range(self.batch_size):
            while True:
                index = np.random.randint(0, self.current_length)
                if self.end_activate and index < self.agent_history_length and index + self.size - self.agent_history_length < self.current_count:
                    continue
                if index >= self.current_count and index - self.agent_history_length <= self.current_count:
                    continue
                break
            self.indices[i] = index

    def add_experience(self, action, frame, reward, terminal):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current_count] = action
        self.frames[self.current_count] = frame
        self.rewards[self.current_count] = reward
        self.terminals[self.current_count] = terminal
        if self.current_count + 1 >= self.size:
            self.end_activate = True
        self.current_count = (self.current_count + 1) % self.size
        if self.current_length < self.size:
            self.current_length += 1

    def get_minibatch(self):
        if self.current_length < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self.get_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self.get_state(idx)
            self.new_states[i] = self.get_state((idx + 1) % self.size)

        return self.states, self.actions[self.indices], self.rewards[
            self.indices], self.new_states, self.terminals[self.indices]


def test_replay():
    from game import Game
    g = Game()
    rm = ReplayMemory()
    for i in range(10):
        done = False
        _, frame = g.reset()
        while not done:
            _, reward, done, new_frame, _ = g.step(1)
            rm.add_experience(1, frame, reward, done)
            frame = new_frame

    minibatch = rm.get_minibatch()
    states = minibatch[0]

    print(np.array_equal(states[0], states[0]))
# #
# testReplay()
#
