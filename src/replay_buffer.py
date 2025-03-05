from collections import deque
import random
import torch

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=1_000_000)
        self.batch_size = 32 # minibatch sample size for training

    def store(self, state, action, reward, next_state, done):
        transitions = list(zip(state, action, reward, next_state, 1 - torch.Tensor(done)))
        self.buffer.extend(transitions)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        # device = "cpu"
        return [torch.stack(e) for e in zip(*batch)]  # states, actions, rewards, next_states, not_dones