from collections import deque
import random
import torch


class ReplayBuffer():
    def __init__(self):
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.buffer = deque(maxlen=1_000_000)
        self.batch_size = 32  # minibatch sample size for training

    def store(self, state, action, reward, next_state, done):
        transitions = list(
            zip(state, action, reward, next_state, 1 - torch.tensor(done, device=self.device).to(int)))
        self.buffer.extend(transitions)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        # states, actions, rewards, next_states, not_dones
        return [torch.stack(e).to(self.device) for e in zip(*batch)]
