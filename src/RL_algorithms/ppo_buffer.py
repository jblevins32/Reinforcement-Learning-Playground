import torch
import numpy as np

class PPOBuffer:
    def __init__(self, state_dim, action_dim, capacity=2048, gamma=0.99, lam=0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam

        # Storage for experiences
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0  # Pointer to track storage index
        self.full = False  # Tracks if buffer is full

    def store(self, state, action, reward, next_state, done, log_prob, value):
        """Stores a single experience tuple."""
        idx = self.ptr % self.capacity  # Circular overwrite

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True        

    def compute_advantages(self):
        """Computes advantages using GAE (Generalized Advantage Estimation)."""
        last_adv = 0
        for t in reversed(range(self.capacity)):
            if t == self.capacity - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            self.advantages[t] = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_adv
            last_adv = self.advantages[t]
        
        self.returns = self.advantages + self.values  # Compute returns

    def sample_batch(self, batch_size=64):
        """Returns a random mini-batch for training PPO."""
        indices = np.random.choice(self.capacity, batch_size, replace=False)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.float32),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.next_states[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.float32),
            torch.tensor(self.log_probs[indices], dtype=torch.float32),
            torch.tensor(self.advantages[indices], dtype=torch.float32),
            torch.tensor(self.returns[indices], dtype=torch.float32),
        )

    def clear(self):
        """Resets buffer."""
        self.ptr = 0
        self.full = False
