"""Ring buffer of (s, a, r, s', done) transitions.

SAC is off-policy, so it can learn from any past experience. We keep a large
buffer of transitions and train on random minibatches, which decorrelates
consecutive steps and stabilises learning.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        # done=1 only when the task actually ends, not on time-limit truncation.
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.acts[i] = act
        self.rews[i] = rew
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        def get(x):
            return torch.as_tensor(x[idx], device=self.device)

        return get(self.obs), get(self.acts), get(self.rews), get(self.next_obs), get(self.dones)

    def __len__(self):
        return self.size
