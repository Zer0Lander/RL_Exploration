"""Actor and critic networks for SAC.

Actor: a Gaussian policy. It outputs a mean and log-std, samples an action,
and squashes it through tanh into [-1, 1] (joint commands are bounded).

Critic: twin Q-networks. Using two and taking the minimum ("clipped double-Q")
counters the overestimation bias a single Q-network picks up from its own
errors in the Bellman target.
"""

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)  # outputs mean and log_std
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob(action))."""
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        dist = torch.distributions.Normal(mean, log_std.exp())

        u = dist.rsample()          # reparameterised sample, so gradients flow through
        action = torch.tanh(u)      # squash into [-1, 1]

        # Correct the log-prob for the tanh squash (change of variables). This is
        # the numerically stable form of log(1 - tanh(u)^2).
        log_prob = dist.log_prob(u) - (2 * (torch.log(torch.tensor(2.0)) - u - torch.nn.functional.softplus(-2 * u)))
        return action, log_prob.sum(-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Action for env interaction, no gradient needed."""
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        if deterministic:  # evaluation: take the mode
            return torch.tanh(mean)
        std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        return torch.tanh(mean + std * torch.randn_like(std))


class Critic(nn.Module):
    """Twin Q-networks. Called with (obs, act), returns (q1, q2)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1)
        self.q2 = mlp(obs_dim + act_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
