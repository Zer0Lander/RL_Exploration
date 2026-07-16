"""Soft Actor-Critic (Haarnoja et al. 2018).

Maximum-entropy RL: maximise return plus an entropy bonus,
    J = E[ sum_t r_t + alpha * H(pi(.|s_t)) ].
alpha is auto-tuned to hold the policy entropy near a target of -act_dim.

Each update, on a minibatch from the replay buffer:
  1. Critic: regress both Q-networks toward the soft Bellman target
        y = r + gamma * (1-done) * (min(Q1', Q2')(s', a') - alpha*logpi(a'|s'))
     with a' sampled from the current policy and Q' the target networks.
  2. Actor: shift the policy toward actions the critics score highly,
        maximise min(Q1, Q2)(s, a) - alpha * logpi(a|s).
  3. Alpha: raise it when entropy is below target, lower it otherwise.
  4. Polyak-average the target critics toward the live critics.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .networks import Actor, Critic


@dataclass
class SACConfig:
    gamma: float = 0.99        # discount factor
    tau: float = 0.005         # polyak averaging speed for target networks
    lr: float = 3e-4
    batch_size: int = 256


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, cfg: SACConfig = SACConfig()):
        self.cfg = cfg
        self.device = device

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Learn log_alpha rather than alpha to keep the temperature positive.
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.act(obs_t, deterministic).squeeze(0).cpu().numpy()

    def update(self, buffer) -> dict:
        obs, act, rew, next_obs, done = buffer.sample(self.cfg.batch_size)

        # 1. critic update
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            soft_q_next = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = rew.unsqueeze(-1) + self.cfg.gamma * (1 - done.unsqueeze(-1)) * soft_q_next

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. actor update (freeze critics so only the actor gets gradients)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_act, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_act)
        actor_loss = (self.alpha.detach() * logp - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # 3. temperature update
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. polyak-average the target critics toward the live critics
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.lerp_(p, self.cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
