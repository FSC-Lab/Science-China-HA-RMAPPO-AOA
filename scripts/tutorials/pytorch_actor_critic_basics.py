#!/usr/bin/env python3
"""Minimal PyTorch actor-critic/PPO mechanics for undergraduate study.

This is not a complete RL algorithm: it uses a fixed synthetic batch so the tensor
shapes, distributions, gradients, and clipped PPO objective remain easy to inspect.
Run from the repository root:

    python scripts/tutorials/pytorch_actor_critic_basics.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Separate MLP actor and critic with a learned Gaussian log standard deviation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)


def joint_log_prob(distribution: Normal, action: torch.Tensor) -> torch.Tensor:
    """Sum independent action-dimension log probabilities, preserving [B, 1]."""

    return distribution.log_prob(action).sum(dim=-1, keepdim=True)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    obs_dim = 10
    action_dim = 2
    clip_param = 0.2
    entropy_coef = 0.01
    value_coef = 0.5

    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs = torch.randn(batch_size, obs_dim, device=device)

    # Pretend these actions and log probabilities were collected by the old policy.
    with torch.no_grad():
        old_distribution = model.distribution(obs)
        actions = old_distribution.sample()
        old_log_probs = joint_log_prob(old_distribution, actions)
        old_values = model.value(obs)

        # Synthetic learning targets stand in for GAE advantages and returns.
        advantages = torch.randn(batch_size, 1, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        returns = old_values + advantages

    print(f"device:        {device}")
    print(f"observations:  {tuple(obs.shape)}")
    print(f"actions:       {tuple(actions.shape)}")
    print(f"log probs:     {tuple(old_log_probs.shape)}")
    print(f"values:        {tuple(old_values.shape)}")

    for update in range(1, 6):
        distribution = model.distribution(obs)
        new_log_probs = joint_log_prob(distribution, actions)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        values = model.value(obs)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(
            ratio, 1.0 - clip_param, 1.0 + clip_param
        ) * advantages

        policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
        value_loss = 0.5 * (returns - values).square().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        assert torch.isfinite(loss)
        assert actions.shape == (batch_size, action_dim)
        assert new_log_probs.shape == (batch_size, 1)
        assert values.shape == (batch_size, 1)

        print(
            f"update {update}: total={loss.item(): .4f}, "
            f"policy={policy_loss.item(): .4f}, value={value_loss.item(): .4f}, "
            f"entropy={entropy.mean().item(): .4f}, grad_norm={float(grad_norm): .4f}"
        )

    with torch.no_grad():
        deterministic_action = model.actor(obs[:1])
    print(f"deterministic action shape: {tuple(deterministic_action.shape)}")
    print("Tutorial completed successfully.")


if __name__ == "__main__":
    main()
