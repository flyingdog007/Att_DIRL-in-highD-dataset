import torch
import torch.nn as nn
from torch.distributions import Normal

from config import (
    ACTOR_INPUT_DIM,
    ACTOR_HIDDEN_DIMS,
    OUTPUT_DIM,
    ACTION_LOG_STD_MIN,
    ACTION_LOG_STD_MAX,
)


class Actor(nn.Module):

    def __init__(self,
                 input_dim=ACTOR_INPUT_DIM,
                 hidden_dims=None,
                 output_dim=OUTPUT_DIM):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = list(ACTOR_HIDDEN_DIMS)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(in_dim, output_dim)
        self.log_std_head = nn.Linear(in_dim, output_dim)

    def forward(self, actor_input):
        features = self.backbone(actor_input)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = log_std.clamp(ACTION_LOG_STD_MIN, ACTION_LOG_STD_MAX)
        return mu, log_std

    def get_action_and_logprob(self, actor_input):
        mu, log_std = self.forward(actor_input)
        std = log_std.exp()

        dist = Normal(mu, std)
        u = dist.rsample()

        action_norm = torch.tanh(u)

        log_prob = dist.log_prob(u)
        log_prob = log_prob - torch.log(1.0 - action_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action_norm, log_prob, entropy

    def get_deterministic_action(self, actor_input):
        mu, _ = self.forward(actor_input)
        action_norm = torch.tanh(mu)
        return action_norm

    def evaluate_action(self, actor_input, action_norm):
        mu, log_std = self.forward(actor_input)
        std = log_std.exp()
        dist = Normal(mu, std)

        u = torch.atanh(action_norm.clamp(-0.999, 0.999))

        log_prob = dist.log_prob(u) - torch.log(1.0 - action_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
