import torch.nn as nn

from config import (
    CRITIC_INPUT_DIM,
    CRITIC_HIDDEN_DIMS,
    CRITIC_OUTPUT_DIM,
)


class Critic(nn.Module):

    def __init__(self,
                 input_dim=CRITIC_INPUT_DIM,
                 hidden_dims=None,
                 output_dim=CRITIC_OUTPUT_DIM):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = list(CRITIC_HIDDEN_DIMS)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, critic_input):
        return self.network(critic_input)
