import torch
import torch.nn as nn
import torch.nn.functional as F

from actor import Actor
from critic import Critic
from config import (
    PPO_CLIP_EPS,
    PPO_VF_COEF,
    PPO_ENTROPY_COEF,
)


class PPOAgent(nn.Module):

    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()

    def get_action_train(self, actor_input):
        action_norm, log_prob, entropy = self.actor.get_action_and_logprob(actor_input)
        return action_norm, log_prob, entropy

    def get_action_infer(self, actor_input):
        return self.actor.get_deterministic_action(actor_input)

    def compute_ppo_loss(self, log_prob_new, log_prob_old, advantage,
                         value, R_target, entropy):
        ratio = torch.exp(log_prob_new - log_prob_old)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantage
        L_clip = torch.min(surr1, surr2).mean()

        L_vf = F.mse_loss(value.squeeze(-1), R_target.squeeze(-1))

        L_ent = entropy.mean()

        L_ppo = -L_clip + PPO_VF_COEF * L_vf - PPO_ENTROPY_COEF * L_ent

        return {
            'loss_ppo':  L_ppo,
            'loss_clip': L_clip.detach(),
            'loss_vf':   L_vf.detach(),
            'loss_ent':  L_ent.detach(),
        }
