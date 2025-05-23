import logging

import torch

from YRC.models.rl import PPOModel
from YRC.policies.base import BasePolicy


class PPOPolicy(BasePolicy):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.model = PPOModel(self.model)
        self.model.to(self.model.device)

    def forward(self, obs):
        return self.model(obs)

    def get_action_and_value(self, obs, action=None):
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def predict(self, obs):
        dist, _ = self.forward(obs)
        return dist

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

    def set_learning_rate(self, learning_rate):
        self.optim.param_groups[0]["lr"] = learning_rate
