import logging
import os
from dataclasses import dataclass

import torch

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class RandomPolicyConfig:
    prob: float = 0.5


class RandomPolicy(Policy):
    def __init__(self, config, env):
        self.prob = config.coord_policy.prob
        self.device = get_global_variable("device")

    def act(self, obs, greedy=False):
        benchmark = get_global_variable("benchmark")
        env_obs = obs["env_obs"]

        if isinstance(env_obs, dict):
            if benchmark == "cliport":
                action_shape = (1,)
            elif benchmark == "minigrid":
                action_shape = (env_obs["direction"].shape[0],)
        else:
            action_shape = (env_obs.shape[0],)

        action = torch.rand(action_shape).to(self.device) < self.prob
        action = action.int()
        return action.cpu().numpy()

    def update_params(self, prob):
        self.prob = prob

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save({"prob": self.prob}, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.prob = ckpt["prob"]
