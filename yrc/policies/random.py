from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class RandomPolicyConfig:
    """
    Configuration dataclass for RandomPolicy.

    Parameters
    ----------
    cls : str, optional
        Name of the policy class. Default is "RandomPolicy".
    prob : float, optional
        Probability of selecting the expert action. Setting this value prevents RandomAlgorithm from conducting a grid search.

    Examples
    --------
    >>> config = RandomPolicyConfig(prob=0.7)
    >>> print(config.cls)
    'RandomPolicy'
    >>> print(config.prob)
    0.7
    """

    cls: str = "RandomPolicy"
    prob: Optional[float] = None
    load_path: Optional[str] = None


class RandomPolicy(Policy):
    def __init__(self, config, env):
        self.prob = config.prob
        self.device = get_global_variable("device")
        self.EXPERT = env.EXPERT
        self.config = config

    def act(self, obs, temperature=None):
        if isinstance(obs, dict):
            batch_size = obs["base_obs"].shape[0]
        elif isinstance(obs, np.ndarray):
            batch_size = obs.shape[0]
        else:
            raise ValueError("obs must be a dict or a numpy array")

        action = torch.where(
            torch.rand(batch_size, device=self.device) < self.prob,
            self.EXPERT,
            1 - self.EXPERT,
        )
        return action

    def reset(self, done):
        pass

    def set_params(self, params):
        self.prob = params["prob"]

    def get_params(self):
        return {"prob": self.prob}

    def train(self):
        pass

    def eval(self):
        pass
