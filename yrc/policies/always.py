from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class AlwaysPolicyConfig:
    """
    Configuration dataclass for AlwaysPolicy.

    Parameters
    ----------
    cls : str, optional
        Name of the policy class. Default is "AlwaysPolicy".
    agent : str, optional
        The agent type to always select. Options are "novice" or "expert". Default is "novice".
    """

    cls: str = "AlwaysPolicy"
    agent: str = "novice"
    load_path: Optional[str] = None


class AlwaysPolicy(Policy):
    def __init__(self, config, env):
        self.choice = env.NOVICE if config.agent == "novice" else env.EXPERT
        self.device = get_global_variable("device")
        self.config = config

    def act(self, obs, temperature=None):
        if isinstance(obs, dict):
            batch_size = obs["base_obs"].shape[0]
        elif isinstance(obs, np.ndarray):
            batch_size = obs.shape[0]
        else:
            raise ValueError("obs must be a dict or a numpy array")

        return torch.ones((batch_size,)).to(self.device) * self.choice

    def reset(self, done: "numpy.ndarray") -> None:
        pass

    def load_model_checkpoint(self, model_state_dict):
        pass

    def train(self):
        pass

    def eval(self):
        pass
