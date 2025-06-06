from dataclasses import dataclass
from typing import Any, Optional

import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import yrc.models as model_factory
from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class PPOPolicyConfig:
    """
    Configuration dataclass for PPOPolicy.

    Parameters
    ----------
    cls : str, optional
        Name of the policy class. Default is "PPOPolicy".
    model : Any, optional
        Model configuration or class name. Default is "ImpalaCoordPPOModel".
    load_path : Optional[str], optional
        Path to a checkpoint to load the policy weights from. Default is None.
    """

    cls: str = "PPOPolicy"
    model: Any = "ImpalaCoordPPOModel"
    load_path: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = model_factory.config_cls[self.model]()
        elif isinstance(self.model, dict):
            if "cls" not in self.model:
                raise IndexError(
                    "Please specify policy.model.cls through YAML file or flag"
                )
            self.model = model_factory.config_cls[self.model["cls"]](**self.model)
        else:
            raise ValueError("model must be a string or a dictionary")


class PPOPolicy(Policy):
    def __init__(self, config, env):
        model_cls = getattr(model_factory, config.model.cls)
        self.model = model_cls(config.model, env)
        self.model.to(get_global_variable("device"))
        self.config = config

    def reset(self, done: "numpy.ndarray") -> None:
        pass

    def act(self, obs, temperature=1.0, return_model_output=False):
        model_output = self.model(obs)
        if temperature == 0:
            action = model_output.logits.argmax(dim=-1)
        else:
            dist = Categorical(logits=model_output.logits)
            action = dist.sample()
        if return_model_output:
            return action, model_output
        return action

    def set_params(self, params):
        self.model.load_state_dict(params)

    def get_params(self):
        return self.model.state_dict()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
