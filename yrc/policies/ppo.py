from dataclasses import dataclass
from typing import Any, Optional

import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import yrc.models as model_factory
from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class PPOPolicyConfig:
    cls: str = "PPOPolicy"
    model: Any = "ImpalaCoordPPOModel"
    load_path: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = model_factory.config_cls[self.model]()
        elif isinstance(self.model, dict):
            self.model = model_factory.config_cls[self.model["cls"]](**self.model)
        else:
            raise ValueError("model must be a string or a dictionary")


class PPOPolicy(Policy):
    def __init__(self, config=None, env=None, model=None):
        if model is None:
            model_cls = getattr(model_factory, config.model.cls)
            self.model = model_cls(config.model, env)
        else:
            self.model = model
        self.model.to(get_global_variable("device"))
        self.config = config

    def reset(self, done: "numpy.ndarray") -> None:
        pass

    def act(self, obs, greedy=False, return_model_output=False):
        model_output = self.model(obs)
        dist = Categorical(logits=model_output.logits)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        if return_model_output:
            return action, model_output
        return action

    def load_model_checkpoint(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
