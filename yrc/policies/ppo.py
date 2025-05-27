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
        self.config = config
        if model is None:
            model_cls = getattr(model_factory, config.model.cls)
            self.model = model_cls(config.model, env)
        else:
            self.model = model
        self.model.to(get_global_variable("device"))

    def reset(self, done: "numpy.ndarray") -> None:
        pass

    def act(self, obs, greedy=False):
        logit, _ = self.model(obs)
        dist = Categorical(logits=logit)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.cpu().numpy()

    def get_action_and_value(self, obs, action=None):
        logit, value = self.model(obs)
        dist = Categorical(logits=logit)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        _, value = self.model(obs)
        return value

    def load_model_checkpoint(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)
