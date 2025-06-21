from copy import deepcopy as dc
from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions.categorical import Categorical

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class LogitPolicyConfig:
    cls: str = "LogitPolicy"
    metric: str = "max_logit"
    threshold: Optional[float] = None
    temperature: Optional[float] = None


class LogitPolicy(Policy):
    def __init__(self, config, env):
        self.config = config
        self.params = {"threshold": config.threshold, "temperature": config.temperature}
        self.device = get_global_variable("device")
        self.EXPERT = env.EXPERT

    def act(self, obs, temperature=None):
        logits = obs["novice_logits"]
        if not torch.is_tensor(logits):
            logits = torch.from_numpy(logits).to(self.device).float()
        score = self.compute_confidence(logits)
        # query expert when confidence score < threshold
        action = torch.where(
            score < self.params["threshold"],
            self.EXPERT,
            1 - self.EXPERT,
        )
        return action

    def compute_confidence(self, logits):
        # NOTE: higher = more confident
        metric = self.config.metric
        logits = logits / self.params["temperature"]
        if metric == "max_logit":
            score = logits.max(dim=-1)[0]
        elif metric == "max_prob":
            score = logits.softmax(dim=-1).max(dim=-1)[0]
        elif metric == "margin":
            if logits.size(-1) > 1:
                # Multi-class case
                top2 = logits.softmax(dim=-1).topk(2, dim=-1)[0]
                score = top2[:, 0] - top2[:, 1]
                score = score
            else:
                # Binary case when logits has shape (B, 1)
                prob = logits.sigmoid().squeeze(-1)
                score = torch.abs(2 * prob - 1)
        elif metric == "entropy":
            # NOTE: we compute NEGATIVE entropy so that higher = more confident
            score = -Categorical(logits=logits).entropy()
        elif metric == "energy":
            score = logits.logsumexp(dim=-1)
        else:
            raise NotImplementedError(f"Unrecognized metric: {metric}")
        return score

    def reset(self, done: "numpy.ndarray") -> None:
        pass

    def get_params(self):
        return dc(self.params)

    def set_params(self, params):
        for k, v in params.items():
            if k not in self.params:
                raise KeyError(f"Parameter {k} not recognized in LogitPolicy")
            self.params[k] = dc(v)

    def train(self):
        pass

    def eval(self):
        pass
