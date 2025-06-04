import logging
import os
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
        self.args = config.coord_policy
        self.params = {"threshold": config.threshold, "temperature": config.temperature}
        self.device = get_global_variable("device")

    def act(self, obs, temperature=None):
        logits = obs["novice_logits"]
        if not torch.is_tensor(logits):
            logits = torch.from_numpy(logits).to(self.device).float()
        score = self.compute_confidence(logits)
        action = (score < self.params["threshold"]).long()
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
                # Original behavior for multi-class case
                top2 = logits.softmax(dim=-1).topk(2, dim=-1)[0]
                score = top2[:, 0] - top2[:, 1]
                score = score.unsqueeze(-1)
            else:
                # Binary case when logit has shape (..., 1)
                prob = logits.sigmoid().unsqueeze(-1)
                score = torch.abs(2 * prob - 1)
        elif metric == "neg_entropy":
            score = -Categorical(logits=logits).entropy()
        elif metric == "neg_energy":
            score = logits.logsumexp(dim=-1)
        else:
            raise NotImplementedError(f"Unrecognized metric: {metric}")

        return score

    def update_params(self, params):
        self.params = dc(params)

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save(self.params, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        self.params = torch.load(load_path)
