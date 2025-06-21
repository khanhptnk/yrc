import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class PyODPolicyConfig:
    cls: str = "PyODPolicy"
    method: str = "DeepSVDD"
    feature_type: str = "hidden"
    pyod_config: Optional[Dict[str, Any]] = None


class PyODPolicy(Policy):
    def __init__(self, config, env):
        self.config = config
        self.threshold = None
        self.clf = self._get_pyod_class(config)(**config.pyod_config)

        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.to(get_global_variable("device"))

        self.feature_type = config.feature_type
        self.EXPERT = env.EXPERT

    def _get_pyod_class(self, config):
        try:
            module_name, cls_name = config.method.split(".")
            module_name = f"pyod.models.{module_name}"
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            return cls
        except Exception as e:
            raise ImportError(f"Could not import {config.method} from PyOD: {e}")

    def reset(self, done: "numpy.ndarray") -> None:
        pass

    def _make_input(self, obs):
        inp = []
        if "obs" in self.feature_type:
            base_obs = obs["base_obs"]
            if base_obs.ndim > 2:
                # If env_obs is a tensor with more than 2 dimensions, flatten it
                base_obs = base_obs.reshape(base_obs.shape[0], -1)
            inp.append(base_obs)

        if "hidden" in self.feature_type:
            inp.append(obs["novice_hidden"])
        if "dist" in self.feature_type:
            inp.append(obs["novice_logit"].softmax(dim=-1))

        assert len(inp) > 0, "No features selected for PyOD input"

        inp = np.concatenate(inp, axis=1)

        return inp

    def fit(self, data):
        X = self._make_input(data)
        self.clf.fit(X)

    def get_train_scores(self):
        return self.clf.decision_scores_

    def act(self, obs, temperature=None):
        inp = self._make_input(obs)
        score = self.clf.decision_function(inp)
        score = torch.from_numpy(score).float().to(get_global_variable("device"))

        action = torch.where(
            score < self.threshold,
            self.EXPERT,
            1 - self.EXPERT,
        )
        return action

    def set_params(self, params):
        if "threshold" in params:
            self.threshold = params["threshold"]
        if "clf" in params:
            self.clf = params["clf"]

    def get_params(self):
        return {"threshold": self.threshold, "clf": self.clf}

    def train(self):
        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.train()

    def eval(self):
        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.eval()
