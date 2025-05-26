import importlib
import logging

import torch

from yrc.utils.global_variables import get_global_variable


def make(config, env):
    policy_cls = getattr(importlib.import_module("yrc.policies"), config.cls)
    policy = policy_cls(config, env)
    return policy


def load(path, env):
    ckpt = torch.load(path, map_location=get_global_variable("device"))
    config = ckpt["policy_config"]

    policy_cls = getattr(importlib.import_module("yrc.policies"), config.cls)
    policy = policy_cls(config, env)
    policy.load_model_checkpoint(ckpt["model_state_dict"])
    logging.info(f"Loaded policy from {path}")

    return policy


class Policy:
    # get logit
    def forward(self, obs):
        pass

    # get action distribution
    def predict(self, obs):
        pass

    # draw an action
    def act(self, obs, greedy=False):
        pass

    # update model parameters
    def update_params(self):
        pass

    # get pre-softmax hidden features
    def get_hidden(self):
        pass

    # set to training mode
    def train(self):
        pass

    # set to eval mode
    def eval(self):
        pass

    def reset(self, should_reset):
        pass

    def load_model_checkpoint(self, load_path):
        pass
