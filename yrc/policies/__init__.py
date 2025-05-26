import logging

import torch

from yrc.core.utils import get_global_variable

from .always import AlwaysPolicy, AlwaysPolicyConfig
from .logit import LogitPolicy, LogitPolicyConfig
from .ood import OODPolicy, OODPolicyConfig
from .ppo import PPOPolicy, PPOPolicyConfig
from .random import RandomPolicy, RandomPolicyConfig

config_cls = {
    "AlwaysPolicy": AlwaysPolicyConfig,
    "LogitPolicy": LogitPolicyConfig,
    "OODPolicy": OODPolicyConfig,
    "PPOPolicy": PPOPolicyConfig,
    "RandomPolicy": RandomPolicyConfig,
}


def load(path, env):
    ckpt = torch.load(path, map_location=get_global_variable("device"))
    config = ckpt["policy_config"]

    policy_cls = globals()[config.cls]

    policy = policy_cls(config, env)
    policy.load_model_checkpoint(ckpt["model_state_dict"])
    logging.info(f"Loaded policy from {path}")

    return policy
