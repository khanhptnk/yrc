import logging

import torch
from omegaconf import OmegaConf

import yrc
from yrc.core.config import YRCConfig, configure
from yrc.core.environment import CoordEnv
from yrc.core.evaluator import Evaluator
from yrc.utils.global_variables import get_global_variable


def make_config(args, dotlist_args=None):
    config = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    if dotlist_args is not None:
        cli_config = OmegaConf.from_dotlist(dotlist_args)
        config = OmegaConf.merge(config, cli_config)
    config = YRCConfig(**OmegaConf.to_container(config, resolve=True))

    configure(config)

    return config


def make_algorithm(config):
    return yrc.algorithms.registry[config.name](config)


def make_policy(config, env):
    return yrc.policies.registry[config.name](config, env)


def load_policy(path, env):
    ckpt = torch.load(
        path, map_location=get_global_variable("device"), weights_only=False
    )
    config = ckpt["policy_config"]

    policy = make_policy(config, env)
    policy.set_params(ckpt["model_state_dict"])
    logging.info(f"Loaded policy from {path}")

    return policy


def register_environment(name, config_cls):
    yrc.environments.registry[name] = config_cls


def register_algorithm(name, algorithm_cls):
    yrc.algorithms.registry[name] = algorithm_cls


def register_policy(name, policy_cls):
    yrc.policies.registry[name] = policy_cls


def register_model(name, model_cls):
    yrc.models.registry[name] = model_cls
