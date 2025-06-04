from omegaconf import OmegaConf

import yrc
import yrc.core.algorithm as algo_factory
import yrc.core.environment as env_factory
import yrc.core.policy as policy_factory
from yrc.core.config import YRCConfig, configure
from yrc.core.environment import CoordEnv
from yrc.core.evaluator import Evaluator


def make_config(args, dotlist_args=None):
    config = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    if dotlist_args is not None:
        cli_config = OmegaConf.from_dotlist(dotlist_args)
        config = OmegaConf.merge(config, cli_config)
    config = YRCConfig(**OmegaConf.to_container(config, resolve=True))

    configure(config)

    return config


def make_base_env(split, config):
    return env_factory.make_base_env(split, config)


def make_algorithm(config):
    return algo_factory.make(config)


def make_policy(config, env):
    return policy_factory.make(config, env)


def load_policy(path, env):
    return policy_factory.load(path, env)


def register_config(name, config_cls):
    yrc.environments.config_cls[name] = config_cls
