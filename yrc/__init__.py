from omegaconf import OmegaConf

import yrc.core.algorithm as algo_factory
import yrc.core.environment as env_factory
import yrc.core.policy as policy_factory
from yrc.core import CoordEnv, Evaluator
from yrc.core.config import YRCBenchConfig, configure


def make_config(args, dotlist_args=None):
    config = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    if dotlist_args is not None:
        cli_config = OmegaConf.from_dotlist(dotlist_args)
        config = OmegaConf.merge(config, cli_config)
    config = YRCBenchConfig(**OmegaConf.to_container(config, resolve=True))

    configure(config)

    return config


def make_base_env(split, config):
    return env_factory.make_base_env(split, config)


def make_algorithm(config, env):
    return algo_factory.make(config, env)


def make_policy(config, env):
    return policy_factory.make(config, env)


def make_evaluator(config):
    return Evaluator(config)


def load_policy(path, env):
    return policy_factory.load(path, env)
