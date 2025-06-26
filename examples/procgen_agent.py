import argparse
import importlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import environments
import yrc
from environments.procgen.config import ProcgenConfig


def parse_args():
    parser = argparse.ArgumentParser(description="YRC training script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        default=["test"],
        help="Evaluation splits (space-separated list)",
    )
    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)
    return args, config


def make_base_env(split, config):
    module = importlib.import_module(f"env_utils.{config.suite}")
    make_fn = getattr(module, "make_env")
    return make_fn(split, config)


def train(args, config):
    envs = {
        split: environments.procgen.make_env(split, config.env)
        for split in ["train"] + args.eval_splits
    }
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in args.eval_splits:
        validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def evaluate(args, config):
    eval_splits = args.eval_splits
    envs = {
        split: environments.procgen.make_env(split, config.env) for split in eval_splits
    }
    policy = yrc.load_policy(config.policy.load_path, envs[eval_splits[0]])

    for split in eval_splits:
        evaluator = yrc.Evaluator(config.evaluation, envs[split])
        evaluator.evaluate(policy)


def main():
    # register the Procgen configuration with YRC
    # NOTE: do this before parsing args to ensure the config is available
    yrc.register_env_config("procgen", ProcgenConfig)

    args, config = parse_args()
    if config.eval_name is None:
        train(args, config)
    else:
        evaluate(args, config)


if __name__ == "__main__":
    main()
