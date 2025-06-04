"""
YRC script for Procgen environments.

Training novice:
    python procgen_agent.py --config configs/procgen_ppo.yaml name=$TRAIN_NAME env.train.distribution_mode=easy env.test.distribution_mode=easy

Evaluate novice:
    python examples/procgen_agent.py --config configs/procgen_ppo.yaml --eval eval_name=$EVAL_NAME env.test.distribution_mode=easy policy.load_path=experiments/$TRAIN_NAME/best_test.ckpt

To train/evaluate expert, change the distribution_model to "hard".
For more options, see the ProcgenConfig class in environments/procgen/config.py.

"""

import argparse
import importlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import environments
import yrc
from environments.procgen import ProcgenConfig


def parse_args():
    parser = argparse.ArgumentParser(description="YRC training script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run evaluation instead of training",
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
    evaluator = yrc.Evaluator(config.evaluation)
    algorithm.train(
        policy=policy,
        envs=envs,
        evaluator=evaluator,
        train_split="train",
        eval_splits=args.eval_splits,
    )


def eval(args, config):
    eval_splits = args.eval_splits
    envs = {
        split: environments.procgen.make_env(split, config.env) for split in eval_splits
    }
    policy = yrc.load_policy(config.policy.load_path, envs[eval_splits[0]])
    evaluator = yrc.Evaluator(config.evaluation)
    evaluator.eval(policy, envs, eval_splits)


def main():
    # register the Procgen configuration with YRC
    # NOTE: do this before parsing args to ensure the config is available
    yrc.register_config("procgen", ProcgenConfig)

    args, config = parse_args()
    if args.eval:
        eval(args, config)
    else:
        train(args, config)


if __name__ == "__main__":
    main()
