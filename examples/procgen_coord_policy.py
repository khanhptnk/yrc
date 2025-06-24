import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import environments
import yrc
from environments.procgen import ProcgenConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="YRC coordination policy training script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file",
    )

    parser.add_argument(
        "--eval_splits",
        nargs="+",
        default=["val_sim", "val_true"],
        help="Evaluation splits (space-separated list)",
    )
    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)

    return args, config


def compute_reward_per_action(config):
    with open("environments/metadata/test_eval_info.json") as f:
        data = json.load(f)
    test_eval_info = data[config.suite][config.name]
    mean_episode_reward = test_eval_info["reward_mean"]
    mean_episode_length = test_eval_info["episode_length_mean"]
    reward_per_action = mean_episode_reward / mean_episode_length
    return reward_per_action


def train(args, config):
    eval_splits = args.eval_splits
    splits = ["train"] + eval_splits

    base_envs = {}
    for split in splits:
        base_envs[split] = environments.procgen.make_env(split, config.env)

    train_novice = yrc.load_policy(config.train_novice, base_envs["train"])
    train_expert = yrc.load_policy(config.train_expert, base_envs["train"])
    test_novice = yrc.load_policy(config.test_novice, base_envs[eval_splits[0]])
    test_expert = yrc.load_policy(config.test_expert, base_envs[eval_splits[0]])

    reward_per_action = compute_reward_per_action(config.env)

    envs = {}
    for split in splits:
        if split in ["train", "val_sim"]:
            novice, expert = train_novice, train_expert
        else:
            novice, expert = test_novice, test_expert
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )

    # Set costs for the coordination environment
    reward_per_action = compute_reward_per_action(config.env)
    for split in splits:
        envs[split].set_costs(reward_per_action)

    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)
    evaluator = yrc.Evaluator(config.evaluation)

    algorithm.train(
        policy,
        envs,
        evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
    )


def evaluate(args, config):
    splits = args.eval_splits

    base_envs = {}
    for split in splits:
        base_envs[split] = environments.procgen.make_env(split, config.env)

    novice = yrc.load_policy(config.test_novice, base_envs[splits[0]])
    expert = yrc.load_policy(config.test_expert, base_envs[splits[0]])

    envs = {}
    for split in splits:
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )

    # Set costs for the coordination environment
    reward_per_action = compute_reward_per_action(config.env)
    for split in splits:
        envs[split].set_costs(reward_per_action)

    if config.policy.load_path is not None:
        # Load the policy from the specified path
        policy = yrc.load_policy(config.policy.load_path, envs[splits[0]])
    else:
        logging.info("WARNING: No policy load path specified, using default policy.")
        policy = yrc.make_policy(config.policy, envs[splits[0]])
    evaluator = yrc.Evaluator(config.evaluation)

    evaluator.eval(policy, envs, splits)


def main():
    # register the Procgen configuration with YRC
    # NOTE: do this before parsing args to ensure the config is available
    yrc.register_config("procgen", ProcgenConfig)

    args, config = parse_args()
    if config.eval_name is None:
        train(args, config)
    else:
        evaluate(args, config)


if __name__ == "__main__":
    main()
