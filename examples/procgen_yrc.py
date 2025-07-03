import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import yrc
from environments.procgen import ProcgenConfig, make_env

splits = ["train", "test"]


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
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Mode to run",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["agent", "coord"],
        default="coord",
        help="Policy type",
    )
    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)

    return args, config


def compute_reward_per_action(config):
    with open("environments/metadata/test_eval_info.json") as f:
        data = json.load(f)
    test_eval_info = data[config.name][config.task]
    mean_episode_reward = test_eval_info["reward_mean"]
    mean_episode_length = test_eval_info["episode_length_mean"]
    reward_per_action = mean_episode_reward / mean_episode_length
    return reward_per_action


def make_base_envs(config):
    base_envs = {}
    for split in splits:
        base_envs[split] = make_env(split, config.env)
    return base_envs


def make_coord_envs(config, base_envs):
    some_base_env = list(base_envs.values())[0]
    train_novice = yrc.load_policy(config.train_novice, some_base_env)
    train_expert = yrc.load_policy(config.train_expert, some_base_env)
    test_novice = yrc.load_policy(config.test_novice, some_base_env)
    test_expert = yrc.load_policy(config.test_expert, some_base_env)

    envs = {}
    for split in splits:
        if split in ["train", "val_sim"]:
            novice, expert = train_novice, train_expert
        else:
            novice, expert = test_novice, test_expert
        envs[split] = yrc.CoordEnv(
            config.coordination,
            base_envs[split],
            novice,
            expert,
            # open_expert=True,    # uncomment to use features from expert
        )

    # Set costs for the coordination environment
    base_penalty = compute_reward_per_action(config.env)
    for split in splits:
        envs[split].set_costs(base_penalty)

    return envs


def train(args, config):
    base_envs = make_base_envs(config)
    envs = make_coord_envs(config, base_envs) if args.type == "coord" else base_envs
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split == "train":
            continue
        validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def evaluate(args, config):
    base_envs = make_base_envs(config)
    envs = make_coord_envs(config, base_envs) if args.type == "coord" else base_envs

    if config.policy.load_path is not None:
        # Load the policy from the specified path
        policy = yrc.load_policy(config.policy.load_path, envs[splits[0]])
    else:
        logging.info("WARNING: No policy load path specified, using default policy.")
        policy = yrc.make_policy(config.policy, envs[splits[0]])

    for split in splits:
        if split == "train":
            continue
        logging.info(f"Evaluating on {split} split")
        evaluator = yrc.Evaluator(config.evaluation, envs[split])
        evaluator.evaluate(policy)


def main():
    # register the Procgen configuration with YRC
    # NOTE: do this before parsing args to ensure the config is available
    yrc.register_environment("procgen", ProcgenConfig)

    args, config = parse_args()
    if args.mode == "train":
        train(args, config)
    elif args.mode == "eval":
        evaluate(args, config)


if __name__ == "__main__":
    main()
