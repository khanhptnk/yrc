import argparse
import json
import logging
import pprint
from copy import deepcopy as dc

import yrc
from yrc.utils.global_variables import get_global_variable


def get_test_eval_info(config, env):
    with open("yrc/core/test_eval_info.json") as f:
        data = json.load(f)

    backup_data = dc(data)

    env_suite = get_global_variable("env_suite")
    env_name = config.env.name

    if env_name not in data[env_suite]:
        logging.info(f"Missing info about {env_suite}-{env_name}!")
        logging.info("Calculating missing info (taking a few minutes)...")
        evaluator = yrc.Evaluator(config.evaluation)
        # eval expert agent on test environment to get statistics
        summary = evaluator.eval(
            env["test"].expert,
            {"test": env["test"].base_env},
            ["test"],
            num_episodes=env["test"].num_envs,
        )["test"]
        data[env_suite][env_name] = summary

        with open("yrc/core/backup_test_eval_info.json", "w") as f:
            json.dump(backup_data, f, indent=2)
        with open("yrc/core/test_eval_info.json", "w") as f:
            json.dump(data, f, indent=2)
        logging.info("Saved info!")

    info = data[env_suite][env_name]

    logging.info(f"{pprint.pformat(info, indent=2)}")
    return info


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

    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)

    return config


def main():
    config = parse_args()

    splits = ["train", "val_sim", "val_true", "test"]

    base_env = {}
    for split in splits:
        base_env[split] = yrc.make_base_env(split, config.env)

    train_novice = yrc.load_policy(config.train_novice, base_env["train"])
    train_expert = yrc.load_policy(config.train_expert, base_env["train"])
    test_novice = yrc.load_policy(config.test_novice, base_env["test"])
    test_expert = yrc.load_policy(config.test_expert, base_env["test"])

    env = {}
    for split in splits:
        if split in ["train", "val_sim"]:
            novice, expert = train_novice, train_expert
        else:
            novice, expert = test_novice, test_expert
        env[split] = yrc.CoordEnv(config.coordination, base_env[split], novice, expert)

    test_eval_info = get_test_eval_info(config, env)
    for split in env:
        env[split].set_costs(test_eval_info)

    policy = yrc.make_policy(config.policy, base_env["train"])
    algorithm = yrc.make_algorithm(config.algorithm, base_env["train"])
    evaluator = yrc.make_evaluator(config.evaluation)

    algorithm.train(
        policy,
        base_env,
        evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
    )


if __name__ == "__main__":
    main()
