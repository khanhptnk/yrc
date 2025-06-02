import argparse

import yrc
from yrc.utils.evaluation import get_test_eval_info


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

    policy = yrc.make_policy(config.policy, env["train"])
    algorithm = yrc.make_algorithm(config.algorithm)
    evaluator = yrc.Evaluator(config.evaluation)

    algorithm.train(
        policy,
        env,
        evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
    )


if __name__ == "__main__":
    main()
