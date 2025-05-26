import argparse

import yrc


def parse_args():
    parser = argparse.ArgumentParser(description="YRC training script")
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

    base_env = {}
    for split in ["train", "test"]:
        base_env[split] = yrc.make_base_env(split, config.env)

    policy = yrc.make_policy(config.policy, base_env["train"])
    algorithm = yrc.make_algorithm(config.algorithm, base_env["train"])
    evaluator = yrc.make_evaluator(config.evaluation)

    algorithm.train(
        policy,
        base_env,
        evaluator,
        train_split="train",
        eval_splits=["test"],
    )


if __name__ == "__main__":
    main()
