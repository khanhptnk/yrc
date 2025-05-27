import argparse

import yrc


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


if __name__ == "__main__":
    config = parse_args()

    base_env = yrc.make_base_env("test", config.env)

    novice = yrc.load_policy(config.test_novice, base_env)
    expert = yrc.load_policy(config.test_expert, base_env)

    env = yrc.CoordEnv(config.coordination, base_env, novice, expert)

    policy = yrc.make_policy(config.policy, env)

    if config.policy.load_path is not None:
        yrc.load_policy(config.policy.load_path, env)

    evaluator = yrc.make_evaluator(config.evaluation)

    evaluator.eval(policy, {"test": env}, eval_splits=["test"])
