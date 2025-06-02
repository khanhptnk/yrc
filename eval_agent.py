import argparse

import yrc


def parse_args():
    parser = argparse.ArgumentParser(description="YRC evaluation script")
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


def main():
    args, config = parse_args()
    eval_splits = args.eval_splits
    # Create environments for each split
    envs = {split: yrc.make_base_env(split, config.env) for split in eval_splits}
    policy = yrc.load_policy(config.policy.load_path, envs[eval_splits[0]])
    evaluator = yrc.Evaluator(config.evaluation)
    evaluator.eval(policy, envs, eval_splits)


if __name__ == "__main__":
    main()
