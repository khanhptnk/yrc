import argparse
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.env_util import make_vec_env

import yrc


@dataclass
class MiniGridConfig:
    suite: str = "MiniGrid"
    num_envs: int = 8
    seed: int = 0
    train: Optional[str] = None
    test_easy: Optional[str] = "DistShift1-v0"
    test_hard: Optional[str] = "DistShift2-v0"


config_dict = {
    "novice": MiniGridConfig(train="DistShift1-v0"),
    "expert": MiniGridConfig(train="DistShift2-v0"),
    "coord": MiniGridConfig(train="DistShift2-v0"),
}

splits = ["train", "test_easy", "test_hard"]


def parse_args():
    parser = argparse.ArgumentParser(description="YRC training script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "novice",
            "expert",
            "coord",
        ],
        help="Mode to run",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)
    return args, config


def make_base_env(split, config):
    env_id = f"{config.suite}-{getattr(config, split)}"

    # env_fn should be a callable that returns a new environment instance
    def env_fn(env_id=env_id):
        return ImgObsWrapper(gym.make(env_id))

    env = make_vec_env(env_fn, n_envs=config.num_envs, seed=config.seed)
    return env


def train_agent(config):
    envs = {}
    for split in splits:
        envs[split] = make_base_env(split, config.env)
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def train_coord(config):
    base_envs = {}
    for split in splits:
        base_envs[split] = make_base_env(split, config)

    novice = yrc.load_policy(config.train_novice, base_envs["train"])
    expert = yrc.load_policy(config.train_expert, base_envs["train"])

    envs = {}
    for split in splits:
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )
        envs[split].set_costs(0.1)

    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def main():
    yrc.register_env_config("minigrid", MiniGridConfig)

    args, config = parse_args()
    if args.mode == "coord":
        config.env = config_dict["coord"]
        train_coord(config)
    else:
        if args.mode == "novice":
            config.env = config_dict["novice"]
        else:
            config.env = config_dict["expert"]
        train_agent(config)


if __name__ == "__main__":
    main()
