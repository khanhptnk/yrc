import argparse
from dataclasses import dataclass, field

import yrc

splits = ["train", "val_sim", "val_true", "test"]


@dataclass
class MiniGridDistributionConfig:
    name: str = "S9N1-v0"


@dataclass
class MiniGridConfig:
    suite: str = "MiniGrid"
    name: str = "LavaCrossing"
    num_envs: int = 32
    seed: int = 0
    train: MiniGridDistributionConfig = field(
        default_factory=lambda: MiniGridDistributionConfig(name="S9N1-v0")
    )
    val_sim: MiniGridDistributionConfig = field(
        default_factory=lambda: MiniGridDistributionConfig(name="S9N1-v0")
    )
    val_true: MiniGridDistributionConfig = field(
        default_factory=lambda: MiniGridDistributionConfig(name="S9N2-v0")
    )
    test: MiniGridDistributionConfig = field(
        default_factory=lambda: MiniGridDistributionConfig(name="S9N2-v0")
    )


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
            "train_agent",
            "evaluate_agent",
            "train_coord_policy",
            "evaluate_coord_policy",
        ],
        help="Mode to run",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)
    return args, config


def minigrid_to_sb3_env(env_fn, n_envs=1, seed=0, **kwargs):
    """
    Wrap a Gym environment factory for use with Stable Baselines3.
    Each vectorized environment will be independent.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make_env(rank):
        def _init():
            env = env_fn()
            env.seed(seed + rank)
            return env

        return _init

    env_fns = [make_env(i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)


def make_base_envs(config):
    envs = {}
    for split in ["train", "val_sim", "val_true", "test"]:
        env_id = f"{config.suite}-{config.name}{getattr(config, split).name}"

        # env_fn should be a callable that returns a new environment instance
        def env_fn(env_id=env_id):
            import gym

            return gym.make(env_id)

        envs[split] = minigrid_to_sb3_env(
            env_fn, n_envs=config.num_envs, seed=config.seed
        )
    return envs


def train_agent(config):
    envs = make_base_envs(config.env)
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split == "train":
            continue
        validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def main():
    yrc.register_env_config("minigrid", MiniGridConfig)

    args, config = parse_args()
    globals()[args.mode](config)


if __name__ == "__main__":
    main()
