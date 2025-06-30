import argparse
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.env_util import make_vec_env

import yrc
from yrc.models.ppo import PPOModelOutput
from yrc.utils.global_variables import get_global_variable


@dataclass
class MiniGridConfig:
    name: str = "minigrid"
    num_envs: int = 8
    seed: int = 0
    train: Optional[str] = None
    test_easy: Optional[str] = "DistShift1-v0"
    test_hard: Optional[str] = "DistShift2-v0"


splits = ["train", "test_easy", "test_hard"]


@dataclass
class MiniGridPPOModelConfig:
    name: str = "minigrid_ppo"


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class MiniGridPPOModel(nn.Module):
    config_cls = MiniGridPPOModelConfig

    def __init__(self, config, env):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (batch, 3, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, env.action_space.n)
        self.value_head = nn.Linear(128, 1)
        self.device = get_global_variable("device")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                orthogonal_init(m, gain=np.sqrt(2))
            if (
                isinstance(m, nn.Linear)
                and m is not self.policy_head
                and m is not self.value_head
            ):
                orthogonal_init(m, gain=np.sqrt(2))
        orthogonal_init(self.policy_head, gain=0.01)
        orthogonal_init(self.value_head, gain=1.0)

        # NOTE: model must have these attributes
        self.hidden_dim = 128
        self.logit_dim = env.action_space.n

    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs["base_obs"]
        x = (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        )  # (batch, 7, 7, 3)
        x = x.permute(0, 3, 1, 2)  # (batch, 3, 7, 7)
        x = self.conv(x)
        hidden = self.fc(x)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return PPOModelOutput(logits, value, hidden)


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

    dotlist_args.append(f"env.name={MiniGridConfig.name}")
    if args.mode == "novice":
        dotlist_args.append("env.train=DistShift1-v0")
    elif args.mode == "expert":
        dotlist_args.append("env.train=DistShift2-v0")
    elif args.mode == "coord":
        dotlist_args.append("env.train=DistShift2-v0")

    config = yrc.make_config(args, dotlist_args)
    return args, config


def make_base_env(split, config):
    env_id = f"MiniGrid-{getattr(config, split)}"

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
        base_envs[split] = make_base_env(split, config.env)

    novice = yrc.load_policy(config.train_novice, base_envs["train"])
    expert = yrc.load_policy(config.train_expert, base_envs["train"])

    envs = {}
    for split in splits:
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )
        envs[split].set_costs(0.05)

    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def main():
    # NOTE: register the MiniGrid configuration with YRC
    # This should be done before parsing args to ensure the config is available
    yrc.register_environment(MiniGridConfig.name, MiniGridConfig)
    yrc.register_model("minigrid_ppo", MiniGridPPOModel)

    args, config = parse_args()
    if args.mode == "coord":
        train_coord(config)
    else:
        train_agent(config)


if __name__ == "__main__":
    main()
