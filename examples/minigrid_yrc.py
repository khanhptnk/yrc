import argparse
import logging
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
    train: Optional[str] = "DistShift2-v0"
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
        choices=["train", "eval", "visualize"],
        help="Mode to run",
    )
    parser.add_argument(
        "--type", type=str, choices=["agent", "coord"], help="Policy type"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    args, dotlist_args = parser.parse_known_args()
    config = yrc.make_config(args, dotlist_args)
    return args, config


def make_base_env(config, split, render_mode="rgb_array"):
    env_id = f"MiniGrid-{getattr(config, split)}"

    # env_fn should be a callable that returns a new environment instance
    def env_fn(env_id=env_id, render_mode=render_mode):
        return ImgObsWrapper(gym.make(env_id, render_mode=render_mode))

    return make_vec_env(env_fn, n_envs=config.num_envs, seed=config.seed)


def make_base_envs(config):
    base_envs = {}
    for split in splits:
        base_envs[split] = make_base_env(config.env, split)
    return base_envs


def make_coord_envs(base_envs, config):
    novice = yrc.load_policy(config.train_novice, base_envs["train"])
    expert = yrc.load_policy(config.train_expert, base_envs["train"])

    envs = {}
    for split in splits:
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )
        envs[split].set_costs(0.05)
    logging.info(
        f"Expert query cost per action: {envs['train'].expert_query_cost_per_action}"
    )
    return envs


def train(args, config):
    base_envs = make_base_envs(config)
    envs = make_coord_envs(base_envs, config) if args.type == "coord" else base_envs
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


def evaluate(args, config):
    base_envs = make_base_envs(config)
    envs = make_coord_envs(base_envs, config) if args.type == "coord" else base_envs
    if config.policy.load_path is None:
        logging.info("WARNING: No policy load path specified, using default policy.")
        policy = yrc.make_policy(config.policy, envs[splits[0]])
    else:
        policy = yrc.load_policy(config.policy.load_path, envs["train"])
    for split in splits:
        logging.info(f"Evaluating on {split} split")
        yrc.Evaluator(config.evaluation, envs[split]).evaluate(policy)


def visualize(args, config):
    base_env = make_base_env(config.env, "test_hard", render_mode="human")
    if args.type == "coord":
        novice = yrc.load_policy(config.train_novice, base_env)
        expert = yrc.load_policy(config.train_expert, base_env)
        env = yrc.CoordEnv(config.coordination, base_env, novice, expert)
        env.set_costs(0.05)
    else:
        env = base_env
    policy = yrc.load_policy(config.policy.load_path, env)

    for _ in range(100):
        policy.reset([True] * config.env.num_envs)
        has_done = False
        obs = env.reset()
        total_reward = 0
        for i in range(config.evaluation.num_steps):
            action = policy.act(obs)
            obs, reward, done, info = env.step(action.cpu().numpy())
            has_done |= done
            total_reward += reward
            if has_done:
                break
        print(total_reward)


def main():
    # NOTE: register the MiniGrid configuration with YRC
    # This should be done before parsing args to ensure the config is available
    yrc.register_environment(MiniGridConfig.name, MiniGridConfig)
    yrc.register_model("minigrid_ppo", MiniGridPPOModel)

    args, config = parse_args()
    if args.mode == "train":
        train(args, config)
    elif args.mode == "eval":
        evaluate(args, config)
    elif args.mode == "visualize":
        visualize(args, config)


if __name__ == "__main__":
    main()
