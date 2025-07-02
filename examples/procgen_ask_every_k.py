import argparse
import logging
import os
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from dataclasses import dataclass, field

import numpy as np
import torch

import yrc
from environments.procgen import ProcgenConfig, make_env
from yrc.core import Algorithm, Policy
from yrc.utils.global_variables import get_global_variable

splits = ["train", "test"]


@dataclass
class AskEveryKAlgorithmConfig:
    name: str = "ask_every_k"
    candidates: List[int] = field(default_factory=lambda: [5, 10, 15, 20])


class AskEveryKAlgorithm(Algorithm):
    config_cls = AskEveryKAlgorithmConfig

    def __init__(self, config):
        self.config = config

    def train(self, policy, env, validators):
        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_k = None
        best_result = {}
        for split in validators:
            best_result[split] = {"reward_mean": -float("inf")}

        for k in config.candidates:
            logging.info(f"Evaluating k={k}")
            policy.set_params({"k": k})
            for split, validator in validators.items():
                result = validator.evaluate(policy)
                if result["reward_mean"] > best_result[split]["reward_mean"]:
                    best_result[split] = result
                    best_k = k
                    self.save_checkpoint(policy, f"best_{split}")

            for split, validator in validators.items():
                logging.info(f"BEST result for {split} (k={best_k}):")
                validator.summarizer.write(best_result[split])

    def save_checkpoint(self, policy, name):
        save_path = f"{self.save_dir}/{name}.ckpt"
        torch.save(
            {
                "policy_config": policy.config,
                "model_state_dict": policy.get_params(),
            },
            save_path,
        )
        logging.info(f"Saved checkpoint to {save_path}")


@dataclass
class AskEveryKPolicyConfig:
    name: str = "ask_every_k"
    load_path: Optional[str] = None


class AskEveryKPolicy(Policy):
    config_cls = AskEveryKPolicyConfig

    def __init__(self, config, env):
        self.config = config
        self.EXPERT = env.EXPERT
        self.k = None
        self.step = np.array([0] * env.num_envs)
        self.device = get_global_variable("device")

    def reset(self, done):
        self.batch_size = len(done)
        if self.batch_size < len(self.step):
            self.step = self.step[: self.batch_size]
        self.step[done] = 0

    def act(self, obs, temperature=None):
        batch_size = self.batch_size
        assert obs["base_obs"].shape[0] == batch_size
        action = torch.zeros(batch_size).long().to(self.device)
        for i in range(batch_size):
            if self.step[i] % self.k == 0:
                action[i] = self.EXPERT
            else:
                action[i] = 1 - self.EXPERT
            self.step[i] += 1
        return action

    def set_params(self, params):
        self.k = params["k"]

    def get_params(self):
        return {"k": self.k}

    def train(self):
        pass

    def eval(self):
        pass


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
        if split == "train":
            novice, expert = train_novice, train_expert
        else:
            novice, expert = test_novice, test_expert
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )

    # Set costs for the coordination environment
    base_penalty = compute_reward_per_action(config.env)
    for split in splits:
        envs[split].set_costs(base_penalty)

    return envs


def train(config):
    base_envs = make_base_envs(config)
    envs = make_coord_envs(config, base_envs)
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


if __name__ == "__main__":
    yrc.register_environment("procgen", ProcgenConfig)
    yrc.register_algorithm("ask_every_k", AskEveryKAlgorithm)
    yrc.register_policy("ask_every_k", AskEveryKPolicy)

    _, config = parse_args()
    train(config)
