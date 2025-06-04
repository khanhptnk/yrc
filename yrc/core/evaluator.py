import logging
from dataclasses import dataclass
from typing import Dict, List

import gym
import numpy as np

import yrc
from yrc.core.environment import CoordEnv


@dataclass
class EvaluatorConfig:
    """
    Configuration for the Evaluator.

    Attributes
    ----------
    validation_episodes : int
        Number of episodes to use for validation evaluation. Default is 256.
    test_episodes : int
        Number of episodes to use for test evaluation. Default is 256.
    act_greedy : bool
        If True, the policy acts greedily during evaluation. Default is False.
    log_action_id: bool
        The action index to track and log the fraction of times this action is taken during evaluation episodes.
    """

    validation_episodes: int = 256
    test_episodes: int = 256
    temperature: float = 1.0
    log_action_id: int = CoordEnv.EXPERT


class Evaluator:
    def __init__(self, config):
        self.config = config

    def eval(
        self,
        policy: "yrc.core.Policy",
        envs: Dict[str, "gym.Env"],
        eval_splits: List[str],
        num_episodes: int = None,
    ) -> dict:
        """
        Evaluate a policy on multiple environment splits and summarize the results.

        For each split in `eval_splits`, this method runs evaluation episodes using the provided
        policy and environment, collects statistics, and returns a summary dictionary for each split.

        Parameters
        ----------
        policy : yrc.core.Policy
            The policy to evaluate. Must implement an `act` method and have a `.model` attribute.
        envs : Dict[str, gym.Env]
            A dictionary mapping split names to environment instances.
        eval_splits : List[str]
            List of split names (keys in `envs`) to evaluate.
        num_episodes : int, optional
            Number of episodes to run per split. If None, uses values from config.

        Returns
        -------
        summary : dict
            A dictionary mapping split names to summary statistics for each evaluation.

        Examples
        --------
        >>> summary = evaluator.eval(policy, envs, ['val', 'test'], num_episodes=100)
        >>> print(summary['val']['reward_mean'])
        """
        config = self.config
        policy.eval()

        self.summarizer = EvaluationSummarizer(config)
        summary = {}

        for split in eval_splits:
            if num_episodes is None:
                if "val" in split:
                    num_episodes = config.validation_episodes
                else:
                    assert "test" in split
                    num_episodes = config.test_episodes
                assert num_episodes % envs[split].num_envs == 0

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            num_iterations = num_episodes // envs[split].num_envs

            self.summarizer.clear()

            for _ in range(num_iterations):
                self._eval_one_iteration(policy, envs[split])

            summary[split] = self.summarizer.write()

            envs[split].close()

        return summary

    def _eval_one_iteration(self, policy, env):
        self.summarizer.initialize_episode(env)

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)

        while not has_done.all():
            action = policy.act(obs, temperature=self.config.temperature)
            obs, reward, done, info = env.step(action.cpu().numpy())
            # NOTE: put this before update has_done to include last step in summary
            self.summarizer.add_episode_step(env, action, reward, info, has_done)
            has_done |= done

        self.summarizer.finalize_episode()


class EvaluationSummarizer:
    def __init__(self, config):
        self.log_action_id = config.log_action_id
        self.clear()

    def clear(self):
        self.log = {}

    def initialize_episode(self, env):
        self.episode_log = {
            "reward": [0] * env.num_envs,
            "base_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.log_action_id}": 0,
        }

    def finalize_episode(self):
        if self.log:
            for k, v in self.episode_log.items():
                if isinstance(v, list):
                    self.log[k].extend(v)
                else:
                    self.log[k] += v
        else:
            self.log.update(self.episode_log)

    def add_episode_step(self, env, action, reward, info, has_done):
        for i in range(env.num_envs):
            if "base_reward" in info[i]:
                self.episode_log["base_reward"][i] += info[i]["base_reward"] * (
                    1 - has_done[i]
                )

            self.episode_log["reward"][i] += reward[i] * (1 - has_done[i])
            self.episode_log["episode_length"][i] += 1 - has_done[i]
            if not has_done[i]:
                self.episode_log[f"action_{self.log_action_id}"] += (
                    action[i] == self.log_action_id
                ).sum()

    def summarize(self):
        log = self.log
        self.summary = {
            "steps": int(sum(log["episode_length"])),
            "all_rewards": log["reward"],
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "reward_mean": float(np.mean(log["reward"])),
            "reward_std": float(np.std(log["reward"])),
            "base_reward_mean": float(np.mean(log["base_reward"])),
            "base_reward_std": float(np.std(log["base_reward"])),
            f"action_{self.log_action_id}_frac": float(
                log[f"action_{self.log_action_id}"] / sum(log["episode_length"])
            ),
        }
        return self.summary

    def write(self, summary=None):
        if summary is None:
            summary = self.summarize()

        log_str = (
            f"   Steps:         {summary['steps']}\n"
            f"   Episode length: mean {summary['episode_length_mean']:7.2f}  "
            f"min {summary['episode_length_min']:7.2f}  "
            f"max {summary['episode_length_max']:7.2f}\n"
            f"   Reward:         mean {summary['reward_mean']:.2f} "
            f"± {(1.96 * summary['reward_std']) / (len(summary['all_rewards']) ** 0.5):.2f}\n"
            f"   Base Reward:    mean {summary['base_reward_mean']:.2f} "
            f"± {(1.96 * summary['base_reward_std']) / (len(summary['all_rewards']) ** 0.5):.2f}\n"
            f"   Action {self.log_action_id} fraction: {summary[f'action_{self.log_action_id}_frac']:7.2f}\n"
        )

        logging.info(log_str)
        return summary
