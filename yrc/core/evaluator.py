import logging
from dataclasses import dataclass
from typing import Dict, List

import gym
import numpy as np

import yrc


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
    """

    validation_episodes: int = 256
    test_episodes: int = 256
    act_greedy: bool = False


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
        policy.model.eval()

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

            summary[split] = self.summarizer.summarize()
            self.summarizer.write()

            envs[split].close()

        return summary

    def _eval_one_iteration(self, policy, env):
        self.summarizer.reset_episode(env)

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)

        while not has_done.all():
            action = policy.act(obs, greedy=self.config.act_greedy)

            obs, reward, done, info = env.step(action)
            # NOTE: put this before update has_done to include last step in summary
            self.summarizer.add_to_episode(
                env, action, obs, reward, done, info, has_done
            )

            has_done |= done

        self.summarizer.finalize_episode()


class EvaluationSummarizer:
    def __init__(self, config):
        self.logged_action = 1

    def clear(self):
        self.log = {}

    def reset_episode(self, env):
        self.episode_log = {
            "reward": [0] * env.num_envs,
            "env_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.logged_action}": 0,
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

    def add_to_episode(self, env, action, obs, reward, done, info, has_done):
        for i in range(env.num_envs):
            if "env_reward" in info[i]:
                self.episode_log["env_reward"][i] += info[i]["env_reward"] * (
                    1 - has_done[i]
                )

            self.episode_log["reward"][i] += reward[i] * (1 - has_done[i])
            self.episode_log["episode_length"][i] += 1 - has_done[i]
            if not has_done[i]:
                self.episode_log[f"action_{self.logged_action}"] += (
                    action[i] == self.logged_action
                ).sum()

    def summarize(self):
        log = self.log
        self.summary = {
            "steps": int(sum(log["episode_length"])),
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "reward_mean": float(np.mean(log["reward"])),
            "raw_reward": log["reward"],
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            f"action_{self.logged_action}_frac": float(
                log[f"action_{self.logged_action}"] / sum(log["episode_length"])
            ),
        }
        return self.summary

    def write(self, summary=None):
        if summary is None:
            summary = self.summary

        log_str = f"   Steps:       {summary['steps']}\n"
        log_str += "   Episode:    "
        log_str += f"mean {summary['episode_length_mean']:7.2f}  "
        log_str += f"min {summary['episode_length_min']:7.2f}  "
        log_str += f"max {summary['episode_length_max']:7.2f}\n"
        log_str += "   Reward:     "
        log_str += f"mean {summary['reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['env_reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += f"   Action {self.logged_action} fraction: {summary[f'action_{self.logged_action}_frac']:7.2f}\n"

        logging.info(log_str)
