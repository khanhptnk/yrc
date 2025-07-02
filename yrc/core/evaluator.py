import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import yrc
from yrc.core.environment import CoordEnv


@dataclass
class EvaluatorConfig:
    """
    Configuration for the Evaluator.

    Parameters
    ----------
    num_episodes : int, optional
        Number of episodes to use for evaluation. Default is 256.
    max_num_steps : int, optional
        Maximum number of steps per episode. Default is 256.
    temperature : float, optional
        Temperature parameter for action selection. Default is 1.0.
    log_action_id : int, optional
        The action index to track and log during evaluation. Default is CoordEnv.EXPERT.

    Examples
    --------
    >>> config = EvaluatorConfig(num_episodes=100, temperature=0.5)
    """

    num_episodes: int = 256
    max_num_steps: int = 256
    temperature: float = 1.0
    log_action_id: int = CoordEnv.EXPERT


class Evaluator:
    """
    Evaluator for running policy evaluation on environments and summarizing results.

    Examples
    --------
    >>> evaluator = Evaluator(EvaluatorConfig(), env)
    >>> summary = evaluator.evaluate(policy)
    """

    config_cls = EvaluatorConfig

    def __init__(self, config: EvaluatorConfig, env: "gym.Env") -> None:
        """
        Initialize the Evaluator.

        Parameters
        ----------
        config : EvaluatorConfig
            Configuration object for the evaluator.
        env : gym.Env
            The environment instance to evaluate on.

        Returns
        -------
        None
        """
        self.config = config
        self.env = env

    def evaluate(
        self,
        policy: "yrc.core.Policy",
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a policy on the environment and summarize the results.

        Parameters
        ----------
        policy : yrc.core.Policy
            The policy to evaluate. Must implement an `act` method and have a `.model` attribute.
        num_episodes : int, optional
            Number of episodes to run. If None, uses value from config.

        Returns
        -------
        dict
            A dictionary mapping split names to summary statistics for each evaluation.

        Examples
        --------
        >>> summary = evaluator.evaluate(policy, num_episodes=100)
        >>> print(summary['reward_mean'])
        """
        config = self.config
        env = self.env

        if num_episodes is None:
            num_episodes = config.num_episodes

        assert (
            num_episodes % env.num_envs == 0
        ), "Number of episodes must be divisible by the number of environments in each split."

        policy.eval()

        num_iterations = num_episodes // env.num_envs

        self.summarizer = EvaluationSummarizer(config)

        for _ in range(num_iterations):
            self._eval_one_iteration(policy, env)

        summary = self.summarizer.write()

        return summary

    def _eval_one_iteration(self, policy: "yrc.core.Policy", env: "gym.Env") -> None:
        """
        Run a single evaluation iteration for the policy on the environment.

        Parameters
        ----------
        policy : yrc.core.Policy
            The policy to evaluate.
        env : gym.Env
            The environment instance to evaluate on.

        Returns
        -------
        None
        """
        self.summarizer.initialize_episode(env)

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)

        policy.reset(np.ones_like(has_done))

        for _ in range(self.config.max_num_steps):
            action = policy.act(obs, temperature=self.config.temperature)
            obs, reward, done, info = env.step(action.cpu().numpy())
            # NOTE: put this before update has_done to include last step in summary
            self.summarizer.add_episode_step(env, action, reward, info, has_done)
            has_done |= done

            if has_done.all():
                break

        self.summarizer.finalize_episode()


class EvaluationSummarizer:
    """
    Summarizer for evaluation statistics and logging.

    Examples
    --------
    >>> summarizer = EvaluationSummarizer(EvaluatorConfig())
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        """
        Initialize the EvaluationSummarizer.

        Parameters
        ----------
        config : EvaluatorConfig
            Configuration object for the summarizer.

        Returns
        -------
        None
        """
        self.log_action_id = config.log_action_id
        self.clear()

    def clear(self) -> None:
        """
        Clear the summary statistics log.

        Returns
        -------
        None
        """
        self.log = {}

    def initialize_episode(self, env: "gym.Env") -> None:
        """
        Initialize logging for a new evaluation episode.

        Parameters
        ----------
        env : gym.Env
            The environment instance for the episode.

        Returns
        -------
        None
        """
        self.episode_log = {
            "reward": [0] * env.num_envs,
            "base_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.log_action_id}": 0,
        }

    def finalize_episode(self) -> None:
        """
        Finalize and aggregate statistics for the episode.

        Returns
        -------
        None
        """
        if self.log:
            for k, v in self.episode_log.items():
                if isinstance(v, list):
                    self.log[k].extend(v)
                else:
                    self.log[k] += v
        else:
            self.log.update(self.episode_log)

    def add_episode_step(
        self,
        env: "gym.Env",
        action: "torch.Tensor",
        reward: np.ndarray,
        info: List[Dict[str, Any]],
        has_done: np.ndarray,
    ) -> None:
        """
        Log statistics for each episode step.

        Parameters
        ----------
        env : gym.Env
            The environment instance.
        action : torch.Tensor
            Actions taken at this step.
        reward : np.ndarray
            Rewards received at this step.
        info : list of dict
            Additional info for each environment.
        has_done : np.ndarray
            Boolean array indicating which episodes are done.

        Returns
        -------
        None
        """
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

    def summarize(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the current log.

        Returns
        -------
        dict
            Dictionary of summary statistics.

        Examples
        --------
        >>> summary = summarizer.summarize()
        """
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

    def write(self, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pretty-print and log the summary statistics.

        Parameters
        ----------
        summary : dict, optional
            Precomputed summary statistics. If None, will compute from log.

        Returns
        -------
        dict
            The summary statistics that were logged.

        Examples
        --------
        >>> logged_summary = summarizer.write()
        """
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
