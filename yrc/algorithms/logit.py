import logging
import pprint
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable


@dataclass
class LogitAlgorithmConfig:
    """
    Configuration for the LogitAlgorithm, which tunes thresholds and temperatures for confidence-based policies.

    Parameters
    ----------
    name : str, optional
        Name of the algorithm class. Default is "logit".
    num_rollouts : int, optional
        Number of rollouts to use for score generation. Default is 128.
    percentiles : list of float, optional
        List of percentiles to use for threshold selection. Default is range(0, 101, 10).
    explore_temps : list of float, optional
        List of temperatures to use during exploration rollouts. Default is [1.0].
    score_temps : list of float, optional
        List of temperatures to use when scoring. Default is [1.0].

    Examples
    --------
    >>> config = LogitAlgorithmConfig()
    """

    name: str = "logit"
    num_rollouts: int = 128
    percentiles: List[float] = field(default_factory=lambda: list(range(0, 101, 10)))
    explore_temps: List[float] = field(default_factory=lambda: [1.0])
    score_temps: List[float] = field(default_factory=lambda: [1.0])


class LogitAlgorithm(Algorithm):
    """
    Algorithm for tuning confidence-based policies using logit thresholds and temperatures.

    Examples
    --------
    >>> algo = LogitAlgorithm(LogitAlgorithmConfig())
    """

    config_cls = LogitAlgorithmConfig

    def __init__(self, config: LogitAlgorithmConfig) -> None:
        """
        Initialize the LogitAlgorithm.

        Parameters
        ----------
        config : LogitAlgorithmConfig
            Configuration object for the LogitAlgorithm.

        Returns
        -------
        None

        Examples
        --------
        >>> algo = LogitAlgorithm(LogitAlgorithmConfig())
        """
        self.config = config

    def train(
        self,
        policy: "yrc.core.Policy",
        env: "gym.Env",
        validators: Dict[str, "yrc.core.Evaluator"],
    ) -> None:
        """
        Train the LogitAlgorithm by searching for the best threshold and temperature parameters
        based on rollout scores and evaluation results.

        Parameters
        ----------
        policy : yrc.core.Policy
            The policy to be trained and evaluated.
        env : gym.Env
            The environment used for training and rollouts.
        validators : dict of str to yrc.core.Evaluator
            Dictionary mapping split names to evaluator instances for evaluation.

        Returns
        -------
        None

        Examples
        --------
        >>> algorithm = LogitAlgorithm(LogitAlgorithmConfig())
        >>> algorithm.train(policy, env, validators)
        """
        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_params = {}
        best_result = {}
        for split in validators:
            best_result[split] = {"reward_mean": -float("inf")}

        self.score_fn = policy.compute_confidence

        for explore_temp in config.explore_temps:
            logging.info(f"Exploration temperature: {explore_temp}")
            for score_temp in config.score_temps:
                policy.set_params({"temperature": score_temp})
                # Generate scores by rolling out (simulated) novice in training environment
                scores = self._generate_scores(
                    env.base_env,
                    env.novice,
                    explore_temp,
                    config.num_rollouts,
                )
                thresholds = [np.percentile(scores, pct) for pct in config.percentiles]
                logging.info("Thresholds: " + pprint.pformat(thresholds, indent=2))
                for threshold in thresholds:
                    policy.set_params({"threshold": threshold})

                    cur_params = policy.get_params()
                    logging.info("Parameters: " + pprint.pformat(cur_params, indent=2))

                    # Evaluate policy on all splits
                    eval_result = {}
                    for split, validator in validators.items():
                        logging.info(f"Evaluating on {split} split")
                        eval_result[split] = validator.evaluate(policy)
                        if (
                            eval_result[split]["reward_mean"]
                            > best_result[split]["reward_mean"]
                        ):
                            best_params[split] = cur_params
                            best_result[split] = eval_result[split]
                            self.save_checkpoint(policy, f"best_{split}")

                    # Log best result so far
                    for split, validator in validators.items():
                        logging.info(f"BEST {split} so far")
                        logging.info(
                            "Parameters: "
                            + pprint.pformat(best_params[split], indent=2)
                        )
                        validator.summarizer.write(best_result[split])

    def save_checkpoint(self, policy: "yrc.core.Policy", name: str) -> None:
        """
        Save the current policy configuration and parameters to a checkpoint file.

        Parameters
        ----------
        policy : yrc.core.Policy
            The policy whose parameters are to be saved.
        name : str
            Name for the checkpoint file.

        Returns
        -------
        None

        Examples
        --------
        >>> self.save_checkpoint(policy, "best_test")
        """
        save_path = f"{self.save_dir}/{name}.ckpt"
        torch.save(
            {
                "policy_config": policy.config,
                "model_state_dict": policy.get_params(),
            },
            save_path,
        )
        logging.info(f"Saved checkpoint to {save_path}")

    def _generate_scores(
        self,
        env: "gym.Env",
        policy: "yrc.core.Policy",
        temperature: float,
        num_rollouts: int,
    ) -> list:
        """
        Generate confidence scores by rolling out the policy in the environment.

        Parameters
        ----------
        env : gym.Env
            The environment used for rollouts.
        policy : yrc.core.Policy
            The policy to be evaluated.
        temperature : float
            Temperature parameter for action selection.
        num_rollouts : int
            Total number of rollout episodes to generate.

        Returns
        -------
        scores : list of float
            List of confidence scores collected from rollouts.

        Examples
        --------
        >>> scores = self._generate_scores(env, policy, 1.0, 128)
        """

        @torch.no_grad()
        def rollout_once():
            policy.eval()
            obs = env.reset()
            has_done = np.array([False] * env.num_envs)
            policy.reset(np.ones_like(has_done))

            while not has_done.all():
                action, model_output = policy.act(
                    obs, temperature=temperature, return_model_output=True
                )

                score = self.score_fn(model_output.logits)

                for i in range(env.num_envs):
                    if not has_done[i]:
                        scores.append(score[i].item())

                obs, _, done, _ = env.step(action.cpu().numpy())
                has_done |= done

            return scores

        assert (
            num_rollouts % env.num_envs == 0
        ), "LogitAlgorithm requires num_rollouts to be divisible by num_envs"
        scores = []
        for i in range(num_rollouts // env.num_envs):
            rollout_once()
        return scores
