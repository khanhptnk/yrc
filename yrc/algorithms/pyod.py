import logging
import pprint
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable
from yrc.utils.logging import configure_logging


@dataclass
class PyODAlgorithmConfig:
    """
    Configuration dataclass for PyODAlgorithm.

    Parameters
    ----------
    cls : str, optional
        Name of the algorithm class. Default is "PyODAlgorithm".
    num_rollouts : int, optional
        Number of rollouts to use for data generation. Default is 128.
    percentiles : list of float, optional
        List of percentiles to use for threshold selection. Default is range(0, 101, 10).
    explore_temps : list of float, optional
        List of temperatures to use during exploration rollouts. Default is [1.0].
    accept_rate : float, optional
        Acceptance rate for sampling data during rollouts. Default is 0.05.
    """

    name: str = "pyod"
    num_rollouts: int = 128
    percentiles: List[float] = field(default_factory=lambda: list(range(0, 101, 10)))
    explore_temps: List[float] = field(default_factory=lambda: [1.0])
    accept_rate: float = 0.05


class PyODAlgorithm(Algorithm):
    """
    Algorithm for out-of-distribution (OOD) detection using PyOD models.

    Examples
    --------
    >>> algo = PyODAlgorithm(PyODAlgorithmConfig())
    """

    config_cls = PyODAlgorithmConfig

    def __init__(self, config: PyODAlgorithmConfig) -> None:
        """
        Initialize the PyODAlgorithm.

        Parameters
        ----------
        config : PyODAlgorithmConfig
            Configuration object for the PyODAlgorithm.

        Returns
        -------
        None

        Examples
        --------
        >>> algo = PyODAlgorithm(PyODAlgorithmConfig())
        """
        self.config = config
        self.random = random.Random(get_global_variable("seed") + 543)

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        env: "gym.Env",
        validators: Dict[str, "yrc.core.Evaluator"],
    ) -> None:
        """
        Train the PyODAlgorithm by searching for the best threshold parameter
        that maximizes evaluation reward.

        Parameters
        ----------
        policy : yrc.policies.PPOPolicy
            The policy to be evaluated and tuned.
        env : gym.Env
            The environment instance for training and data generation.
        validators : dict of str to yrc.core.Evaluator
            Dictionary mapping split names to evaluator instances for evaluation.

        Returns
        -------
        None

        Examples
        --------
        >>> algorithm = PyODAlgorithm(PyODAlgorithmConfig())
        >>> algorithm.train(policy, env, validators)
        """
        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_threshold = {}
        best_result = {}
        for split in validators:
            best_result[split] = {"reward_mean": -float("inf")}

        for explore_temp in config.explore_temps:
            logging.info(f"Exploration temperature: {explore_temp}")
            data = self._generate_data(
                env.base_env,
                env.novice,
                explore_temp,
                config.num_rollouts,
                config.accept_rate,
            )

            # Train OOD detector
            policy.fit(data)
            # NOTE: weird bug, pyod messes up logging so reconfigure logging
            configure_logging(get_global_variable("log_file"))

            # Threshold search
            scores = policy.get_train_scores()
            thresholds = [np.percentile(scores, pct) for pct in config.percentiles]
            logging.info("Thresholds: " + pprint.pformat(thresholds, indent=2))

            for threshold in thresholds:
                policy.set_params({"threshold": threshold})

                logging.info("Threshold: " + str(threshold))

                eval_result = {}
                for split, validator in validators.items():
                    logging.info(f"Evaluating on {split} split")
                    eval_result[split] = validator.evaluate(policy)
                    if (
                        eval_result[split]["reward_mean"]
                        > best_result[split]["reward_mean"]
                    ):
                        best_threshold[split] = threshold
                        best_result[split] = eval_result[split]
                        self.save_checkpoint(policy, f"best_{split}")

                # Log best result so far
                for split, validator in validators.items():
                    logging.info(f"BEST {split} so far")
                    logging.info(
                        "Parameters: " + pprint.pformat(best_threshold[split], indent=2)
                    )
                    validator.summarizer.write(best_result[split])

    def save_checkpoint(self, policy: "yrc.policies.PPOPolicy", name: str) -> None:
        """
        Save the current policy configuration and parameters to a checkpoint file.

        Parameters
        ----------
        policy : yrc.policies.PPOPolicy
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

    def _generate_data(
        self,
        env: "gym.Env",
        policy: "yrc.policies.PPOPolicy",
        temperature: float,
        num_rollouts: int,
        accept_rate: float,
    ) -> dict:
        """
        Generate data for OOD detection by rolling out the policy in the environment.

        Parameters
        ----------
        env : gym.Env
            The environment used for rollouts.
        policy : yrc.policies.PPOPolicy
            The policy to be evaluated.
        temperature : float
            Temperature parameter for action selection.
        num_rollouts : int
            Total number of rollout episodes to generate.
        accept_rate : float
            Acceptance rate for sampling data during rollouts.

        Returns
        -------
        data : dict
            Dictionary containing collected data arrays for each feature.

        Examples
        --------
        >>> data = self._generate_data(env, policy, 1.0, 128, 0.05)
        """

        @torch.no_grad()
        def rollout_once():
            policy.eval()
            obs = env.reset()
            has_done = np.array([False] * env.num_envs)

            while not has_done.all():
                action, model_output = policy.act(
                    obs, temperature=temperature, return_model_output=True
                )

                for i in range(env.num_envs):
                    if not has_done[i] and self.random.random() < accept_rate:
                        new_data = {
                            "base_obs": obs[i],
                            "novice_hidden": model_output.hidden[i].cpu().numpy(),
                            "novice_logits": model_output.logits[i].cpu().numpy(),
                        }
                        for k, v in new_data.items():
                            if k not in data:
                                data[k] = []
                            data[k].append(v)

                obs, _, done, _ = env.step(action.cpu().numpy())
                has_done |= done

        assert (
            num_rollouts % env.num_envs == 0
        ), "LogitAlgorithm requires num_rollouts to be divisible by num_envs"

        data = {}

        for i in range(num_rollouts // env.num_envs):
            rollout_once()

        for k in data:
            data[k] = np.stack(data[k])

        return data
