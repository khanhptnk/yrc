import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable


@dataclass
class RandomAlgorithmConfig:
    """
    Configuration dataclass for RandomAlgorithm.

    Parameters
    ----------
    cls : str, optional
        Name of the algorithm class. Default is "RandomAlgorithm".
    probs : list of float, optional
        List of probabilities to search. Default is np.arange(0, 1.01, 0.1).
    """

    cls: str = "RandomAlgorithm"
    probs: List[float] = field(default_factory=lambda: np.arange(0, 1.01, 0.1).tolist())


class RandomAlgorithm(Algorithm):
    def __init__(self, config):
        self.config = config

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        env: "gym.Env",
        validators: Dict[str, "yrc.core.Evaluator"],
    ):
        """
        Train the RandomAlgorithm by searching for the best probability parameter
        that maximizes evaluation reward.

        Parameters
        ----------
        policy : yrc.policies.PPOPolicy
            The policy to be evaluated and tuned.
        envs : dict of str to gym.Env
            Dictionary mapping split names to environment instances.
        evaluator : yrc.core.Evaluator
            Evaluator object for policy evaluation and summary logging.
        train_split : str, optional
            The environment split to use for training. Default is "train".
        eval_splits : list of str, optional
            List of environment splits to use for evaluation. Default is ["val_sim, val_true"].

        Returns
        -------
        None

        Examples
        --------
        >>> algorithm = RandomAlgorithm(RandomAlgorithmConfig())
        >>> algorithm.train(policy, envs, evaluator)
        """
        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_prob = {}
        best_result = {}
        for split in validators:
            best_result[split] = {"reward_mean": -float("inf")}

        for prob in config.probs:
            logging.info(f"Prob: {prob}")

            policy.set_params({"prob": prob})

            eval_result = {}
            for split, validator in validators.items():
                logging.info(f"Evaluating on {split} split")
                eval_result[split] = validator.evaluate(policy)
                if (
                    eval_result[split]["reward_mean"]
                    > best_result[split]["reward_mean"]
                ):
                    best_prob[split] = prob
                    best_result[split] = eval_result[split]
                    self.save_checkpoint(policy, f"best_{split}")

            # Log best result so far
            for split, validator in validators.items():
                logging.info(f"BEST {split} so far")
                logging.info(f"Prob: {best_prob[split]}")
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
