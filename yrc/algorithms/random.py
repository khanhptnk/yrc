import logging
from dataclasses import dataclass
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
    min_prob : float, optional
        Minimum probability value to consider. Default is 0.
    max_prob : float, optional
        Maximum probability value to consider. Default is 1.01.
    prob_step : float, optional
        Step size for probability increments. Default is 0.1.
    """

    cls: str = "RandomAlgorithm"
    min_prob: float = 0
    max_prob: float = 1.01
    prob_step: float = 0.1
    probs = List[float] = list(np.arange(0, 1.01, 0.1))


class RandomAlgorithm(Algorithm):
    def __init__(self, config):
        self.config = config

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        envs: Dict[str, "gym.Env"],
        evaluator: "yrc.core.Evaluator",
        train_split: str = "train",
        eval_splits: List[str] = ["test"],
    ):
        """
        Train the AlwaysAlgorithm, which always returns the same action regardless of input.

        Parameters
        ----------
        policy : Policy
            The policy to use for generating actions.
        envs : dict
            Dictionary of environments keyed by split name.
        evaluator : Evaluator, optional
            Evaluator for evaluating the policy performance. Default is None.
        train_split : str, optional
            The training split to use. Default is "train".
        eval_splits : list, optional
            List of evaluation splits. Default is None.

        Returns
        -------
        None
        """

        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_prob = {}
        best_result = {}
        for split in eval_splits:
            best_result[split] = {"reward_mean": -1e9}

        for prob in config.probs:
            logging.info(f"Prob: {prob}")

            policy.set_probability(prob)
            eval_results = evaluator.eval(policy, envs, eval_splits)

            for split in eval_splits:
                if (
                    eval_results[split]["reward_mean"]
                    > best_result[split]["reward_mean"]
                ):
                    best_prob[split] = prob
                    best_result[split] = eval_results[split]
                    self.save_checkpoint(policy, f"best_{split}")

                # log best result so far
                logging.info(f"BEST {split} so far")
                logging.info(f"Prob: {best_prob[split]}")
                evaluator.summarizer.write(best_result[split])

    def save_checkpoint(self, policy, name):
        save_path = f"{self.save_dir}/{name}.ckpt"
        torch.save(
            {
                "policy_config": policy.config,
                "model_state_dict": {"prob": policy.get_probability()},
            },
            save_path,
        )
        logging.info(f"Saved checkpoint to {save_path}")
