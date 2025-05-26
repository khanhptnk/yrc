import logging
from dataclasses import dataclass

import numpy as np

from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable


@dataclass
class RandomAlgorithmConfig:
    cls: str = "RandomAlgorithm"
    min_prob: int = 0
    max_prob: int = 1.01
    prob_step: int = 0.1


class RandomAlgorithm(Algorithm):
    def __init__(self, config, env):
        self.config = config

    def train(
        self,
        policy,
        envs,
        evaluator=None,
        train_split=None,
        eval_splits=None,
    ):
        config = self.config
        save_dir = get_global_variable("experiment_dir")

        best_summary = {}
        for split in eval_splits:
            best_summary[split] = {"reward_mean": -1e9}

        best_prob = {}
        cand_probs = list(np.arange(config.min_prob, config.max_prob, config.prob_step))

        logging.info("Candidate probs: " + str(cand_probs))

        for prob in cand_probs:
            logging.info(f"Prob: {prob}")

            policy.update_params(prob)
            split_summary = evaluator.eval(policy, envs, eval_splits)

            for split in eval_splits:
                if (
                    split_summary[split]["reward_mean"]
                    > best_summary[split]["reward_mean"]
                ):
                    best_prob[split] = prob
                    best_summary[split] = split_summary[split]
                    policy.save_model(f"best_{split}", save_dir)

                # log best result so far
                logging.info(f"Best {split} so far")
                logging.info(f"Prob: {best_prob[split]}")
                evaluator.write_summary(f"best_{split}", best_summary[split])

        policy.update_params(best_prob)
