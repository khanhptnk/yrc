import logging
import pprint
from dataclasses import dataclass

import numpy as np

from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable


@dataclass
class LogitAlgorithmConfig:
    min_pct: int = 0
    max_pct: int = 101
    pct_step: int = 10
    num_rollouts: int = 64


class LogitAlgorithm(Algorithm):
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

        best_params = {}
        scores = policy.generate_scores(envs["train"], config.num_rollouts)
        cand_thresholds = [
            np.percentile(scores, pct)
            for pct in range(config.min_pct, config.max_pct, config.pct_step)
        ]

        logging.info("Candidate thresholds: " + str(cand_thresholds))

        for explore_temp in [1]:
            for threshold in cand_thresholds:
                params = {
                    "threshold": threshold,
                    "explore_temp": explore_temp,
                    "score_temp": 1,
                }

                logging.info("Parameters: " + pprint.pformat(params, indent=2))

                policy.update_params(params)
                split_summary = evaluator.eval(policy, envs, eval_splits)

                for split in eval_splits:
                    if (
                        split_summary[split]["reward_mean"]
                        > best_summary[split]["reward_mean"]
                    ):
                        best_params[split] = params
                        best_summary[split] = split_summary[split]
                        policy.save_model(f"best_{split}", save_dir)

                    # log best result so far
                    logging.info(f"Best {split} so far")
                    logging.info(
                        "Parameters: " + pprint.pformat(best_params[split], indent=2)
                    )
                    evaluator.write_summary(f"best_{split}", best_summary[split])

        policy.update_params(best_params)
