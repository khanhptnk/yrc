import logging
import pprint
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable
from yrc.utils.logging import configure_logging


@dataclass
class PyODAlgorithmConfig:
    cls: str = "PyODAlgorithm"
    num_rollouts: int = 128
    percentiles: List[float] = field(default_factory=lambda: list(range(0, 101, 10)))
    explore_temps: List[float] = field(default_factory=lambda: [1.0])
    accept_rate: float = 0.05


class PyODAlgorithm(Algorithm):
    def __init__(self, config):
        self.config = config
        self.random = random.Random(get_global_variable("seed") + 543)

    def train(
        self,
        policy,
        envs,
        evaluator=None,
        train_split=None,
        eval_splits=None,
    ):
        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_threshold = {}
        best_result = {}
        for split in eval_splits:
            best_result[split] = {"reward_mean": -float("inf")}

        train_env = envs[train_split]

        for explore_temp in config.explore_temps:
            logging.info(f"Exploration temperature: {explore_temp}")
            data = self._generate_data(
                train_env.base_env,
                train_env.novice,
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

                eval_results = evaluator.eval(policy, envs, eval_splits)

                for split in eval_splits:
                    if (
                        eval_results[split]["reward_mean"]
                        > best_result[split]["reward_mean"]
                    ):
                        best_threshold[split] = threshold
                        best_result[split] = eval_results[split]
                        self.save_checkpoint(policy, f"best_{split}")

                    # log best result so far
                    logging.info(f"BEST {split} so far")
                    logging.info(
                        "Parameters: " + pprint.pformat(best_threshold[split], indent=2)
                    )
                    evaluator.summarizer.write(best_result[split])

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

    def _generate_data(self, env, policy, temperature, num_rollouts, accept_rate):
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
