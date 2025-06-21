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
    Configuration dataclass for LogitAlgorithm.

    Parameters
    ----------
    num_rollouts : int, optional
        Number of rollouts to use for score generation. Default is 128.
    percentiles : list of float, optional
        List of percentiles to use for threshold selection. Default is range(0, 101, 10).
    explore_temps : list of float, optional
        List of temperatures to use during exploration rollouts. Default is [1.0].
    score_temps : list of float, optional
        List of temperatures to use when scoring. Default is [1.0].
    """

    cls: str = "LogitAlgorithm"
    num_rollouts: int = 128
    percentiles: List[float] = field(default_factory=lambda: list(range(0, 101, 10)))
    explore_temps: List[float] = field(default_factory=lambda: [1.0])
    score_temps: List[float] = field(default_factory=lambda: [1.0])


class LogitAlgorithm(Algorithm):
    def __init__(self, config):
        self.config = config

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        envs: Dict[str, "gym.Env"],
        evaluator: "yrc.core.Evaluator",
        train_split: str = "train",
        eval_splits: List[str] = ["val_sim", "val_true"],
    ):
        """
        Train the LogitAlgorithm by searching for the best threshold and temperature parameters
        based on rollout scores and evaluation results.

        Parameters
        ----------
        policy : yrc.policies.PPOPolicy
            The policy to be trained and evaluated.
        envs : dict of str to gym.Env
            Dictionary mapping split names to environment instances.
        evaluator : yrc.core.Evaluator
            Evaluator object for policy evaluation and summary logging.
        train_split : str, optional
            The environment split to use for training. Default is "train".
        eval_splits : list of str, optional
            List of environment splits to use for evaluation. Default is ["val_sim", "val_true"].

        Returns
        -------
        None
        """
        config = self.config
        self.save_dir = get_global_variable("experiment_dir")

        best_params = {}
        best_result = {}
        for split in eval_splits:
            best_result[split] = {"reward_mean": -float("inf")}

        train_env = envs[train_split]

        self.score_fn = policy.compute_confidence

        for explore_temp in config.explore_temps:
            logging.info(f"Exploration temperature: {explore_temp}")
            for score_temp in config.score_temps:
                policy.set_params({"temperature": score_temp})
                # Generate scores by rolling out (simulated) novice in training environment
                scores = self._generate_scores(
                    train_env.base_env,
                    train_env.novice,
                    explore_temp,
                    config.num_rollouts,
                )
                thresholds = [np.percentile(scores, pct) for pct in config.percentiles]
                logging.info("Thresholds: " + pprint.pformat(thresholds, indent=2))
                for threshold in thresholds:
                    policy.set_params({"threshold": threshold})

                    cur_params = policy.get_params()
                    logging.info("Parameters: " + pprint.pformat(cur_params, indent=2))

                    eval_results = evaluator.eval(policy, envs, eval_splits)

                    for split in eval_splits:
                        if (
                            eval_results[split]["reward_mean"]
                            > best_result[split]["reward_mean"]
                        ):
                            best_params[split] = cur_params
                            best_result[split] = eval_results[split]
                            self.save_checkpoint(policy, f"best_{split}")

                        # log best result so far
                        logging.info(f"BEST {split} so far")
                        logging.info(
                            "Parameters: "
                            + pprint.pformat(best_params[split], indent=2)
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

    def _generate_scores(self, env, policy, temperature, num_rollouts):
        @torch.no_grad()
        def rollout_once():
            policy.eval()
            obs = env.reset()
            has_done = np.array([False] * env.num_envs)

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
