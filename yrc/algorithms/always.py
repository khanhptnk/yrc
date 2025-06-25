from dataclasses import dataclass
from typing import Dict, List

from yrc.core.algorithm import Algorithm


@dataclass
class AlwaysAlgorithmConfig:
    """
    Configuration dataclass for AlwaysAlgorithm.

    Parameters
    ----------
    cls : str, optional
        Name of the algorithm class. Default is "AlwaysAlgorithm".
    """

    cls: str = "AlwaysAlgorithm"


class AlwaysAlgorithm(Algorithm):
    def __init__(self, config):
        pass

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        env: "gym.Env",
        validators: Dict[str, "yrc.core.Evaluator"],
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
            List of evaluation splits. Default is ["val_sim, val_true"].

        Returns
        -------
        None
        """
        pass
