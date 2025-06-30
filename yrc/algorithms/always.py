from dataclasses import dataclass
from typing import Dict

from yrc.core.algorithm import Algorithm


@dataclass
class AlwaysAlgorithmConfig:
    """
    Configuration for the AlwaysAlgorithm, which always returns the same action.

    Parameters
    ----------
    name : str, optional
        Name of the algorithm class. Default is "always".

    Examples
    --------
    >>> config = AlwaysAlgorithmConfig()
    """

    name: str = "always"


class AlwaysAlgorithm(Algorithm):
    """
    Algorithm that always returns the same action, regardless of input.

    Examples
    --------
    >>> algo = AlwaysAlgorithm(AlwaysAlgorithmConfig())
    """

    config_cls = AlwaysAlgorithmConfig

    def __init__(self, config: AlwaysAlgorithmConfig) -> None:
        """
        Initialize the AlwaysAlgorithm.

        Parameters
        ----------
        config : AlwaysAlgorithmConfig
            Configuration object for the AlwaysAlgorithm.

        Returns
        -------
        None

        Examples
        --------
        >>> algo = AlwaysAlgorithm(AlwaysAlgorithmConfig())
        """
        pass

    def train(
        self,
        policy: "yrc.core.Policy",
        env: "gym.Env",
        validators: Dict[str, "yrc.core.Evaluator"],
    ) -> None:
        """
        Run the AlwaysAlgorithm training procedure.

        This method evaluates the provided policy in the given environment using the specified evaluators.
        The AlwaysAlgorithm always returns the same action, regardless of the input observation.

        Parameters
        ----------
        policy : yrc.core.Policy
            The policy instance to use for generating actions.
        env : gym.Env
            The environment in which the policy is evaluated.
        validators : dict of str to yrc.core.Evaluator
            Dictionary mapping split names to evaluator instances for evaluation.

        Returns
        -------
        None

        Examples
        --------
        >>> algorithm = AlwaysAlgorithm(AlwaysAlgorithmConfig())
        >>> algorithm.train(policy, env, validators)
        """
        pass
