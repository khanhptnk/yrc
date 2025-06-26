from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class RandomPolicyConfig:
    """
    Configuration dataclass for RandomPolicy.

    Parameters
    ----------
    cls : str, optional
        Name of the policy class. Default is "RandomPolicy".
    prob : float, optional
        Probability of selecting the expert action. Setting this value prevents RandomAlgorithm from conducting a grid search.
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Attributes
    ----------
    cls : str
        Name of the policy class.
    prob : float or None
        Probability of selecting the expert action.
    load_path : str or None
        Path to a checkpoint to load.

    Examples
    --------
    >>> config = RandomPolicyConfig(prob=0.7)
    >>> print(config.cls)
    'RandomPolicy'
    >>> print(config.prob)
    0.7
    """

    cls: str = "RandomPolicy"
    prob: Optional[float] = None
    load_path: Optional[str] = None


class RandomPolicy(Policy):
    """
    Policy that selects the expert action with a fixed probability.

    Parameters
    ----------
    config : RandomPolicyConfig
        Configuration object for the policy.
    env : object
        The environment instance, used to determine expert index.

    Attributes
    ----------
    prob : float
        Probability of selecting the expert action.
    device : torch.device or str
        Device for computation.
    EXPERT : int
        Index of the expert action.
    config : RandomPolicyConfig
        Configuration object for the policy.

    Examples
    --------
    >>> policy = RandomPolicy(RandomPolicyConfig(prob=0.7), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    def __init__(self, config, env):
        """
        Initialize the RandomPolicy.

        Parameters
        ----------
        config : RandomPolicyConfig
            Configuration object for the policy.
        env : object
            The environment instance, used to determine expert index.
        """
        self.prob = config.prob
        self.device = get_global_variable("device")
        self.EXPERT = env.EXPERT
        self.config = config

    def act(self, obs, temperature=None):
        """
        Select actions randomly based on the configured probability.

        Parameters
        ----------
        obs : dict or np.ndarray
            Batch of observations. If dict, must contain 'base_obs'.
        temperature : float, optional
            Unused. Included for API compatibility.

        Returns
        -------
        torch.Tensor
            Tensor of selected actions (expert or not) for the batch.

        Raises
        ------
        ValueError
            If obs is not a dict or numpy array.
        """
        if isinstance(obs, dict):
            batch_size = obs["base_obs"].shape[0]
        elif isinstance(obs, np.ndarray):
            batch_size = obs.shape[0]
        else:
            raise ValueError("obs must be a dict or a numpy array")

        action = torch.where(
            torch.rand(batch_size, device=self.device) < self.prob,
            self.EXPERT,
            1 - self.EXPERT,
        )
        return action

    def reset(self, done):
        """
        Reset the policy state at episode boundaries.

        Parameters
        ----------
        done : np.ndarray
            Boolean array indicating which episodes in a batch require a reset.

        Returns
        -------
        None
        """
        pass

    def set_params(self, params):
        """
        Set the parameters of the policy.

        Parameters
        ----------
        params : dict
            Dictionary of policy parameters to set.

        Returns
        -------
        None
        """
        self.prob = params["prob"]

    def get_params(self):
        """
        Get the current parameters of the policy.

        Returns
        -------
        dict
            Dictionary of policy parameters.
        """
        return {"prob": self.prob}

    def train(self):
        """
        Set the policy to training mode.

        Returns
        -------
        None
        """
        pass

    def eval(self):
        """
        Set the policy to evaluation mode.

        Returns
        -------
        None
        """
        pass
