from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class AlwaysPolicyConfig:
    """
    Configuration dataclass for AlwaysPolicy.

    Parameters
    ----------
    cls : str, optional
        Name of the policy class. Default is "AlwaysPolicy".
    agent : str, optional
        The agent type to always select. Options are "novice" or "expert". Default is "novice".
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Attributes
    ----------
    cls : str
        Name of the policy class.
    agent : str
        The agent type to always select.
    load_path : str or None
        Path to a checkpoint to load.

    Examples
    --------
    >>> config = AlwaysPolicyConfig(agent="expert")
    """

    cls: str = "AlwaysPolicy"
    agent: str = "novice"
    load_path: Optional[str] = None


class AlwaysPolicy(Policy):
    """
    Policy that always selects the same agent (novice or expert) for every action.

    Parameters
    ----------
    config : AlwaysPolicyConfig
        Configuration object for the policy.
    env : object
        The environment instance, used to determine agent indices.

    Attributes
    ----------
    choice : int
        The constant action (agent index) to select.
    device : torch.device or str
        Device for computation.
    config : AlwaysPolicyConfig
        Configuration object for the policy.

    Examples
    --------
    >>> policy = AlwaysPolicy(AlwaysPolicyConfig(agent="novice"), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    def __init__(self, config, env):
        """
        Initialize the AlwaysPolicy.

        Parameters
        ----------
        config : AlwaysPolicyConfig
            Configuration object for the policy.
        env : object
            The environment instance, used to determine agent indices.
        """
        self.choice = env.NOVICE if config.agent == "novice" else env.EXPERT
        self.device = get_global_variable("device")
        self.config = config

    def act(self, obs, temperature=None):
        """
        Select the constant action for a batch of observations.

        Parameters
        ----------
        obs : dict or np.ndarray
            Batch of observations. If dict, must contain 'base_obs'.
        temperature : float, optional
            Unused. Included for API compatibility.

        Returns
        -------
        torch.Tensor
            Tensor of constant actions (agent indices) for the batch.

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

        return torch.ones((batch_size,)).to(self.device) * self.choice

    def reset(self, done: "numpy.ndarray") -> None:
        """
        Reset the policy state at episode boundaries.

        Parameters
        ----------
        done : numpy.ndarray
            Boolean array indicating which episodes in a batch require a reset.

        Returns
        -------
        None
        """
        pass

    def get_params(self):
        """
        Get the current parameters of the policy.

        Returns
        -------
        dict
            Dictionary of policy parameters.
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
        pass

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
