from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    name : str, optional
        Name of the policy class. Default is "always".
    agent : str, optional
        The agent type to always select. Options are "novice" or "expert". Default is "novice".
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Examples
    --------
    >>> config = AlwaysPolicyConfig(agent="expert")
    """

    name: str = "always"
    agent: str = "novice"
    load_path: Optional[str] = None


class AlwaysPolicy(Policy):
    """
    Policy that always selects the same agent (novice or expert) for every action.

    Examples
    --------
    >>> policy = AlwaysPolicy(AlwaysPolicyConfig(agent="novice"), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    config_cls = AlwaysPolicyConfig

    def __init__(self, config: AlwaysPolicyConfig, env: "gym.Env") -> None:
        """
        Initialize the AlwaysPolicy.

        Parameters
        ----------
        config : AlwaysPolicyConfig
            Configuration object for the policy.
        env : gym.Env
            The environment instance, used to determine agent indices.

        Returns
        -------
        None

        Examples
        --------
        >>> policy = AlwaysPolicy(AlwaysPolicyConfig(agent="novice"), env)
        """
        self.choice = env.NOVICE if config.agent == "novice" else env.EXPERT
        self.device = get_global_variable("device")
        self.config = config

    def act(self, obs: Any, temperature: Optional[float] = None) -> torch.Tensor:
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

        Examples
        --------
        >>> action = policy.act(obs)
        """
        if isinstance(obs, dict):
            batch_size = obs["base_obs"].shape[0]
        elif isinstance(obs, np.ndarray):
            batch_size = obs.shape[0]
        else:
            raise ValueError("obs must be a dict or a numpy array")

        return torch.ones((batch_size,)).to(self.device) * self.choice

    def reset(self, done: np.ndarray) -> None:
        """
        Reset the policy state at episode boundaries.

        Parameters
        ----------
        done : np.ndarray
            Boolean array indicating which episodes in a batch require a reset.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.reset(done)
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current parameters of the policy.

        Returns
        -------
        dict
            Dictionary of policy parameters.

        Examples
        --------
        >>> params = policy.get_params()
        """
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the parameters of the policy.

        Parameters
        ----------
        params : dict
            Dictionary of policy parameters to set.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.set_params(params)
        """
        pass

    def train(self) -> None:
        """
        Set the policy to training mode.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.train()
        """
        pass

    def eval(self) -> None:
        """
        Set the policy to evaluation mode.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.eval()
        """
        pass
