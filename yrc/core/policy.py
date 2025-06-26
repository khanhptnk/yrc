import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy
import torch

from yrc.utils.global_variables import get_global_variable


def make(config, env):
    """
    Instantiate and return a policy object based on the provided configuration and environment.

    Parameters
    ----------
    config : object
        Configuration object with a 'cls' attribute specifying the policy class name.
    env : object
        The environment instance to be passed to the policy constructor.

    Returns
    -------
    policy : Policy
        Instantiated policy object.

    Examples
    --------
    >>> policy = make(config, env)
    """
    policy_cls = getattr(importlib.import_module("yrc.policies"), config.cls)
    policy = policy_cls(config, env)
    return policy


def load(path, env):
    """
    Load a policy from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    env : object
        The environment instance to be passed to the policy constructor.

    Returns
    -------
    policy : Policy
        Instantiated policy object with loaded parameters.

    Examples
    --------
    >>> policy = load('checkpoint.ckpt', env)
    """
    ckpt = torch.load(
        path, map_location=get_global_variable("device"), weights_only=False
    )
    config = ckpt["policy_config"]

    policy = make(config, env)
    policy.set_params(ckpt["model_state_dict"])
    logging.info(f"Loaded policy from {path}")

    return policy


class Policy(ABC):
    """
    Abstract base class for all policies in the YRC framework.

    This class defines the interface that all policy implementations must follow.

    Examples
    --------
    >>> class MyPolicy(Policy):
    ...     def act(self, obs):
    ...         return ...
    ...     def reset(self, done):
    ...         pass
    ...     def set_params(self, params):
    ...         pass
    ...     def get_params(self):
    ...         return {}
    ...     def train(self):
    ...         pass
    ...     def eval(self):
    ...         pass
    """

    @abstractmethod
    def act(self, obs: Any, *args, **kwargs) -> torch.Tensor:
        """
        Select an action based on the given observation.

        Parameters
        ----------
        obs : Any
            The current observation from the environment.

        Returns
        -------
        action : torch.Tensor
            The selected action. The format depends on the policy implementation.

        Examples
        --------
        >>> action = policy.act(obs)
        """
        pass

    @abstractmethod
    def reset(self, done: "numpy.ndarray") -> None:
        """
        Reset the internal state of the policy.

        This method should be overridden by subclasses to implement any necessary
        logic for resetting the policy's state to its initial configuration, such as
        clearing hidden states or episode-specific variables.

        Parameters
        ----------
        done : numpy.ndarray
            Boolean array indicating which episodes in a batch require a reset.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.reset(done)
        """
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the policy.

        This method should be overridden by subclasses to update the policy's parameters
        based on the provided dictionary, such as loading model weights or hyperparameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the new parameters for the policy.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.set_params(params)
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Returns the current parameters of the policy.

        This method should be overridden by subclasses to return the relevant parameters
        of the policy, such as model weights or hyperparameters.

        Returns
        -------
        params : dict
            A dictionary containing the current parameters of the policy.

        Examples
        --------
        >>> params = policy.get_params()
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Set the policy to training mode.

        This method should be overridden by subclasses to implement any necessary
        logic for preparing the policy for training, such as setting dropout or batch normalization layers.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.train()
        """
        pass

    @abstractmethod
    def eval(self) -> None:
        """
        Set the policy to evaluation mode.

        This method should be overridden by subclasses to implement any necessary
        logic for preparing the policy for evaluation, such as disabling dropout or batch normalization layers.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.eval()
        """
        pass
