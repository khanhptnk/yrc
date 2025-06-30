from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy
import torch


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
    def act(self, obs: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Select an action based on the given observation.

        Parameters
        ----------
        obs : Any
            The current observation from the environment.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The selected action. The format depends on the policy implementation.

        Examples
        --------
        >>> action = policy.act(obs)
        """
        pass

    @abstractmethod
    def reset(self, done: numpy.ndarray) -> None:
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
    def set_params(self, params: Dict[str, Any]) -> None:
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
    def get_params(self) -> Dict[str, Any]:
        """
        Returns the current parameters of the policy.

        This method should be overridden by subclasses to return the relevant parameters
        of the policy, such as model weights or hyperparameters.

        Returns
        -------
        dict
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
