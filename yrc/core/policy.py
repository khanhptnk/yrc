import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy
import torch

from yrc.utils.global_variables import get_global_variable


def make(config, env):
    policy_cls = getattr(importlib.import_module("yrc.policies"), config.cls)
    policy = policy_cls(config, env)
    return policy


def load(path, env):
    ckpt = torch.load(path, map_location=get_global_variable("device"))
    config = ckpt["policy_config"]

    policy_cls = getattr(importlib.import_module("yrc.policies"), config.cls)
    policy = policy_cls(config, env)
    policy.load_model_checkpoint(ckpt["model_state_dict"])
    logging.info(f"Loaded policy from {path}")

    return policy


class Policy(ABC):
    @abstractmethod
    def act(self, obs: Any, greedy: float = False) -> torch.Tensor:
        """
        Selects an action based on the given observation.

        Parameters
        ----------
        obs : Any
            The current observation from the environment.
        greedy : bool, optional
            If True, selects the action greedily (e.g., for evaluation).
            If False, may use a stochastic or exploratory policy. Defaults to False.

        Returns
        -------
        action : torch.Tensor
            The selected action. The format depends on the policy implementation.

        Examples
        --------
        >>> action = policy.act(obs, greedy=True)
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
    def load_model_checkpoint(self, load_path: str) -> None:
        """
        Loads a model checkpoint from the specified file path.

        Parameters
        ----------
        load_path : str
            The file path to the model checkpoint to be loaded.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.load_model_checkpoint("checkpoints/model.pt")
        """
        pass
