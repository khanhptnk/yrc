from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.distributions.categorical import Categorical

import yrc.models as model_factory
from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class PPOPolicyConfig:
    """
    Configuration dataclass for PPOPolicy.

    Parameters
    ----------
    name : str, optional
        Name of the policy class. Default is "ppo".
    model : Any, optional
        Model configuration or class name. Default is "impala_coord_ppo".
    load_path : Optional[str], optional
        Path to a checkpoint to load the policy weights from. Default is None.

    Examples
    --------
    >>> config = PPOPolicyConfig(model="impala_coord_ppo")
    """

    name: str = "ppo"
    model: Any = "impala_coord_ppo"
    load_path: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Post-initialization logic for PPOPolicyConfig.

        Converts string or dictionary model fields into their respective configuration objects.

        Raises
        ------
        IndexError
            If required keys are missing in configuration dictionaries.
        ValueError
            If model is not a string or a dictionary.

        Examples
        --------
        >>> config = PPOPolicyConfig(model="impala_coord_ppo")
        """
        if isinstance(self.model, str):
            self.model = model_factory.registry[self.model].config_cls()
        elif isinstance(self.model, dict):
            if "name" not in self.model:
                raise IndexError(
                    "Please specify policy.model.name through YAML file or flag. "
                    "This name must already be registered in yrc.models.registry."
                )
            self.model = model_factory.registry[self.model["name"]].config_cls(
                **self.model
            )
        else:
            raise ValueError("model must be a string or a dictionary")


class PPOPolicy(Policy):
    """
    Policy class for PPO, wrapping a model and providing action selection and parameter management.

    Examples
    --------
    >>> policy = PPOPolicy(PPOPolicyConfig(), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    config_cls = PPOPolicyConfig

    def __init__(self, config: PPOPolicyConfig, env: "gym.Env") -> None:
        """
        Initialize the PPOPolicy.

        Parameters
        ----------
        config : PPOPolicyConfig
            Configuration object for the policy.
        env : gym.Env
            The environment instance, used to determine model input/output dimensions.

        Returns
        -------
        None

        Examples
        --------
        >>> policy = PPOPolicy(PPOPolicyConfig(), env)
        """
        model_cls = model_factory.registry[config.model.name]
        self.model = model_cls(config.model, env)
        self.model.to(get_global_variable("device"))
        self.config = config

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

        Examples
        --------
        >>> policy.reset(done)
        """
        pass

    def act(
        self, obs: Any, temperature: float = 1.0, return_model_output: bool = False
    ) -> Any:
        """
        Select an action based on the observation and temperature.

        Parameters
        ----------
        obs : Any
            Observation input to the policy.
        temperature : float, optional
            Sampling temperature. If 0, selects the argmax action. Default is 1.0.
        return_model_output : bool, optional
            If True, also return the model output. Default is False.

        Returns
        -------
        action : torch.Tensor or tuple
            Selected action, or (action, model_output) if return_model_output is True.

        Examples
        --------
        >>> action = policy.act(obs)
        >>> action, model_output = policy.act(obs, return_model_output=True)
        """
        model_output = self.model(obs)
        if temperature == 0:
            action = model_output.logits.argmax(dim=-1)
        else:
            dist = Categorical(logits=model_output.logits / temperature)
            action = dist.sample()
        if return_model_output:
            return action, model_output
        return action

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the model parameters from a state dictionary.

        Parameters
        ----------
        params : dict
            State dictionary of model parameters.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.set_params(params)
        """
        self.model.load_state_dict(params)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters as a state dictionary.

        Returns
        -------
        dict
            State dictionary of model parameters.

        Examples
        --------
        >>> params = policy.get_params()
        """
        return self.model.state_dict()

    def train(self) -> None:
        """
        Set the policy/model to training mode.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.train()
        """
        self.model.train()

    def eval(self) -> None:
        """
        Set the policy/model to evaluation mode.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.eval()
        """
        self.model.eval()
