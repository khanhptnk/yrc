from dataclasses import dataclass
from typing import Any, Optional

import torch.nn.functional as F
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
    cls : str, optional
        Name of the policy class. Default is "PPOPolicy".
    model : Any, optional
        Model configuration or class name. Default is "ImpalaCoordPPOModel".
    load_path : Optional[str], optional
        Path to a checkpoint to load the policy weights from. Default is None.

    Attributes
    ----------
    cls : str
        Name of the policy class.
    model : Any
        Model configuration or class name.
    load_path : Optional[str]
        Path to a checkpoint to load the policy weights from.
    """

    cls: str = "PPOPolicy"
    model: Any = "ImpalaCoordPPOModel"
    load_path: Optional[str] = None

    def __post_init__(self):
        """
        Post-initialization logic for PPOPolicyConfig.

        Converts string or dictionary model fields into their respective configuration objects.

        Raises
        ------
        IndexError
            If required keys are missing in configuration dictionaries.
        ValueError
            If model is not a string or a dictionary.
        """
        if isinstance(self.model, str):
            self.model = model_factory.config_cls[self.model]()
        elif isinstance(self.model, dict):
            if "cls" not in self.model:
                raise IndexError(
                    "Please specify policy.model.cls through YAML file or flag"
                )
            self.model = model_factory.config_cls[self.model["cls"]](**self.model)
        else:
            raise ValueError("model must be a string or a dictionary")


class PPOPolicy(Policy):
    """
    Policy class for PPO, wrapping a model and providing action selection and parameter management.

    Parameters
    ----------
    config : PPOPolicyConfig
        Configuration object for the policy.
    env : object
        The environment instance, used to determine model input/output dimensions.

    Attributes
    ----------
    model : nn.Module
        The underlying model used for action selection.
    config : PPOPolicyConfig
        Configuration object for the policy.

    Examples
    --------
    >>> policy = PPOPolicy(PPOPolicyConfig(), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    def __init__(self, config, env):
        """
        Initialize the PPOPolicy.

        Parameters
        ----------
        config : PPOPolicyConfig
            Configuration object for the policy.
        env : object
            The environment instance, used to determine model input/output dimensions.
        """
        model_cls = getattr(model_factory, config.model.cls)
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
        """
        pass

    def act(self, obs, temperature=1.0, return_model_output=False):
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

    def set_params(self, params):
        """
        Set the model parameters from a state dictionary.

        Parameters
        ----------
        params : dict
            State dictionary of model parameters.

        Returns
        -------
        None
        """
        self.model.load_state_dict(params)

    def get_params(self):
        """
        Get the current model parameters as a state dictionary.

        Returns
        -------
        dict
            State dictionary of model parameters.
        """
        return self.model.state_dict()

    def train(self):
        """
        Set the policy/model to training mode.

        Returns
        -------
        None
        """
        self.model.train()

    def eval(self):
        """
        Set the policy/model to evaluation mode.

        Returns
        -------
        None
        """
        self.model.eval()
