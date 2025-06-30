import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class PyODPolicyConfig:
    """
    Configuration dataclass for PyODPolicy.

    Parameters
    ----------
    name : str, optional
        Name of the policy class. Default is "pyod".
    method : str, optional
        PyOD method to use. Default is "deepsvdd.DeepSVDD".
    feature_type : str, optional
        Type of feature representation to use. Default is "hidden".
    pyod_config : dict, optional
        Additional configuration for the PyOD model. Default is None.
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Examples
    --------
    >>> config = PyODPolicyConfig(method="deepsvdd.DeepSVDD", feature_type="hidden")
    """

    name: str = "pyod"
    method: str = "deepsvdd.DeepSVDD"
    feature_type: str = "hidden"
    pyod_config: Optional[Dict[str, Any]] = None
    load_path: Optional[str] = None


class PyODPolicy(Policy):
    """
    Policy that uses a PyOD outlier detector for action selection based on OOD scores.

    Examples
    --------
    >>> policy = PyODPolicy(PyODPolicyConfig(), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    config_cls = PyODPolicyConfig

    def __init__(self, config: PyODPolicyConfig, env: "gym.Env") -> None:
        """
        Initialize the PyODPolicy.

        Parameters
        ----------
        config : PyODPolicyConfig
            Configuration object for the policy.
        env : gym.Env
            The environment instance, used to determine expert index.

        Returns
        -------
        None

        Examples
        --------
        >>> policy = PyODPolicy(PyODPolicyConfig(), env)
        """
        self.config = config
        self.threshold = None
        self.device = get_global_variable("device")

        config.pyod_config["device"] = self.device
        config.pyod_config["random_state"] = get_global_variable("seed")
        self.clf = self._get_pyod_class(config)(**config.pyod_config)

        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.to(self.device)

        self.feature_type = config.feature_type
        self.EXPERT = env.EXPERT

    def _get_pyod_class(self, config: PyODPolicyConfig) -> type:
        """
        Dynamically import and return the PyOD class specified in the config.

        Parameters
        ----------
        config : PyODPolicyConfig
            Configuration object for the policy.

        Returns
        -------
        type
            The PyOD class to instantiate.

        Raises
        ------
        ImportError
            If the specified class cannot be imported.

        Examples
        --------
        >>> cls = policy._get_pyod_class(config)
        """
        try:
            module_name, cls_name = config.method.split(".")
            module_name = f"yrc.lib.pyod.pyod.models.{module_name}"
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            return cls
        except Exception as e:
            raise ImportError(f"Could not import {config.method} from PyOD: {e}")

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

    def _make_input(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Construct the input feature array for the PyOD model from the observation.

        Parameters
        ----------
        obs : dict
            Observation dictionary containing required features.

        Returns
        -------
        np.ndarray
            Concatenated feature array for the PyOD model.

        Raises
        ------
        AssertionError
            If no features are selected for PyOD input.

        Examples
        --------
        >>> inp = policy._make_input(obs)
        """
        inp = []
        if "obs" in self.feature_type:
            base_obs = obs["base_obs"]
            if base_obs.ndim > 2:
                # If env_obs is a tensor with more than 2 dimensions, flatten it
                base_obs = base_obs.reshape(base_obs.shape[0], -1)
            inp.append(base_obs)

        if "hidden" in self.feature_type:
            inp.append(obs["novice_hidden"])
        if "dist" in self.feature_type:
            inp.append(obs["novice_logit"].softmax(dim=-1))

        assert len(inp) > 0, "No features selected for PyOD input"

        inp = np.concatenate(inp, axis=1)

        return inp

    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the PyOD model using the provided data.

        Parameters
        ----------
        data : dict
            Data dictionary containing features for fitting the model.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.fit(data)
        """
        X = self._make_input(data)
        self.clf.fit(X)

    def get_train_scores(self) -> np.ndarray:
        """
        Get the OOD decision scores from the PyOD model after fitting.

        Returns
        -------
        np.ndarray
            Array of decision scores for the training data.

        Examples
        --------
        >>> scores = policy.get_train_scores()
        """
        return self.clf.decision_scores_

    def act(
        self, obs: Dict[str, Any], temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Select actions based on OOD scores from the PyOD model.

        Parameters
        ----------
        obs : dict
            Observation dictionary containing required features.
        temperature : float, optional
            Unused. Included for API compatibility.

        Returns
        -------
        torch.Tensor
            Tensor of selected actions (expert or not) for the batch.

        Examples
        --------
        >>> action = policy.act(obs)
        """
        inp = self._make_input(obs)
        score = self.clf.decision_function(inp)
        score = torch.from_numpy(score).float().to(get_global_variable("device"))

        action = torch.where(
            score < self.threshold,
            self.EXPERT,
            1 - self.EXPERT,
        )
        return action

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
        >>> policy.set_params({'threshold': 0.5, 'clf': clf})
        """
        if "threshold" in params:
            self.threshold = params["threshold"]
        if "clf" in params:
            self.clf = params["clf"]

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
        return {"threshold": self.threshold, "clf": self.clf}

    def train(self) -> None:
        """
        Set the PyOD model to training mode if applicable.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.train()
        """
        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.train()

    def eval(self) -> None:
        """
        Set the PyOD model to evaluation mode if applicable.

        Returns
        -------
        None

        Examples
        --------
        >>> policy.eval()
        """
        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.eval()
