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
    cls : str, optional
        Name of the policy class. Default is "PyODPolicy".
    method : str, optional
        PyOD method to use. Default is "DeepSVDD".
    feature_type : str, optional
        Type of feature representation to use. Default is "hidden".
    pyod_config : dict, optional
        Additional configuration for the PyOD model. Default is None.
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Attributes
    ----------
    cls : str
        Name of the policy class.
    method : str
        PyOD method to use.
    feature_type : str
        Type of feature representation to use.
    pyod_config : dict or None
        Additional configuration for the PyOD model.
    load_path : str or None
        Path to a checkpoint to load.
    """

    cls: str = "PyODPolicy"
    method: str = "DeepSVDD"
    feature_type: str = "hidden"
    pyod_config: Optional[Dict[str, Any]] = None
    load_path: Optional[str] = None


class PyODPolicy(Policy):
    """
    Policy that uses a PyOD outlier detector for action selection based on OOD scores.

    Parameters
    ----------
    config : PyODPolicyConfig
        Configuration object for the policy.
    env : object
        The environment instance, used to determine expert index.

    Attributes
    ----------
    config : PyODPolicyConfig
        Configuration object for the policy.
    threshold : float or None
        Threshold for OOD score to select expert action.
    device : torch.device or str
        Device for computation.
    clf : object
        PyOD model instance.
    feature_type : str
        Type of feature representation used.
    EXPERT : int
        Index of the expert action.
    """

    def __init__(self, config, env):
        """
        Initialize the PyODPolicy.

        Parameters
        ----------
        config : PyODPolicyConfig
            Configuration object for the policy.
        env : object
            The environment instance, used to determine expert index.
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

    def _get_pyod_class(self, config):
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
        """
        try:
            module_name, cls_name = config.method.split(".")
            module_name = f"yrc.lib.pyod.pyod.models.{module_name}"
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            return cls
        except Exception as e:
            raise ImportError(f"Could not import {config.method} from PyOD: {e}")

    def reset(self, done):
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

    def _make_input(self, obs):
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

    def fit(self, data):
        """
        Fit the PyOD model using the provided data.

        Parameters
        ----------
        data : dict
            Data dictionary containing features for fitting the model.

        Returns
        -------
        None
        """
        X = self._make_input(data)
        self.clf.fit(X)

    def get_train_scores(self):
        """
        Get the OOD decision scores from the PyOD model after fitting.

        Returns
        -------
        np.ndarray
            Array of decision scores for the training data.
        """
        return self.clf.decision_scores_

    def act(self, obs, temperature=None):
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
        if "threshold" in params:
            self.threshold = params["threshold"]
        if "clf" in params:
            self.clf = params["clf"]

    def get_params(self):
        """
        Get the current parameters of the policy.

        Returns
        -------
        dict
            Dictionary of policy parameters.
        """
        return {"threshold": self.threshold, "clf": self.clf}

    def train(self):
        """
        Set the PyOD model to training mode if applicable.

        Returns
        -------
        None
        """
        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.train()

    def eval(self):
        """
        Set the PyOD model to evaluation mode if applicable.

        Returns
        -------
        None
        """
        if hasattr(self.clf, "model_") and isinstance(self.clf.model_, nn.Module):
            self.clf.model_.eval()
