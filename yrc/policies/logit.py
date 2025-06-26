from copy import deepcopy as dc
from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions.categorical import Categorical

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class LogitPolicyConfig:
    """
    Configuration dataclass for LogitPolicy.

    Parameters
    ----------
    cls : str, optional
        Name of the policy class. Default is "LogitPolicy".
    metric : str, optional
        Confidence metric to use. Default is "max_logit".
    threshold : float, optional
        Confidence threshold for expert query. Default is None.
    temperature : float, optional
        Temperature for scaling logits. Default is None.
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Attributes
    ----------
    cls : str
        Name of the policy class.
    metric : str
        Confidence metric to use.
    threshold : float or None
        Confidence threshold for expert query.
    temperature : float or None
        Temperature for scaling logits.
    load_path : str or None
        Path to a checkpoint to load.
    """

    cls: str = "LogitPolicy"
    metric: str = "max_logit"
    threshold: Optional[float] = None
    temperature: Optional[float] = None
    load_path: Optional[str] = None


class LogitPolicy(Policy):
    """
    Policy that selects actions based on logit confidence metrics and thresholds.

    Parameters
    ----------
    config : LogitPolicyConfig
        Configuration object for the policy.
    env : object
        The environment instance, used to determine expert index.

    Attributes
    ----------
    config : LogitPolicyConfig
        Configuration object for the policy.
    params : dict
        Dictionary of current policy parameters (threshold, temperature).
    device : torch.device or str
        Device for computation.
    EXPERT : int
        Index of the expert action.

    Examples
    --------
    >>> policy = LogitPolicy(LogitPolicyConfig(), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    def __init__(self, config, env):
        """
        Initialize the LogitPolicy.

        Parameters
        ----------
        config : LogitPolicyConfig
            Configuration object for the policy.
        env : object
            The environment instance, used to determine expert index.
        """
        self.config = config
        self.params = {"threshold": config.threshold, "temperature": config.temperature}
        self.device = get_global_variable("device")
        self.EXPERT = env.EXPERT

    def act(self, obs, temperature=None):
        """
        Select actions based on confidence scores and threshold.

        Parameters
        ----------
        obs : dict
            Observation dictionary containing 'novice_logits'.
        temperature : float, optional
            Unused. Included for API compatibility.

        Returns
        -------
        torch.Tensor
            Tensor of selected actions (expert or not) for the batch.
        """
        logits = obs["novice_logits"]
        if not torch.is_tensor(logits):
            logits = torch.from_numpy(logits).to(self.device).float()
        score = self.compute_confidence(logits)
        # query expert when confidence score < threshold
        action = torch.where(
            score < self.params["threshold"],
            self.EXPERT,
            1 - self.EXPERT,
        )
        return action

    def compute_confidence(self, logits):
        """
        Compute confidence scores from logits using the configured metric.

        Parameters
        ----------
        logits : torch.Tensor
            Logits tensor from the policy.

        Returns
        -------
        torch.Tensor
            Confidence scores for each sample in the batch.
        """
        # NOTE: higher = more confident
        metric = self.config.metric
        logits = logits / self.params["temperature"]
        if metric == "max_logit":
            score = logits.max(dim=-1)[0]
        elif metric == "max_prob":
            score = logits.softmax(dim=-1).max(dim=-1)[0]
        elif metric == "margin":
            if logits.size(-1) > 1:
                # Multi-class case
                top2 = logits.softmax(dim=-1).topk(2, dim=-1)[0]
                score = top2[:, 0] - top2[:, 1]
                score = score
            else:
                # Binary case when logits has shape (B, 1)
                prob = logits.sigmoid().squeeze(-1)
                score = torch.abs(2 * prob - 1)
        elif metric == "entropy":
            # NOTE: we compute NEGATIVE entropy so that higher = more confident
            score = -Categorical(logits=logits).entropy()
        elif metric == "energy":
            score = logits.logsumexp(dim=-1)
        else:
            raise NotImplementedError(f"Unrecognized metric: {metric}")
        return score

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
        return dc(self.params)

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

        Raises
        ------
        KeyError
            If a parameter key is not recognized by the policy.
        """
        for k, v in params.items():
            if k not in self.params:
                raise KeyError(f"Parameter {k} not recognized in LogitPolicy")
            self.params[k] = dc(v)

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
