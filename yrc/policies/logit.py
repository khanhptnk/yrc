from copy import deepcopy as dc
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    name : str, optional
        Name of the policy class. Default is "logit".
    metric : str, optional
        Confidence metric to use. Default is "max_logit".
    threshold : float, optional
        Confidence threshold for expert query. Default is None.
    temperature : float, optional
        Temperature for scaling logits. Default is None.
    load_path : str, optional
        Path to a checkpoint to load. Default is None.

    Examples
    --------
    >>> config = LogitPolicyConfig(metric="max_prob", threshold=0.8)
    """

    name: str = "logit"
    metric: str = "max_logit"
    threshold: Optional[float] = None
    temperature: Optional[float] = None
    load_path: Optional[str] = None


class LogitPolicy(Policy):
    """
    Policy that selects actions based on logit confidence metrics and thresholds.

    Examples
    --------
    >>> policy = LogitPolicy(LogitPolicyConfig(), env)
    >>> obs = ...
    >>> action = policy.act(obs)
    """

    config_cls = LogitPolicyConfig

    def __init__(self, config: LogitPolicyConfig, env: "gym.Env") -> None:
        """
        Initialize the LogitPolicy.

        Parameters
        ----------
        config : LogitPolicyConfig
            Configuration object for the policy.
        env : gym.Env
            The environment instance, used to determine expert index.

        Returns
        -------
        None

        Examples
        --------
        >>> policy = LogitPolicy(LogitPolicyConfig(), env)
        """
        self.config = config
        self.params = {"threshold": config.threshold, "temperature": config.temperature}
        self.device = get_global_variable("device")
        self.EXPERT = env.EXPERT

    def act(
        self, obs: Dict[str, Any], temperature: Optional[float] = None
    ) -> torch.Tensor:
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

        Examples
        --------
        >>> action = policy.act(obs)
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

    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
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

        Raises
        ------
        NotImplementedError
            If the configured metric is not recognized.

        Examples
        --------
        >>> score = policy.compute_confidence(logits)
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
        return dc(self.params)

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

        Raises
        ------
        KeyError
            If a parameter key is not recognized by the policy.

        Examples
        --------
        >>> policy.set_params({'threshold': 0.7})
        """
        for k, v in params.items():
            if k not in self.params:
                raise KeyError(f"Parameter {k} not recognized in LogitPolicy")
            self.params[k] = dc(v)

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
