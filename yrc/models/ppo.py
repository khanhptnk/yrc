from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
import torch.nn as nn

from yrc.models.impala import Impala
from yrc.utils.global_variables import get_global_variable
from yrc.utils.model import orthogonal_init


@dataclass
class PPOModelOutput:
    """
    Output container for PPO model forward pass.

    Parameters
    ----------
    logits : torch.Tensor
        The raw action logits output by the policy head.
    value : torch.Tensor
        The value function prediction output by the value head.
    hidden : torch.Tensor
        The hidden feature representation from the model.

    Examples
    --------
    >>> output = PPOModelOutput(logits, value, hidden)
    """

    logits: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor


@dataclass
class ImpalaPPOModelConfig:
    """
    Configuration dataclass for ImpalaPPOModel.

    Parameters
    ----------
    name : str, optional
        Name of the model class. Default is "ImpalaPPOModel".

    Examples
    --------
    >>> config = ImpalaPPOModelConfig()
    """

    name: str = "impala_ppo"


class ImpalaPPOModel(nn.Module):
    """
    PPO model using an IMPALA encoder for feature extraction.

    Examples
    --------
    >>> model = ImpalaPPOModel(ImpalaPPOModelConfig(), env)
    >>> obs = torch.randn(8, 3, 64, 64)
    >>> out = model(obs)
    >>> print(out.logits.shape, out.value.shape)
    """

    config_cls = ImpalaPPOModelConfig

    def __init__(self, config: ImpalaPPOModelConfig, env: "gym.Env") -> None:
        """
        Initialize the ImpalaPPOModel.

        Parameters
        ----------
        config : ImpalaPPOModelConfig
            Configuration object for the model.
        env : gym.Env
            The environment instance, used to determine input and output dimensions.

        Returns
        -------
        None

        Examples
        --------
        >>> model = ImpalaPPOModel(ImpalaPPOModelConfig(), env)
        """
        super().__init__()
        self.device = get_global_variable("device")
        self.embedder = Impala(env.observation_space.shape)
        self.hidden_dim = self.embedder.output_dim
        self.fc_policy = orthogonal_init(
            nn.Linear(self.hidden_dim, env.action_space.n), gain=0.01
        )
        self.fc_value = orthogonal_init(nn.Linear(self.hidden_dim, 1), gain=1.0)
        self.logit_dim = env.action_space.n

    def forward(self, obs: Any) -> PPOModelOutput:
        """
        Forward pass of the ImpalaPPOModel.

        Parameters
        ----------
        obs : torch.Tensor or np.ndarray
            Observation input to the model.

        Returns
        -------
        PPOModelOutput
            Output container with logits, value, and hidden features.

        Examples
        --------
        >>> out = model(obs)
        >>> print(out.logits.shape, out.value.shape)
        """
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(device=self.device)

        hidden = self.embedder(obs)
        logits = self.fc_policy(hidden)
        value = self.fc_value(hidden).reshape(-1)

        return PPOModelOutput(logits, value, hidden)


@dataclass
class ImpalaCoordPPOModelConfig:
    """
    Configuration dataclass for ImpalaCoordPPOModel.

    Parameters
    ----------
    name : str, optional
        Name of the model class. Default is "ImpalaCoordPPOModel".
    feature_type : str, optional
        Type of feature representation to use. Options include:
        "obs", "hidden", "hidden_obs", "dist", "hidden_dist", "obs_dist", "obs_hidden_dist".
        Default is "obs".

    Examples
    --------
    >>> config = ImpalaCoordPPOModelConfig(feature_type="obs_hidden_dist")
    """

    name: str = "impala_coord_ppo"
    feature_type: str = "obs"


class ImpalaCoordPPOModel(nn.Module):
    """
    PPO model for coordination environments, supporting multiple feature types.

    Examples
    --------
    >>> model = ImpalaCoordPPOModel(ImpalaCoordPPOModelConfig(), env)
    >>> obs = {"base_obs": ..., "novice_hidden": ..., "novice_logits": ...}
    >>> out = model(obs)
    >>> print(out.logits.shape, out.value.shape)
    """

    config_cls = ImpalaCoordPPOModelConfig

    def __init__(self, config: ImpalaCoordPPOModelConfig, env: "gym.Env") -> None:
        """
        Initialize the ImpalaCoordPPOModel.

        Parameters
        ----------
        config : ImpalaCoordPPOModelConfig
            Configuration object for the model.
        env : gym.Env
            The coordination environment instance.

        Returns
        -------
        None

        Examples
        --------
        >>> model = ImpalaCoordPPOModel(ImpalaCoordPPOModelConfig(), env)
        """
        super().__init__()

        self.device = get_global_variable("device")
        self.embedder = Impala(env.base_env.observation_space.shape)

        self.feature_type = config.feature_type
        if self.feature_type == "obs":
            self.hidden_dim = self.embedder.output_dim
        elif self.feature_type == "hidden":
            self.hidden_dim = env.novice.hidden_dim
        elif self.feature_type == "hidden_obs":
            self.hidden_dim = self.embedder.output_dim + env.novice.hidden_dim
        elif self.feature_type == "dist":
            self.hidden_dim = env.base_env.action_space.n
        elif self.feature_type == "hidden_dist":
            self.hidden_dim = env.novice.hidden_dim + env.base_env.action_space.n
        elif self.feature_type == "obs_dist":
            self.hidden_dim = self.embedder.output_dim + env.base_env.action_space.n
        elif self.feature_type == "obs_hidden_dist":
            self.hidden_dim = (
                self.embedder.output_dim
                + env.novice.hidden_dim
                + env.base_env.action_space.n
            )
        else:
            raise NotImplementedError

        self.fc_policy = orthogonal_init(
            nn.Linear(self.hidden_dim, env.action_space.n), gain=0.01
        )
        self.fc_value = orthogonal_init(nn.Linear(self.hidden_dim, 1), gain=1.0)
        self.logit_dim = env.action_space.n

    def forward(self, obs: Dict[str, Any]) -> PPOModelOutput:
        """
        Forward pass of the ImpalaCoordPPOModel.

        Parameters
        ----------
        obs : dict
            Dictionary containing observation components (base_obs, novice_hidden, novice_logits).

        Returns
        -------
        PPOModelOutput
            Output container with logits, value, and hidden features.

        Examples
        --------
        >>> out = model(obs)
        >>> print(out.logits.shape, out.value.shape)
        """
        base_obs = obs["base_obs"]
        if not torch.is_tensor(base_obs):
            base_obs = torch.from_numpy(base_obs).float().to(self.device)
        novice_hidden = obs["novice_hidden"]
        if not torch.is_tensor(novice_hidden):
            novice_hidden = torch.from_numpy(novice_hidden).float().to(self.device)
        novice_logits = obs["novice_logits"]
        if not torch.is_tensor(novice_logits):
            novice_logits = torch.from_numpy(novice_logits).float().to(self.device)

        if self.feature_type == "obs":
            hidden = self.embedder(base_obs)
        elif self.feature_type == "hidden":
            hidden = novice_hidden
        elif self.feature_type == "hidden_obs":
            hidden = torch.cat([self.embedder(base_obs), novice_hidden], dim=-1)
        elif self.feature_type == "dist":
            hidden = novice_logits.softmax(dim=-1)
        elif self.feature_type == "hidden_dist":
            hidden = torch.cat([novice_hidden, novice_logits.softmax(dim=-1)], dim=-1)
        elif self.feature_type == "obs_dist":
            hidden = torch.cat(
                [self.embedder(base_obs), novice_logits.softmax(dim=-1)], dim=-1
            )
        elif self.feature_type == "obs_hidden_dist":
            hidden = torch.cat(
                [
                    self.embedder(base_obs),
                    novice_hidden,
                    novice_logits.softmax(dim=-1),
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError

        logits = self.fc_policy(hidden)
        value = self.fc_value(hidden).reshape(-1)

        return PPOModelOutput(logits, value, hidden)
