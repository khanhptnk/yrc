from dataclasses import dataclass

import torch
import torch.nn as nn

from yrc.models.impala import Impala
from yrc.utils.global_variables import get_global_variable
from yrc.utils.model import orthogonal_init


@dataclass
class PPOModelOutput:
    """
    Output container for PPO model forward pass.

    Attributes
    ----------
    logits : torch.Tensor
        The raw action logits output by the policy head.
    value : torch.Tensor
        The value function prediction output by the value head.
    hidden : torch.Tensor
        The hidden feature representation from the model.
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
    cls : str, optional
        Name of the model class. Default is "ImpalaPPOModel".
    """

    cls: str = "ImpalaPPOModel"


class ImpalaPPOModel(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.device = get_global_variable("device")
        self.embedder = Impala(env.observation_space.shape)
        self.hidden_dim = self.embedder.output_dim
        self.fc_policy = orthogonal_init(
            nn.Linear(self.hidden_dim, env.action_space.n), gain=0.01
        )
        self.fc_value = orthogonal_init(nn.Linear(self.hidden_dim, 1), gain=1.0)
        self.logit_dim = env.action_space.n

    def forward(self, obs):
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
    cls : str, optional
        Name of the model class. Default is "ImpalaCoordPPOModel".
    feature_type : str, optional
        Type of feature representation to use. Options include:
        "obs", "hidden", "hidden_obs", "dist", "hidden_dist", "obs_dist", "obs_hidden_dist".
        Default is "obs".
    """

    cls: str = "ImpalaCoordPPOModel"
    feature_type: str = "obs"


class ImpalaCoordPPOModel(nn.Module):
    def __init__(self, config, env):
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

    def forward(self, obs):
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
