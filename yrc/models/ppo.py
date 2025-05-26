from dataclasses import dataclass

import torch
import torch.nn as nn

from yrc.models.impala import Impala
from yrc.utils.global_variables import get_global_variable
from yrc.utils.model import orthogonal_init


@dataclass
class ImpalaPPOModelConfig:
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
        logit = self.fc_policy(hidden)
        value = self.fc_value(hidden).reshape(-1)
        return logit, value


@dataclass
class ImpalaCoordPPOModelConfig:
    cls: str = "ImpalaPPOModel"
    feature_type: str = (
        "obs"  # obs, hidden, hidden_obs, dist, hidden_dist, obs_dist, obs_hidden_dist
    )


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
        env_obs = (
            obs["env_obs"]["image"]
            if isinstance(obs["env_obs"], dict)
            else obs["env_obs"]
        )
        if not torch.is_tensor(env_obs):
            env_obs = torch.from_numpy(env_obs).float().to(self.device)
        novice_features = obs["novice_features"]
        if not torch.is_tensor(novice_features):
            novice_features = torch.from_numpy(novice_features).float().to(self.device)
        novice_logit = obs["novice_logit"]
        if not torch.is_tensor(novice_logit):
            novice_logit = torch.from_numpy(novice_logit).float().to(self.device)

        if self.feature_type == "obs":
            hidden = self.embedder(env_obs)
        elif self.feature_type == "hidden":
            hidden = novice_features
        elif self.feature_type == "hidden_obs":
            hidden = torch.cat([self.embedder(env_obs), novice_features], dim=-1)
        elif self.feature_type == "dist":
            hidden = novice_logit.softmax(dim=-1)
        elif self.feature_type == "hidden_dist":
            hidden = torch.cat([novice_features, novice_logit.softmax(dim=-1)], dim=-1)
        elif self.feature_type == "obs_dist":
            hidden = torch.cat(
                [self.embedder(env_obs), novice_logit.softmax(dim=-1)], dim=-1
            )
        elif self.feature_type == "obs_hidden_dist":
            hidden = torch.cat(
                [self.embedder(env_obs), novice_features, novice_logit.softmax(dim=-1)],
                dim=-1,
            )
        else:
            raise NotImplementedError

        logit = self.fc_policy(hidden)
        value = self.fc_value(hidden).reshape(-1)
        return logit, value
