import logging

import gymnasium as gym
import torch

import lib.Minigrid.minigrid as minigrid
import yrc.environments.minigrid.wrappers as wrappers
from lib.Minigrid.minigrid.wrappers import StochasticActionWrapper
from yrc.core.config import get_global_variable
from yrc.environments.minigrid.models import MinigridModel
from yrc.environments.minigrid.policies import MinigridPolicy


def create_env(name, config):
    common_config = config.common
    specific_config = getattr(config, name)
    full_env_name = common_config.env_name + specific_config.env_name_suffix
    envs = gym.make_vec(
        full_env_name,
        wrappers=(StochasticActionWrapper,),
        num_envs=common_config.num_envs,
    )
    envs.reset(seed=specific_config.seed)
    envs = wrappers.HardResetWrapper(envs)
    envs.obs_shape = envs.observation_space.spaces["image"].shape[1:]
    return envs


def load_policy(path, env):
    model = MinigridModel(env)
    model.to(get_global_variable("device"))
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    logging.info(f"Loaded model from {path}")

    policy = MinigridPolicy(model, env.num_envs)
    policy.eval()
    return policy
