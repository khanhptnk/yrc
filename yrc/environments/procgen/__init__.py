import logging

import torch

import yrc.environments.procgen.wrappers as wrappers
from lib.procgenAISC.procgen import ProcgenEnv
from yrc.environments.procgen.models import ProcgenModel
from yrc.environments.procgen.policies import ProcgenPolicy
from yrc.utils.global_variables import get_global_variable


def create_env(split, config):
    split_config = getattr(config, split)

    env = ProcgenEnv(
        env_name=config.name,
        num_envs=config.num_envs,
        num_threads=config.num_threads,
        use_backgrounds=config.use_backgrounds,
        use_monochrome_assets=config.use_monochrome_assets,
        restrict_themes=config.restrict_themes,
        start_level=split_config.start_level,
        num_levels=split_config.num_levels,
        distribution_mode=split_config.distribution_mode,
        rand_seed=split_config.seed,
    )

    env = wrappers.VecExtractDictObs(env, "rgb")
    if config.normalize_rew:
        env = wrappers.VecNormalize(
            env, ob=False
        )  # normalizing returns, but not the img frames
    env = wrappers.TransposeFrame(env)
    env = wrappers.ScaledFloatFrame(env)
    # NOTE: this must be done last
    env = wrappers.HardResetWrapper(env)
    return env


def load_policy(path, env):
    model = ProcgenModel(env)
    model.to(get_global_variable("device"))
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Loaded model from {path}")

    policy = ProcgenPolicy(model)
    policy.eval()
    return policy
