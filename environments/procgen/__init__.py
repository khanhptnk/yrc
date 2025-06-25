from environments.procgen.procgenAISC.procgen import ProcgenEnv

from .wrappers import (
    HardResetWrapper,
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)


def make_env(split, config):
    split_config = getattr(config, split)

    env = ProcgenEnv(
        env_name=config.name,
        num_envs=config.num_envs,
        num_threads=config.num_threads,
        start_level=split_config.start_level,
        num_levels=split_config.num_levels,
        distribution_mode=split_config.distribution_mode,
        rand_seed=split_config.seed,
    )

    env = VecExtractDictObs(env, "rgb")
    if config.normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the img frames
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    # NOTE: this must be done last
    env = HardResetWrapper(env)
    return env
