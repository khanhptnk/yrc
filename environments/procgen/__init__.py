from environments.procgen.procgenAISC.procgen import ProcgenEnv

from .config import ProcgenConfig
from .wrappers import (
    HardResetWrapper,
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)


def make_env(split, config):
    """
    Creates and configures a Procgen environment for a given data split.

    Parameters
    ----------
    split : str
        The split of the dataset to use (e.g., "train", "val_sim", "val_true", "test").
    config : ProcgenConfig
        The configuration object containing environment and split-specific settings.

    Returns
    -------
    env : ProcgenEnv or gym.Env
        The fully wrapped Procgen environment instance.

    Examples
    --------
    >>> from environments.procgen.config import ProcgenConfig
    >>> env = make_env("train", ProcgenConfig())
    """

    split_config = getattr(config, split)

    env = ProcgenEnv(
        env_name=config.task,
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
