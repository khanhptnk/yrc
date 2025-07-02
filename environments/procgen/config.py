from dataclasses import dataclass, field


@dataclass
class ProcgenDistributionConfig:
    """
    Configuration for a Procgen environment distribution.

    Parameters
    ----------
    distribution_mode : str, optional
        The difficulty mode of the environment. Options include "easy" or "hard". Default is "easy".
    seed : int, optional
        Random seed for environment reproducibility. Default is 0.
    start_level : int, optional
        The starting level for the environment. Default is 0.
    num_levels : int, optional
        The number of unique levels to sample from. Default is 100000.

    Examples
    --------
    >>> config = ProcgenDistributionConfig(distribution_mode="hard", seed=42, start_level=100, num_levels=500)
    """

    distribution_mode: str = "easy"
    seed: int = 0
    start_level: int = 0
    num_levels: int = 100000


@dataclass
class ProcgenConfig:
    """
    Configuration dataclass for Procgen environments.

    Parameters
    ----------
    name : str, optional
        Name of the environment suite. Default is "procgen".
    task : str, optional
        Name of the specific Procgen environment. Default is "coinrun".
    normalize_rew : bool, optional
        Whether to normalize rewards. Default is False.
    num_envs : int, optional
        Number of parallel environments to run. Default is 128.
    num_threads : int, optional
        Number of threads to use for environment execution. Default is 8.
    train : ProcgenDistributionConfig, optional
        Configuration for the training distribution.
    val_sim : ProcgenDistributionConfig, optional
        Configuration for the simulated validation distribution.
    val_true : ProcgenDistributionConfig, optional
        Configuration for the true validation distribution.
    test : ProcgenDistributionConfig, optional
        Configuration for the test distribution.

    Examples
    --------
    >>> cfg = ProcgenConfig(task="coinrun", num_envs=64)
    """

    name: str = "procgen"
    task: str = "coinrun"
    normalize_rew: bool = False
    num_envs: int = 128
    num_threads: int = 8
    train: ProcgenDistributionConfig = field(default_factory=ProcgenDistributionConfig)
    val_sim: ProcgenDistributionConfig = field(
        default_factory=lambda: ProcgenDistributionConfig(
            distribution_mode="easy", start_level=50000, num_levels=256
        )
    )
    val_true: ProcgenDistributionConfig = field(
        default_factory=lambda: ProcgenDistributionConfig(
            distribution_mode="hard", start_level=50000, num_levels=256
        )
    )
    test: ProcgenDistributionConfig = field(
        default_factory=lambda: ProcgenDistributionConfig(
            distribution_mode="hard", start_level=0, num_levels=100000
        )
    )

    def __post_init__(self):
        """
        Ensures that all distribution configuration attributes are instances of
        ProcgenDistributionConfig, converting from dicts if necessary.

        This method is automatically called after the dataclass is initialized.

        Examples
        --------
        >>> cfg = ProcgenConfig(train={"distribution_mode": "easy"})
        >>> isinstance(cfg.train, ProcgenDistributionConfig)
        True
        """
        if isinstance(self.train, dict):
            self.train = ProcgenDistributionConfig(**self.train)

        if isinstance(self.val_sim, dict):
            self.val_sim = ProcgenDistributionConfig(**self.val_sim)

        if isinstance(self.val_true, dict):
            self.val_true = ProcgenDistributionConfig(**self.val_true)

        if isinstance(self.test, dict):
            self.test = ProcgenDistributionConfig(**self.test)
