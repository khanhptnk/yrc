from dataclasses import dataclass, field


@dataclass
class DistributionConfig:
    distribution_mode: str = "easy"
    seed: int = 0
    start_level: int = 0
    num_levels: int = 100000


@dataclass
class ProcgenConfig:
    suite: str = "procgen"
    name: str = "coinrun"
    normalize_rew: bool = False
    num_envs: int = 128
    num_threads: int = 8
    train: DistributionConfig = field(default_factory=DistributionConfig)
    val_sim: DistributionConfig = field(
        default_factory=lambda: DistributionConfig(
            distribution_mode="easy", start_level=50000, num_levels=256
        )
    )
    val_true: DistributionConfig = field(
        default_factory=lambda: DistributionConfig(
            distribution_mode="hard", start_level=50000, num_levels=256
        )
    )
    test: DistributionConfig = field(
        default_factory=lambda: DistributionConfig(
            distribution_mode="hard", start_level=0, num_levels=100000
        )
    )

    def __post_init__(self):
        if isinstance(self.train, dict):
            self.train = DistributionConfig(**self.train)

        if isinstance(self.val_sim, dict):
            self.val_sim = DistributionConfig(**self.val_sim)

        if isinstance(self.val_true, dict):
            self.val_true = DistributionConfig(**self.val_true)

        if isinstance(self.test, dict):
            self.test = DistributionConfig(**self.test)
