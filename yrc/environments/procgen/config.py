from dataclasses import dataclass, field
from typing import List


@dataclass
class DistributionConfig:
    distribution_mode: str = "easy"
    seed: int = 0
    start_level: int = 0
    num_levels: int = 100000


@dataclass
class ProcgenConfig:
    # general
    suite: str = "procgen"
    name: str = "coinrun"
    normalize_rew: bool = False
    num_envs: int = 64
    num_threads: int = 8
    use_backgrounds: bool = True
    use_monochrome_assets: bool = False
    restrict_themes: bool = False
    # env splits
    train: DistributionConfig = DistributionConfig()
    val_sim: DistributionConfig = DistributionConfig(
        distribution_mode="easy", start_level=50000, num_levels=256
    )
    val_true: DistributionConfig = DistributionConfig(
        distribution_mode="hard", start_level=50000, num_levels=256
    )
    test: DistributionConfig = DistributionConfig(
        distribution_mode="hard", start_level=0, num_levels=100000
    )

    def __post_init__(self):
        # train split
        if isinstance(self.train, dict):
            self.train = DistributionConfig(**self.train)

        # val_sim split
        if isinstance(self.val_sim, dict):
            self.val_sim = DistributionConfig(**self.val_sim)

        # val_true split
        if isinstance(self.val_true, dict):
            self.val_true = DistributionConfig(**self.val_true)

        # test split
        if isinstance(self.test, dict):
            self.test = DistributionConfig(**self.test)
