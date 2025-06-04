from .always import AlwaysAlgorithm, AlwaysAlgorithmConfig
from .logit import LogitAlgorithm, LogitAlgorithmConfig
from .ood import OODAlgorithm, OODAlgorithmConfig
from .ppo import PPOAlgorithm, PPOAlgorithmConfig
from .random import RandomAlgorithm, RandomAlgorithmConfig

config_cls = {
    "AlwaysAlgorithm": AlwaysAlgorithmConfig,
    "LogitAlgorithm": LogitAlgorithmConfig,
    "OODAlgorithm": OODAlgorithmConfig,
    "PPOAlgorithm": PPOAlgorithmConfig,
    "RandomAlgorithm": RandomAlgorithmConfig,
}
