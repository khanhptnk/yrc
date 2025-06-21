from .always import AlwaysAlgorithm, AlwaysAlgorithmConfig
from .logit import LogitAlgorithm, LogitAlgorithmConfig
from .ppo import PPOAlgorithm, PPOAlgorithmConfig
from .pyod import PyODAlgorithm, PyODAlgorithmConfig
from .random import RandomAlgorithm, RandomAlgorithmConfig

config_cls = {
    "AlwaysAlgorithm": AlwaysAlgorithmConfig,
    "LogitAlgorithm": LogitAlgorithmConfig,
    "PyODAlgorithm": PyODAlgorithmConfig,
    "PPOAlgorithm": PPOAlgorithmConfig,
    "RandomAlgorithm": RandomAlgorithmConfig,
}
