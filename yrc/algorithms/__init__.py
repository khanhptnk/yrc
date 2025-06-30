from .always import AlwaysAlgorithm
from .logit import LogitAlgorithm
from .ppo import PPOAlgorithm
from .pyod import PyODAlgorithm
from .random import RandomAlgorithm

registry = {
    "always": AlwaysAlgorithm,
    "logit": LogitAlgorithm,
    "pyod": PyODAlgorithm,
    "ppo": PPOAlgorithm,
    "random": RandomAlgorithm,
}
