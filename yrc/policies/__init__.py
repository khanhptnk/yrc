from .always import AlwaysPolicy
from .logit import LogitPolicy
from .ppo import PPOPolicy
from .pyod import PyODPolicy
from .random import RandomPolicy

registry = {
    "always": AlwaysPolicy,
    "logit": LogitPolicy,
    "pyod": PyODPolicy,
    "random": RandomPolicy,
    "ppo": PPOPolicy,
}
