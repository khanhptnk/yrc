from .always import AlwaysPolicy, AlwaysPolicyConfig
from .logit import LogitPolicy, LogitPolicyConfig
from .ppo import PPOPolicy, PPOPolicyConfig
from .pyod import PyODPolicy, PyODPolicyConfig
from .random import RandomPolicy, RandomPolicyConfig

config_cls = {
    "AlwaysPolicy": AlwaysPolicyConfig,
    "LogitPolicy": LogitPolicyConfig,
    "PyODPolicy": PyODPolicyConfig,
    "PPOPolicy": PPOPolicyConfig,
    "RandomPolicy": RandomPolicyConfig,
}
