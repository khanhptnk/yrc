from .always import AlwaysPolicy, AlwaysPolicyConfig
from .logit import LogitPolicy, LogitPolicyConfig

# from .ood import OODPolicy, OODPolicyConfig
from .ppo import PPOPolicy, PPOPolicyConfig
from .random import RandomPolicy, RandomPolicyConfig

config_cls = {
    "AlwaysPolicy": AlwaysPolicyConfig,
    "LogitPolicy": LogitPolicyConfig,
    # "OODPolicy": OODPolicyConfig,
    "PPOPolicy": PPOPolicyConfig,
    "RandomPolicy": RandomPolicyConfig,
}
