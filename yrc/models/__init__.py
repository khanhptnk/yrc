from .ppo import (
    ImpalaCoordPPOModel,
    ImpalaCoordPPOModelConfig,
    ImpalaPPOModel,
    ImpalaPPOModelConfig,
)

config_cls = {
    "ImpalaPPOModel": ImpalaPPOModelConfig,
    "ImpalaCoordPPOModel": ImpalaCoordPPOModelConfig,
}
