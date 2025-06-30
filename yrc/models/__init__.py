from .ppo import (
    ImpalaCoordPPOModel,
    ImpalaPPOModel,
)

registry = {
    "impala_ppo": ImpalaPPOModel,
    "impala_coord_ppo": ImpalaCoordPPOModel,
}
