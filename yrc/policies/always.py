from dataclasses import dataclass

import numpy as np

from yrc.core.policy import Policy
from yrc.utils.global_variables import get_global_variable


@dataclass
class AlwaysPolicyConfig:
    cls: str = "AlwaysPolicy"
    agent: str = "weak"


class AlwaysPolicy(Policy):
    def __init__(self, config, env):
        agent = config.coord_policy.agent
        assert agent in ["novice", "expert"], f"Unrecognized agent: {agent}!"
        self.choice = env.NOVICE if agent == "novice" else env.EXPERT

    def act(self, obs, greedy=False):
        env_suite = get_global_variable("env_suite")
        env_obs = obs["env_obs"]

        if isinstance(env_obs, dict):
            if env_suite == "cliport":
                action_shape = (1,)
            elif env_suite == "minigrid":
                action_shape = (env_obs["direction"].shape[0],)
        else:
            action_shape = (env_obs.shape[0],)

        action = np.ones(action_shape, dtype=np.int64) * self.choice
        return action
