import importlib

if importlib.util.find_spec("gymnasium") is None:
    import gym
else:
    import gymnasium as gym  # used for minigrid
from copy import deepcopy as dc

import numpy as np

from yrc.utils.global_variables import get_global_variable


def make_base_env(split, config):
    module = importlib.import_module(
        f"yrc.environments.{get_global_variable('env_suite')}"
    )
    create_fn = getattr(module, "create_env")
    env = create_fn(split, config)
    return env


class CoordEnv(gym.Env):
    NOVICE = 0
    EXPERT = 1

    def __init__(self, config, base_env, novice, expert):
        self.config = config
        self.base_env = base_env

        # if isinstance(base_env.observation_space, list):
        #     obs_space = base_env.observation_space[0]
        # elif isinstance(base_env.observation_space, gym.spaces.Dict):
        #     obs_space = base_env.observation_space.spaces["image"]
        # else:
        #     obs_space = base_env.observation_space

        self.novice = novice
        self.expert = expert

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "env_obs": base_env.observation_space,
                "novice_features": gym.spaces.Box(
                    -100, 100, shape=(novice.model.hidden_dim,)
                ),
                "novice_logit": gym.spaces.Box(
                    -100, 100, shape=(novice.model.logit_dim,)
                ),
            }
        )

    def set_costs(self, test_eval_info):
        length = test_eval_info["episode_length_mean"]
        reward = test_eval_info["reward_mean"]
        reward_per_action = reward / length

        # NOTE: paper results were generated with rounding
        # self.expert_query_cost_per_action = round(
        #     reward_per_action * self.config.expert_query_cost_alpha, 2
        # )
        # self.switch_agent_cost_per_action = round(
        #     reward_per_action * self.config.switch_agent_cost_beta, 2
        # )

        self.expert_query_cost_per_action = (
            reward_per_action * self.config.expert_query_cost_alpha,
        )

        self.switch_agent_cost_per_action = (
            reward_per_action * self.config.switch_agent_cost_beta
        )

    @property
    def num_envs(self):
        return self.base_env.num_envs

    # @property
    # def action_shape(self):
    #     return self.action_space.shape

    # @property
    # def obs_shape(self):
    #     return {
    #         "env_obs": self.base_env.obs_shape,
    #         "novice_features": (self.novice.hidden_dim,),
    #         "novice_logit": (self.base_env.action_space.n,),
    #     }

    def reset(self):
        self.prev_action = None
        self.env_obs = self.base_env.reset()
        self._reset_agents(np.array([True] * self.num_envs))
        return self.get_obs()

    def _reset_agents(self, done):
        self.novice.reset(done)
        self.expert.reset(done)

    def step(self, action):
        env_action = self._compute_env_action(action)
        self.env_obs, env_reward, done, env_info = self.base_env.step(env_action)

        info = dc(env_info)
        if len(info) == 0:
            info = [{"env_reward": 0, "env_action": 0}] * self.num_envs
        for i, item in enumerate(info):
            if "env_reward" not in item:
                item["env_reward"] = env_reward[i]
            item["env_action"] = env_action[i]

        reward = self._get_reward(env_reward, action, done)
        self._reset_agents(done)
        self.prev_action = action

        return self.get_obs(), reward, done, info

    def _compute_env_action(self, action):
        # NOTE: this method only works with non-recurrent agent models
        greedy = self.config.act_greedy
        is_novice = action == self.NOVICE
        is_expert = ~is_novice

        if isinstance(self.env_obs, dict):
            if is_novice.any():
                env_action = self.novice.act(self.env_obs, greedy=greedy)
            if is_expert.any():
                if get_global_variable("benchmark") == "cliport":
                    env_action = self.expert.act(
                        self.env_obs, self.base_env, greedy=greedy
                    )
                else:
                    env_action = self.expert.act(self.env_obs, greedy=greedy)
        else:
            env_action = np.zeros_like(action)
            if is_novice.any():
                env_action[is_novice] = self.novice.act(
                    self.env_obs[is_novice], greedy=greedy
                )
            if is_expert.any():
                env_action[is_expert] = self.expert.act(
                    self.env_obs[is_expert], greedy=greedy
                )
        return env_action

    def get_obs(self):
        obs = {
            "env_obs": self.env_obs,
            "novice_features": self.novice.get_hidden(self.env_obs)
            .detach()
            .cpu()
            .numpy(),
            "novice_logit": self.novice.forward(self.env_obs).detach().cpu().numpy(),
        }
        return obs

    def _get_reward(self, env_reward, action, done):
        # cost of querying expert agent
        reward = np.where(
            action == self.EXPERT,
            env_reward - self.expert_query_cost_per_action,
            env_reward,
        )

        # cost of switching
        if self.prev_action is not None:
            switch_indices = ((action != self.prev_action) & (~done)).nonzero()[0]
            if switch_indices.size > 1:
                reward[switch_indices] -= self.switch_agent_cost_per_action

        return reward

    def close(self):
        return self.base_env.close()
