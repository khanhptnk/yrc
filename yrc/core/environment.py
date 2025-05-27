import importlib
import json
from copy import deepcopy as dc
from typing import Any, List, Tuple

import gym
import numpy as np
import torch

import yrc
from yrc.utils.global_variables import get_global_variable


def make_base_env(split, config):
    module = importlib.import_module(
        f"yrc.environments.{get_global_variable('env_suite')}"
    )
    create_fn = getattr(module, "create_env")
    env = create_fn(split, config)
    env.suite = config.suite
    env.name = config.name
    return env


class CoordEnv(gym.Env):
    NOVICE = 0
    EXPERT = 1

    def __init__(
        self,
        config: "yrc.core.config.CoordinationConfig",
        base_env: "gym.Env",
        novice: "yrc.core.Policy",
        expert: "yrc.core.Policy",
    ) -> None:
        """
        Initializes the environment for coordination between novice and expert policies.
        This constructor sets up the environment with the provided configuration, base environment, and two policies (novice and expert). It defines the action and observation spaces, and initializes cost-related settings.

        Parameters
        ----------
        config : yrc.core.config.CoordinationConfig
            Configuration object specifying coordination parameters.
        base_env : gym.Env
            The base Gym environment to be wrapped or extended.
        novice : yrc.Policy
            The novice policy, expected to have a model with `hidden_dim` and `logit_dim` attributes.
        expert : yrc.Policy
            The expert policy.

        Returns
        -------
        None

        Raises
        ------
        None

        Examples
        --------
        >>> config = yrc.core.config.CoordinationConfig(...)
        >>> base_env = gym.make(...)
        >>> novice = yrc.policies.PPOPolicy(...)
        >>> expert = yrc.policies.PPOPolicy(...)
        >>> env = yrc.CoordEnv(config, base_env, novice, expert)
        """
        self.config = config
        self.base_env = base_env

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

        self.set_costs()

    @property
    def num_envs(self):
        return self.base_env.num_envs

    def set_costs(self, reward_per_action: float = None) -> None:
        """
        Sets the cost per action for expert queries and agent switching.

        If `reward_per_action` is not provided, it is computed from environment statistics
        loaded from a JSON metadata file. The mean episode reward is divided by the mean
        episode length to obtain the reward per action. The expert query and switch agent
        costs per action are then calculated by multiplying the reward per action by their
        respective alpha and beta coefficients from the configuration.

        Parameters
        ----------
        reward_per_action : float, optional
            The reward value per action. If None, it is computed from metadata.

        Side Effects
        ------------
        Sets `self.expert_query_cost_per_action` and `self.switch_agent_cost_per_action` attributes.
        """
        if reward_per_action is None:
            with open("yrc/metadata/test_eval_info.json") as f:
                data = json.load(f)
            test_eval_info = data[self.base_env.suite][self.base_env.name]
            mean_episode_reward = test_eval_info["reward_mean"]
            mean_episode_length = test_eval_info["episode_length_mean"]
            reward_per_action = mean_episode_reward / mean_episode_length

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

    def reset(self) -> dict:
        """
        Resets the coordination environment to an initial state.

        This method resets the base environment and both the novice and expert agents.
        It also clears the previous action history and returns the initial observation.

        Returns
        -------
        obs : dict
            The initial observation of the environment, including:
                - "env_obs": The initial observation from the base environment.
                - "novice_features": Numpy array of hidden features from the novice policy.
                - "novice_logit": Numpy array of output logits from the novice policy.

        Examples
        --------
        >>> obs = env.reset()
        >>> print(obs["env_obs"].shape)
        >>> print(obs["novice_features"].shape)
        >>> print(obs["novice_logit"].shape)
        """
        self.prev_action = None
        self.env_obs = self.base_env.reset()
        self.novice.model.eval()
        self.expert.model.eval()
        self._reset_agents(done=np.array([True] * self.num_envs))
        return self.get_obs()

    def _reset_agents(self, done: "numpy.ndarray") -> None:
        self.novice.reset(done)
        self.expert.reset(done)

    def step(
        self, action: "numpy.ndarray"
    ) -> Tuple[dict, "numpy.ndarray", "numpy.ndarray", List[dict]]:
        """
        Advances the environment by one step using the provided action.

        This method computes the environment-specific action, interacts with the base environment,
        processes the resulting observation, reward, and info, and returns the updated state.

        Parameters
        ----------
        action : numpy.ndarray
            The action(s) to take in the environment. Should be a numpy array indicating which agent acts.

        Returns
        -------
        obs : dict
            The next observation of the environment, including:
                - "env_obs": The initial observation from the base environment.
                - "novice_features": Numpy array of hidden features from the novice policy.
                - "novice_logit": Numpy array of output logits from the novice policy.
        reward : numpy.ndarray
            The reward(s) obtained from the environment after taking the action.
        done : numpy.ndarray
            Boolean flag(s) indicating whether the episode has ended for each environment.
        info : list of dict
            Additional information from the environment for each agent or environment instance.

        Raises
        ------
        Exception
            Propagates any exceptions raised by the underlying environment's `step` method.

        Examples
        --------
        >>> obs, reward, done, info = env.step(action)
        """
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

    @torch.no_grad()
    def _compute_env_action(self, action):
        # NOTE: this method only works with non-recurrent agent models
        is_novice = action == self.NOVICE
        is_expert = ~is_novice

        env_action = np.zeros_like(action)
        if is_novice.any():
            env_action[is_novice] = self.novice.act(
                self.env_obs[is_novice], greedy=self.config.act_greedy
            )
        if is_expert.any():
            env_action[is_expert] = self.expert.act(
                self.env_obs[is_expert], greedy=self.config.act_greedy
            )

        return env_action

    @torch.no_grad()
    def get_obs(self) -> dict:
        """
        Returns the current observation for the coordination environment.

        The observation includes:
        - The raw observation from the base environment (`env_obs`).
        - The hidden features from the novice policy (`novice_features`).
        - The output logits from the novice policy (`novice_logit`).

        Returns
        -------
        obs : dict
            A dictionary containing:
                - "env_obs": The current observation from the base environment.
                - "novice_features": Numpy array of hidden features from the novice policy.
                - "novice_logit": Numpy array of output logits from the novice policy.

        Examples
        --------
        >>> obs = env.get_obs()
        >>> print(obs["env_obs"].shape)
        >>> print(obs["novice_features"].shape)
        >>> print(obs["novice_logit"].shape)
        """
        self.novice.model(self.env_obs)
        obs = {
            "env_obs": self.env_obs,
            "novice_features": self.novice.model.hidden_features.detach().cpu().numpy(),
            "novice_logit": self.novice.model.logits.detach().cpu().numpy(),
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

    def close(self) -> None:
        """
        Closes the coordination environment and releases any resources held.

        This method calls the `close` method of the underlying base environment to ensure
        that all resources (such as external processes or files) are properly released.

        Returns
        -------
        None

        Examples
        --------
        >>> env.close()
        """
        return self.base_env.close()
