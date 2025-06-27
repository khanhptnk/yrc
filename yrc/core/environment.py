from copy import deepcopy as dc
from dataclasses import dataclass
from typing import List, Tuple

import gym
import numpy as np
import torch

import yrc


@dataclass
class CoordinationConfig:
    """
    Configuration for coordination environment parameters.

    Parameters
    ----------
    expert_query_cost_weight : float, optional
        The cost coefficient for querying the expert policy. Default is 0.4.
    switch_agent_cost_weight : float, optional
        The cost coefficient for switching between agents. Default is 0.0.
    temperature : float, optional
        The temperature parameter for action sampling. Default is 1.0.

    Attributes
    ----------
    expert_query_cost_weight : float
        The cost coefficient for querying the expert policy.
    switch_agent_cost_weight : float
        The cost coefficient for switching between agents.
    temperature : float
        The temperature parameter for action sampling.

    Examples
    --------
    >>> config = CoordinationConfig()
    """

    expert_query_cost_weight: float = 0.4
    switch_agent_cost_weight: float = 0.0
    temperature: float = 1.0


class CoordEnv(gym.Env):
    """
    Environment for coordinating between novice and expert policies.

    This class wraps a base environment and enables switching between a novice and expert policy,
    applying costs for expert queries and agent switching.

    Parameters
    ----------
    config : CoordinationConfig
        Configuration object specifying coordination parameters.
    base_env : gym.Env
        The base Gym environment to be wrapped or extended.
    novice : Policy
        The novice policy.
    expert : Policy
        The expert policy.

    Attributes
    ----------
    config : CoordinationConfig
        Coordination configuration.
    base_env : gym.Env
        The base environment.
    novice : Policy
        The novice policy.
    expert : Policy
        The expert policy.
    action_space : gym.spaces.Discrete
        Discrete action space (2: novice or expert).
    observation_space : gym.spaces.Dict
        Observation space including base_obs, novice_hidden, and novice_logits.
    expert_query_cost_per_action : float
        Cost per action for querying the expert.
    switch_agent_cost_per_action : float
        Cost per action for switching between agents.

    Examples
    --------
    >>> config = CoordinationConfig()
    >>> base_env = gym.make(...)
    >>> novice = ...
    >>> expert = ...
    >>> env = CoordEnv(config, base_env, novice, expert)
    """

    NOVICE = 0
    EXPERT = 1

    def __init__(self, config, base_env, novice, expert):
        """
        Initialize the coordination environment.

        Parameters
        ----------
        config : CoordinationConfig
            Configuration object specifying coordination parameters.
        base_env : gym.Env
            The base Gym environment to be wrapped or extended.
        novice : Policy
            The novice policy.
        expert : Policy
            The expert policy.

        Returns
        -------
        None

        Examples
        --------
        >>> config = CoordinationConfig(...)
        >>> base_env = gym.make(...)
        >>> novice = ...
        >>> expert = ...
        >>> env = CoordEnv(config, base_env, novice, expert)
        """
        self.config = config
        self.base_env = base_env

        self.novice = novice
        self.expert = expert

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "base_obs": base_env.observation_space,
                "novice_hidden": gym.spaces.Box(
                    -100, 100, shape=(novice.model.hidden_dim,)
                ),
                "novice_logits": gym.spaces.Box(
                    -100, 100, shape=(novice.model.logit_dim,)
                ),
            }
        )
        self.expert_query_cost_per_action = None
        self.switch_agent_cost_per_action = None

    @property
    def num_envs(self):
        """
        Number of parallel environments.

        Returns
        -------
        int
            Number of parallel environments.
        """
        return self.base_env.num_envs

    def set_costs(self, reward_per_action: float) -> None:
        """
        Set the cost per action for expert queries and agent switching.

        Parameters
        ----------
        reward_per_action : float
            The reward value per action. If None, it should be computed from environment
            statistics (mean episode reward divided by mean episode length).

        Returns
        -------
        None

        Examples
        --------
        >>> env.set_costs(0.05)
        >>> print(env.expert_query_cost_per_action)
        >>> print(env.switch_agent_cost_per_action)
        """

        # NOTE: paper results were generated with rounding
        # self.expert_query_cost_per_action = round(
        #     reward_per_action * self.config.expert_query_cost_weight, 2
        # )
        # self.switch_agent_cost_per_action = round(
        #     reward_per_action * self.config.switch_agent_cost_weight, 2
        # )

        self.expert_query_cost_per_action = (
            reward_per_action * self.config.expert_query_cost_weight
        )

        self.switch_agent_cost_per_action = (
            reward_per_action * self.config.switch_agent_cost_weight
        )

    def reset(self) -> dict:
        """
        Reset the coordination environment to an initial state.

        Returns
        -------
        obs : dict
            The initial observation of the environment, including:
                - "base_obs": The initial observation from the base environment.
                - "novice_hidden": Numpy array of hidden features from the novice policy.
                - "novice_logits": Numpy array of output logits from the novice policy.

        Examples
        --------
        >>> obs = env.reset()
        >>> print(obs["base_obs"].shape)
        >>> print(obs["novice_hidden"].shape)
        >>> print(obs["novice_logits"].shape)
        """
        self.prev_action = None
        self.base_obs = self.base_env.reset()
        self.novice.model.eval()
        self.expert.model.eval()
        self._reset_agents(done=np.array([True] * self.num_envs))
        return self._get_obs()

    def _reset_agents(self, done: "numpy.ndarray") -> None:
        """
        Reset the internal state of the novice and expert agents.

        Parameters
        ----------
        done : numpy.ndarray
            Boolean array indicating which episodes in a batch require a reset.

        Returns
        -------
        None
        """
        self.novice.reset(done)
        self.expert.reset(done)

    def step(
        self, action: "numpy.ndarray"
    ) -> Tuple[dict, "numpy.ndarray", "numpy.ndarray", List[dict]]:
        """
        Advance the environment by one step using the provided action.

        Parameters
        ----------
        action : numpy.ndarray
            The action(s) to take in the environment. Should be a numpy array indicating which agent acts.

        Returns
        -------
        obs : dict
            The next observation of the environment, including:
                - "base_obs": The initial observation from the base environment.
                - "novice_hidden": Numpy array of hidden features from the novice policy.
                - "novice_logits": Numpy array of output logits from the novice policy.
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
        base_action = self._compute_env_action(action)
        self.base_obs, base_reward, done, base_info = self.base_env.step(base_action)

        info = dc(base_info)
        for i, item in enumerate(info):
            if "base_reward" not in item:
                item["base_reward"] = base_reward[i]
            item["base_action"] = base_action[i]

        reward = self._get_reward(base_reward, action, done)
        self._reset_agents(done)
        self.prev_action = action

        return self._get_obs(), reward, done, info

    @torch.no_grad()
    def _compute_env_action(self, action):
        """
        Compute the environment-specific action for each agent.

        Parameters
        ----------
        action : numpy.ndarray
            Array indicating which agent (novice or expert) acts for each environment.

        Returns
        -------
        env_action : numpy.ndarray
            Array of actions to be passed to the base environment.
        """
        is_novice = action == self.NOVICE
        is_expert = ~is_novice

        env_action = np.zeros_like(action)
        if is_novice.any():
            env_action[is_novice] = (
                self.novice.act(
                    self.base_obs[is_novice], temperature=self.config.temperature
                )
                .cpu()
                .numpy()
            )
        if is_expert.any():
            env_action[is_expert] = (
                self.expert.act(
                    self.base_obs[is_expert], temperature=self.config.temperature
                )
                .cpu()
                .numpy()
            )

        return env_action

    @torch.no_grad()
    def _get_obs(self) -> dict:
        """
        Return the current observation for the coordination environment.

        Returns
        -------
        obs : dict
            A dictionary containing:
                - "base_obs": The current observation from the base environment.
                - "novice_hidden": Numpy array of hidden features from the novice policy.
                - "novice_logits": Numpy array of output logits from the novice policy.

        Examples
        --------
        >>> obs = env.get_obs()
        >>> print(obs["base_obs"].shape)
        >>> print(obs["novice_hidden"].shape)
        >>> print(obs["novice_logits"].shape)
        """
        # NOTE: novice model must be state-less
        model_output = self.novice.model(self.base_obs)
        obs = {
            "base_obs": self.base_obs,
            "novice_hidden": model_output.hidden.detach().cpu().numpy(),
            "novice_logits": model_output.logits.detach().cpu().numpy(),
        }
        return obs

    def _get_reward(self, base_reward, action, done):
        """
        Compute the reward for the current step, including costs for expert queries and agent switching.

        Parameters
        ----------
        base_reward : numpy.ndarray
            The base reward from the environment.
        action : numpy.ndarray
            The action(s) taken (novice or expert).
        done : numpy.ndarray
            Boolean flag(s) indicating whether the episode has ended for each environment.

        Returns
        -------
        reward : numpy.ndarray
            The computed reward(s) after applying costs.
        """
        # cost of querying expert agent
        reward = np.where(
            action == self.EXPERT,
            base_reward - self.expert_query_cost_per_action,
            base_reward,
        )

        # cost of switching
        if self.prev_action is not None:
            switch_indices = ((action != self.prev_action) & (~done)).nonzero()[0]
            if switch_indices.size > 1:
                reward[switch_indices] -= self.switch_agent_cost_per_action

        return reward

    def close(self) -> None:
        """
        Close the coordination environment and release any resources held.

        Returns
        -------
        None

        Examples
        --------
        >>> env.close()
        """
        return self.base_env.close()
