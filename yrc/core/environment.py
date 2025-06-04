from copy import deepcopy as dc
from dataclasses import dataclass
from typing import List, Tuple

import gym
import numpy as np
import torch

import yrc


@dataclass
class CoordinationConfig:
    expert_query_cost_weight: float = 0.4
    switch_agent_cost_weight: float = 0.0
    temperature: float = 1.0


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
        novice : yrc.core.Policy
            The novice policy.
        expert : yrc.core.Policy
            The expert policy.

        Returns
        -------
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
        return self.base_env.num_envs

    def set_costs(self, reward_per_action: float) -> None:
        """
        Set the cost per action for expert queries and agent switching.

        The cost per action is determined by multiplying the provided `reward_per_action`
        by the expert query and switch agent cost coefficients (`expert_query_cost_weight` and
        `switch_agent_cost_weight`) from the environment configuration. These costs are used to
        penalize the agent for querying the expert or switching between agents.

        Parameters
        ----------
        reward_per_action : float
            The reward value per action. If None, it should be computed from environment
            statistics (mean episode reward divided by mean episode length).

        Side Effects
        ------------
        Sets the following attributes on the environment:
            - expert_query_cost_per_action : float
                The cost per action for querying the expert.
            - switch_agent_cost_per_action : float
                The cost per action for switching between agents.
        Calls `self.reset()` to apply the new cost settings.

        Notes
        -----
        This implementation currently supports only non-recurrent agent policies.

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
        Resets the coordination environment to an initial state.

        This method resets the base environment and both the novice and expert agents.
        It also clears the previous action history and returns the initial observation.

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

        return self.get_obs(), reward, done, info

    @torch.no_grad()
    def _compute_env_action(self, action):
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
    def get_obs(self) -> dict:
        """
        Returns the current observation for the coordination environment.

        The observation includes:
        - The raw observation from the base environment (`base_obs`).
        - The hidden features from the novice policy (`novice_hidden`).
        - The output logits from the novice policy (`novice_logits`).

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
