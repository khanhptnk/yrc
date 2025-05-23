import importlib
import logging

if importlib.util.find_spec("gymnasium") is None:
    import gym
else:
    import gymnasium as gym  # used for minigrid
import json
import pprint
from copy import deepcopy as dc

import numpy as np

from YRC.core import Evaluator
from YRC.core.configs import get_global_variable


def make(config):
    base_envs = make_raw_envs(config)

    # single agent training
    if config.general.single_agent:
        return base_envs

    sim_novice_agent, novice_agent, expert_agent = load_agents(
        config, base_envs["val_sim"]
    )

    coord_envs = {}
    for split in base_envs:
        if config.general.skyline or split not in ["train", "val_sim"]:
            if type(expert_agent) is dict:
                coord_envs[split] = CoordEnv(
                    config.coord_env,
                    base_envs[split],
                    novice_agent,
                    expert_agent[split],
                )
            else:
                coord_envs[split] = CoordEnv(
                    config.coord_env, base_envs[split], novice_agent, expert_agent
                )
        else:
            # NOTE: not skyline and name in ["train", "val_sim"]
            # use novice agent as expert agent
            # use sim_novice agent as novice agent
            coord_envs[split] = CoordEnv(
                config.coord_env, base_envs[split], sim_novice_agent, novice_agent
            )

    # set costs for getting help from expert agent
    test_eval_info = get_test_eval_info(config, coord_envs)
    for split in coord_envs:
        coord_envs[split].set_costs(test_eval_info)

    # reset
    for split in coord_envs:
        coord_envs[split].reset()

    logging.info(
        f"Expert query cost per action: {coord_envs['train'].expert_query_cost_per_action}"
    )
    logging.info(
        f"Switch agent cost per action: {coord_envs['train'].switch_agent_cost_per_action}"
    )

    check_coord_envs(coord_envs)

    return coord_envs


def check_coord_envs(envs):
    for split in envs:
        assert (
            envs[split].expert_query_cost_per_action
            == envs["train"].expert_query_cost_per_action
        )
        assert (
            envs[split].switch_agent_cost_per_action
            == envs["train"].switch_agent_cost_per_action
        )


def get_test_eval_info(config, coord_envs):
    with open("YRC/core/test_eval_info.json") as f:
        data = json.load(f)

    backup_data = dc(data)

    benchmark = config.general.benchmark
    env_name = config.environment.common.env_name

    if env_name not in data[benchmark]:
        logging.info(f"Missing info about {benchmark}-{env_name}!")
        logging.info("Calculating missing info (taking a few minutes)...")
        evaluator = Evaluator(config.evaluation)
        # eval expert agent on test environment to get statistics
        summary = evaluator.eval(
            coord_envs["test"].expert_agent,
            {"test": coord_envs["test"].base_env},
            ["test"],
            num_episodes=coord_envs["test"].num_envs,
        )["test"]
        data[benchmark][env_name] = summary

        with open("YRC/core/backup_test_eval_info.json", "w") as f:
            json.dump(backup_data, f, indent=2)
        with open("YRC/core/test_eval_info.json", "w") as f:
            json.dump(data, f, indent=2)
        logging.info("Saved info!")

    ret = data[benchmark][env_name]

    logging.info(f"{pprint.pformat(ret, indent=2)}")
    return ret


def make_raw_envs(config):
    module = importlib.import_module(f"YRC.envs.{get_global_variable('benchmark')}")
    create_fn = getattr(module, "create_env")

    splits = config.environment.common.splits
    if splits is None:
        splits = ["train", "val_sim", "val_true", "test"]

    envs = {}
    for split in splits:
        env = create_fn(split, config.environment)
        env.name = config.environment.common.env_name
        envs[split] = env

    return envs


def load_agents(config, env):
    module = importlib.import_module(f"YRC.envs.{get_global_variable('benchmark')}")
    load_fn = getattr(module, "load_policy")

    sim_novice_agent = load_fn(config.agents.sim_novice, env)
    novice_agent = load_fn(config.agents.novice, env)
    expert_agent = load_fn(config.agents.expert, env)

    return sim_novice_agent, novice_agent, expert_agent


class CoordEnv(gym.Env):
    NOVICE = 0
    EXPERT = 1

    def __init__(self, config, base_env, novice_agent, expert_agent):
        self.args = config
        self.base_env = base_env
        if isinstance(base_env.observation_space, list):
            obs_space = base_env.observation_space[0]
        elif isinstance(base_env.observation_space, gym.spaces.Dict):
            obs_space = base_env.observation_space.spaces["image"]
        else:
            obs_space = base_env.observation_space
        self.novice_agent = novice_agent
        self.expert_agent = expert_agent

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "env_obs": obs_space,
                "novice_features": gym.spaces.Box(
                    -100, 100, shape=(novice_agent.hidden_dim,)
                ),
                "novice_logit": gym.spaces.Box(
                    -100, 100, shape=(novice_agent.model.logit_dim,)
                ),
            }
        )

    def set_costs(self, test_eval_info):
        length = test_eval_info["episode_length_mean"]
        reward = test_eval_info["reward_mean"]
        reward_per_action = reward / length

        self.expert_query_cost_per_action = round(
            reward_per_action * self.args.expert_query_cost_ratio, 2
        )
        self.switch_agent_cost_per_action = round(
            reward_per_action * self.args.switch_agent_cost_ratio, 2
        )

    @property
    def num_envs(self):
        return self.base_env.num_envs

    # @property
    # def num_actions(self):
    #     return self.action_space.n

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def obs_shape(self):
        return {
            "env_obs": self.base_env.obs_shape,
            "novice_features": (self.novice_agent.hidden_dim,),
            "novice_logit": (self.base_env.action_space.n,),
        }

    def reset(self):
        self.prev_action = None
        self.env_obs = self.base_env.reset()
        self._reset_agents(np.array([True] * self.num_envs))
        return self.get_obs()

    def _reset_agents(self, done):
        self.novice_agent.reset(done)
        self.expert_agent.reset(done)

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
        greedy = self.args.act_greedy
        is_novice = action == self.NOVICE
        is_expert = ~is_novice

        if isinstance(self.env_obs, dict):
            if is_novice.any():
                env_action = self.novice_agent.act(self.env_obs, greedy=greedy)
            if is_expert.any():
                if get_global_variable("benchmark") == "cliport":
                    env_action = self.expert_agent.act(
                        self.env_obs, self.base_env, greedy=greedy
                    )
                else:
                    env_action = self.expert_agent.act(self.env_obs, greedy=greedy)
        else:
            env_action = np.zeros_like(action)
            if is_novice.any():
                env_action[is_novice] = self.novice_agent.act(
                    self.env_obs[is_novice], greedy=greedy
                )
            if is_expert.any():
                env_action[is_expert] = self.expert_agent.act(
                    self.env_obs[is_expert], greedy=greedy
                )
        return env_action

    def get_obs(self):
        obs = {
            "env_obs": self.env_obs,
            "novice_features": self.novice_agent.get_hidden(self.env_obs)
            .detach()
            .cpu()
            .numpy(),
            "novice_logit": self.novice_agent.forward(self.env_obs)
            .detach()
            .cpu()
            .numpy(),
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
