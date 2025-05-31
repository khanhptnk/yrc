import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical

import wandb
from yrc.core import Algorithm, CoordEnv
from yrc.utils.global_variables import get_global_variable
from yrc.utils.logging import configure_logging


@dataclass
class PPOAlgorithmConfig:
    cls: str = "PPOAlgorithm"
    log_freq: int = 10
    num_steps: int = 256
    total_timesteps: int = 1_500_000
    update_epochs: int = 3
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 0.0005
    critic_pretrain_steps: Optional[int] = None
    anneal_lr: Optional[bool] = False


class PPOAlgorithm(Algorithm):
    def __init__(self, config: "yrc.algorithms.PPOAlgorithm"):
        self.config = config

    def init(self, env):
        config = self.config
        self.num_envs = env.num_envs

        self.batch_size = int(self.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

        self.total_reward = {
            "reward": [0.0] * self.num_envs,
            "env_reward": [0.0] * self.num_envs,
        }

        self.buffer = TrainBuffer.new(env, config.num_steps)
        self.global_step = 0

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        envs: Dict[str, "gym.Env"],
        evaluator: "yrc.core.Evaluator",
        train_split: str = "train",
        eval_splits: List[str] = ["test"],
    ):
        """
        Trains the PPO algorithm on the specified environment(s) using the provided policy.

        This method performs multiple training iterations, periodically evaluates the policy
        on specified splits, logs statistics, and saves checkpoints for the best and last models.

        Parameters
        ----------
        policy : yrc.policies.PPOPolicy
            The policy to be trained. Must implement act(), train(), and eval() methods.
        envs : Dict[str, gym.Env]
            A dictionary mapping split names to environment instances.
        evaluator : yrc.core.Evaluator
            Evaluator object for policy evaluation and summary logging. Default is None.
        train_split : str, optional
            The environment split to use for training. Default is "train".
        eval_splits : List[str], optional
            List of environment splits to use for evaluation. Default is ["test"].

        Returns
        -------
        None

        Examples
        --------
        >>> algorithm.train(policy, envs, evaluator, train_split="train", eval_splits=["val", "test"])
        """
        self.init(envs[train_split])

        self.optim = optim.Adam(
            policy.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        # NOTE: weird bug, torch.optim messes up logging, so we need to reconfigure
        configure_logging(get_global_variable("log_file"))

        config = self.config
        save_dir = get_global_variable("experiment_dir")

        best_summary = {}
        for split in eval_splits:
            best_summary[split] = {"reward_mean": -1e9}

        train_log = {}

        for iteration in range(self.num_iterations):
            if iteration % config.log_freq == 0:
                wandb_log = {}
                wandb_log["step"] = self.global_step

                logging.info(f"Iteration {iteration}")
                if iteration > 0:
                    train_summary = self._summarize(train_log)
                    logging.info(f"Train {self.global_step} steps:")
                    self._write_summary(train_summary)
                    self._update_wandb_log(wandb_log, "train", train_summary)

                split_summary = evaluator.eval(policy, envs, eval_splits)
                for split in eval_splits:
                    if (
                        split_summary[split]["reward_mean"]
                        > best_summary[split]["reward_mean"]
                    ):
                        best_summary[split] = split_summary[split]
                        self.save_checkpoint(policy, save_dir, f"best_{split}")

                    logging.info(f"BEST {split} so far")
                    evaluator.summarizer.write(best_summary[split])

                    self._update_wandb_log(wandb_log, split, split_summary[split])
                    self._update_wandb_log(
                        wandb_log, f"best_{split}", best_summary[split]
                    )

                self.save_checkpoint(policy, save_dir, "last")

                wandb.log(wandb_log)

            this_train_log = self._train_one_iteration(
                iteration, policy, train_env=envs[train_split]
            )
            self._aggregate_log(train_log, this_train_log)

        # close env after training
        envs[train_split].close()

    def _train_one_iteration(self, iteration, policy, train_env=None, data_batch=None):
        config = self.config

        buffer = self.buffer

        device = get_global_variable("device")
        log = {}

        cp_steps = config.critic_pretrain_steps
        pretrain_critic = (
            cp_steps is not None and cp_steps > 0 and self.global_step < cp_steps
        )

        # annealing the rate if instructed to do so.
        lrnow = config.learning_rate
        if config.anneal_lr:
            lrnow *= 1 - self.global_step / config.total_timesteps
        self.optim.param_groups[0]["lr"] = lrnow
        log["lr"] = lrnow

        next_obs = ObsTensor.from_numpy(train_env.get_obs()).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        log["reward"], log["env_reward"] = [], []
        log[f"action_{CoordEnv.EXPERT}"] = []
        log["action_prob"] = []

        # NOTE: set policy to eval mode when collecting trajectories
        policy.eval()

        for step in range(config.num_steps):
            self.global_step += self.num_envs

            with torch.no_grad():
                action = policy.act(next_obs.data)
                log_prob = Categorical(logits=policy.model_output.logits).log_prob(
                    action
                )

            buffer.collect(
                step,
                {
                    "obs": next_obs,
                    "dones": next_done,
                    "values": policy.model_output.value,
                    "actions": action,
                    "log_probs": log_prob,
                },
            )

            log[f"action_{CoordEnv.EXPERT}"].extend(
                (action == CoordEnv.EXPERT).long().tolist()
            )
            log["action_prob"].extend(log_prob.exp().tolist())

            next_obs, reward, next_done, info = train_env.step(action.cpu().numpy())

            # keep track of episode reward
            for i in range(self.num_envs):
                self.total_reward["reward"][i] += reward[i]
                if "env_reward" in info[i]:
                    self.total_reward["env_reward"][i] += info[i]["env_reward"]
                if next_done[i]:
                    log["reward"].append(self.total_reward["reward"][i])
                    log["env_reward"].append(self.total_reward["env_reward"][i])
                    self.total_reward["reward"][i] = 0
                    self.total_reward["env_reward"][i] = 0

            buffer.collect(
                step, {"rewards": torch.from_numpy(reward).to(device).float().view(-1)}
            )

            next_obs = ObsTensor.from_numpy(next_obs).to(device)
            next_done = torch.from_numpy(next_done).to(device).float()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = policy.model(next_obs.data).value.reshape(1, -1)

            advantages = torch.zeros_like(buffer.rewards).to(device)

            last_gaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - buffer.dones[t + 1]
                    next_values = buffer.values[t + 1]
                delta = (
                    buffer.rewards[t]
                    + config.gamma * next_values * next_nonterminal
                    - buffer.values[t]
                )
                advantages[t] = last_gaelam = (
                    delta
                    + config.gamma * config.gae_lambda * next_nonterminal * last_gaelam
                )
            returns = advantages + buffer.values

        # flatten buffer
        buffer.set("advantages", advantages)
        buffer.set("returns", returns)
        buffer = buffer.flatten()

        b_inds = np.arange(self.batch_size)

        log["pg_loss"] = []
        log["v_loss"] = []
        log["ent_loss"] = []
        log["loss"] = []
        log["advantage"] = []
        log["value"] = []

        policy.train()

        for _ in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                action = buffer.actions[mb_inds]
                obs = buffer.obs[mb_inds]
                output = policy.model(obs.data)

                new_dist = Categorical(logits=output.logits)
                new_log_prob = new_dist.log_prob(action)
                entropy = new_dist.entropy()

                value = output.value

                log["value"].extend(value.tolist())

                logratio = new_log_prob - buffer.log_probs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = buffer.advantages[mb_inds]
                log["advantage"].extend(mb_advantages.tolist())
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (value - buffer.returns[mb_inds]) ** 2
                    v_clipped = buffer.values[mb_inds] + torch.clamp(
                        value - buffer.values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - buffer.returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((value - buffer.returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                if pretrain_critic:
                    loss = v_loss
                else:
                    loss = (
                        pg_loss
                        - config.ent_coef * entropy_loss
                        + v_loss * config.vf_coef
                    )
                loss.backward()

                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        policy.model.parameters(), config.max_grad_norm
                    )

                self.optim.step()
                self.optim.zero_grad()

                # log
                log["pg_loss"].append(pg_loss.item())
                log["v_loss"].append(v_loss.item())
                log["ent_loss"].append(entropy_loss.item())
                log["loss"].append(loss.item())

        return log

    def _aggregate_log(self, log, new_log):
        for k, v in new_log.items():
            if isinstance(v, list):
                if k not in log:
                    log[k] = v
                else:
                    log[k].extend(v)
            elif isinstance(v, float) or isinstance(v, int):
                log[k] = v
            else:
                raise NotImplementedError

    def _summarize(self, log):
        return {
            "lr": log["lr"],
            "reward_mean": float(np.mean(log["reward"])),
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            "pg_loss": float(np.mean(log["pg_loss"])),
            "v_loss": float(np.mean(log["v_loss"])),
            "ent_loss": float(np.mean(log["ent_loss"])),
            "loss": float(np.mean(log["loss"])),
            "advantage_mean": float(np.mean(log["advantage"])),
            "advantage_std": float(np.std(log["advantage"])),
            "value_mean": float(np.mean(log["value"])),
            "value_std": float(np.std(log["value"])),
            f"action_{CoordEnv.EXPERT}": float(
                np.mean(log[f"action_{CoordEnv.EXPERT}"])
            ),
            "action_prob": float(np.mean(log["action_prob"])),
        }

    def _write_summary(self, summary):
        log_str = "\n"
        log_str += "   Reward:     "
        log_str += (
            f"mean {summary['reward_mean']:7.2f} ± {summary['reward_std']:7.2f}\n"
        )
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_reward_mean']:7.2f} ± {summary['env_reward_std']:7.2f}\n"

        log_str += "   Loss:       "
        log_str += f"pg_loss {summary['pg_loss']:7.4f}  "
        log_str += f"v_loss {summary['v_loss']:7.4f}  "
        log_str += f"ent_loss {summary['ent_loss']:7.4f}  "
        log_str += f"loss {summary['loss']:7.4f}\n"

        log_str += "   Others:     "
        log_str += f"advantage {summary['advantage_mean']:7.4f} ± {summary['advantage_std']:7.4f}  "
        log_str += f"value {summary['value_mean']:7.4f} ± {summary['value_std']:7.4f}\n"

        log_str += f"   Action {CoordEnv.EXPERT} frac: {summary[f'action_{CoordEnv.EXPERT}']:7.2f}\n"
        log_str += f"   Action prob: {summary['action_prob']:7.2f}"

        logging.info(log_str)

        return summary

    def _update_wandb_log(self, wandb_log, split, summary):
        for k, v in summary.items():
            wandb_log[f"{split}/{k}"] = v

    def save_checkpoint(self, policy, save_dir, name):
        save_path = f"{save_dir}/{name}.ckpt"
        torch.save(
            {
                "policy_config": policy.config,
                "model_state_dict": policy.model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
                "global_step": self.global_step,
            },
            save_path,
        )
        logging.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, policy, load_path):
        ckpt = torch.load(load_path, map_location=get_global_variable("device"))
        policy.model.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optim_state_dict"])
        self.global_step = ckpt["global_step"]
        logging.info(
            f"Loaded checkpoint from {load_path}, global step: {self.global_step}"
        )


class TrainBuffer:
    def __init__(self, data_dict):
        self._fields = list(data_dict.keys())
        for k, v in data_dict.items():
            setattr(self, k, v)

    @classmethod
    def new(cls, env, num_steps):
        device = get_global_variable("device")
        num_envs = env.num_envs
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_shape = {
                k: space.shape for k, space in env.observation_space.spaces.items()
            }
        else:
            obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        if isinstance(obs_shape, dict):
            obs_buffer_shape = {
                k: (num_steps, num_envs) + shape for k, shape in obs_shape.items()
            }
        else:
            obs_buffer_shape = (num_steps, num_envs) + obs_shape

        data_dict = {}
        data_dict["obs"] = ObsTensor.zeros(obs_buffer_shape).to(device)
        data_dict["actions"] = torch.zeros((num_steps, num_envs) + action_shape).to(
            device
        )
        data_dict["log_probs"] = torch.zeros((num_steps, num_envs)).to(device)
        data_dict["rewards"] = torch.zeros((num_steps, num_envs)).to(device)
        data_dict["dones"] = torch.zeros((num_steps, num_envs)).to(device)
        data_dict["values"] = torch.zeros((num_steps, num_envs)).to(device)

        return cls(data_dict)

    def collect(self, step, data_dict):
        for k, v in data_dict.items():
            getattr(self, k)[step] = v

    def flatten(self):
        new_data = {}
        for k in self._fields:
            v = getattr(self, k)
            if k in ["obs", "actions"]:
                new_data[k] = v.flatten(0, 1)
            else:
                new_data[k] = v.flatten()
        return TrainBuffer(new_data)

    def set(self, name, value):
        setattr(self, name, value)
        self._fields.append(name)


class ObsTensor:
    def __init__(self, data):
        self.data = data

    @classmethod
    def zeros(cls, obs_shape):
        if isinstance(obs_shape, dict):
            data = {}
            for k, shape in obs_shape.items():
                data[k] = torch.zeros(shape)
        else:
            data = torch.zeros(obs_shape)
        return ObsTensor(data)

    def to(self, device):
        if isinstance(self.data, dict):
            data = {}
            for k in self.data:
                data[k] = self.data[k].to(device)
        else:
            data = self.data.to(device)
        return ObsTensor(data)

    def __setitem__(self, indices, next_obs):
        if isinstance(self.data, dict):
            for k in self.data:
                self.data[k][indices] = next_obs.data[k]
        else:
            self.data[indices] = next_obs.data

    def __getitem__(self, indices):
        if isinstance(self.data, dict):
            data = {}
            for k in self.data:
                data[k] = self.data[k][indices]
        else:
            data = self.data[indices]
        return ObsTensor(data)

    def flatten(self, start_dim=0, end_dim=-1):
        if isinstance(self.data, dict):
            data = {}
            for k in self.data:
                data[k] = self.data[k].flatten(start_dim, end_dim)
        else:
            data = self.data.flatten(start_dim, end_dim)
        return ObsTensor(data)

    @classmethod
    def from_numpy(cls, obs):
        if isinstance(obs, dict):
            data = {}
            for k in obs:
                data[k] = torch.from_numpy(obs[k]).float()
        else:
            data = torch.from_numpy(obs).float()
        return ObsTensor(data)
