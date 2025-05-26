import logging
from dataclasses import dataclass
from typing import Optional

import gym
import numpy as np
import torch
import torch.optim as optim

import wandb
from yrc.core import Algorithm
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
    def __init__(self, config, env):
        self.config = config
        self.num_envs = env.num_envs

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.obs_shape = {
                k: space.shape for k, space in env.observation_space.spaces.items()
            }
        else:
            self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape

        self.batch_size = int(self.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

    def init(self, policy):
        config = self.config

        self.total_reward = {
            "reward": [0.0] * self.num_envs,
            "env_reward": [0.0] * self.num_envs,
        }

        device = get_global_variable("device")

        # initialize all tensors
        if isinstance(self.obs_shape, dict):
            self.obs = {}
            for k, shape in self.obs_shape.items():
                self.obs[k] = torch.zeros((config.num_steps, self.num_envs) + shape).to(
                    device
                )
        else:
            self.obs = torch.zeros(
                (config.num_steps, self.num_envs) + self.obs_shape
            ).to(device)
        self.actions = torch.zeros(
            (config.num_steps, self.num_envs) + self.action_shape
        ).to(device)
        self.logprobs = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((config.num_steps, self.num_envs)).to(device)

        self.global_step = 0

    def train(
        self,
        policy,
        envs,
        evaluator=None,
        train_split=None,
        eval_splits=None,
    ):
        self.init(policy)
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
                    train_summary = self.summarize(train_log)
                    logging.info(f"Train {self.global_step} steps:")
                    self.write_summary(train_summary)
                    self.update_wandb_log(wandb_log, "train", train_summary)

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

                    self.update_wandb_log(wandb_log, split, split_summary[split])
                    self.update_wandb_log(
                        wandb_log, f"best_{split}", best_summary[split]
                    )

                self.save_checkpoint(policy, save_dir, "last")

                wandb.log(wandb_log)

            this_train_log = self._train_one_iteration(
                iteration, policy, train_env=envs[train_split]
            )
            self.aggregate_log(train_log, this_train_log)

        # close env after training
        envs[train_split].close()

    def _train_one_iteration(self, iteration, policy, train_env=None, data_batch=None):
        config = self.config
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

        next_obs = self._wrap_obs(train_env.get_obs())
        next_done = torch.zeros(self.num_envs).to(device)

        log["reward"], log["env_reward"] = [], []
        log["action_1"] = []
        log["action_prob"] = []

        # NOTE: set policy to eval mode when collecting trajectories
        policy.model.eval()

        for step in range(0, config.num_steps):
            self.global_step += self.num_envs
            self._add_obs(step, next_obs)
            self.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            log["action_1"].extend((action == 1).long().tolist())
            log["action_prob"].extend(logprob.exp().tolist())

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

            self.rewards[step] = torch.from_numpy(reward).to(device).float().view(-1)
            next_obs, next_done = (
                self._wrap_obs(next_obs),
                torch.from_numpy(next_done).to(device).float(),
            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = policy.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = (
                    self.rewards[t]
                    + config.gamma * nextvalues * nextnonterminal
                    - self.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + self.values

        # flatten the batch
        b_obs = self._flatten_obs()
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        log["pg_loss"] = []
        log["v_loss"] = []
        log["ent_loss"] = []
        log["loss"] = []
        log["advantage"] = []
        log["value"] = []

        policy.model.train()

        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    self._slice_obs(b_obs, mb_inds), b_actions.long()[mb_inds]
                )

                log["value"].extend(newvalue.tolist())

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
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
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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

    def aggregate_log(self, log, new_log):
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

    def summarize(self, log):
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
            "action_1": float(np.mean(log["action_1"])),
            "action_prob": float(np.mean(log["action_prob"])),
        }

    def write_summary(self, summary):
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

        log_str += f"   Action 1 frac: {summary['action_1']:7.2f}\n"
        log_str += f"   Action prob: {summary['action_prob']:7.2f}"

        logging.info(log_str)

        return summary

    def update_wandb_log(self, wandb_log, split, summary):
        for k, v in summary.items():
            wandb_log[f"{split}/{k}"] = v

    def _wrap_obs(self, obs):
        device = get_global_variable("device")
        if isinstance(self.obs_shape, dict):
            ret = {}
            for k in self.obs_shape:
                ret[k] = torch.from_numpy(obs[k]).to(device).float()
            return ret
        return torch.from_numpy(obs).to(device).float()

    def _add_obs(self, step, next_obs):
        if isinstance(self.obs_shape, dict):
            for k in self.obs_shape:
                self.obs[k][step] = next_obs[k]
        else:
            self.obs[step] = next_obs

    def _flatten_obs(self):
        if isinstance(self.obs_shape, dict):
            ret = {}
            for k, shape in self.obs_shape.items():
                ret[k] = self.obs[k].reshape((-1,) + shape)
            return ret
        return self.obs.reshape((-1,) + self.obs_shape)

    def _slice_obs(self, b_obs, indices):
        if isinstance(self.obs_shape, dict):
            ret = {}
            for k in self.obs_shape:
                ret[k] = b_obs[k][indices]
            return ret
        return b_obs[indices]

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
