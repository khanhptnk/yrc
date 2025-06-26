import logging
from dataclasses import dataclass
from typing import Dict

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical

import wandb
from yrc.core import Algorithm
from yrc.utils.global_variables import get_global_variable
from yrc.utils.logging import configure_logging
from yrc.utils.wandb import WandbLogger


@dataclass
class PPOAlgorithmConfig:
    """
    Configuration dataclass for PPOAlgorithm.

    Contains all hyperparameters and settings required to initialize and run PPO training.

    Parameters
    ----------
    cls : str, optional
        Name of the algorithm class (default: "PPOAlgorithm").
    log_freq : int, optional
        Frequency (in iterations) to log training statistics.
    save_freq : int, optional
        Frequency (in iterations) to save model checkpoints.
    num_steps : int, optional
        Number of steps to run in each environment per iteration.
    total_timesteps : int, optional
        Total number of environment steps to train for.
    update_epochs : int, optional
        Number of epochs to update the policy per iteration.
    gamma : float, optional
        Discount factor for rewards.
    gae_lambda : float, optional
        Lambda for Generalized Advantage Estimation.
    num_minibatches : int, optional
        Number of minibatches for each update epoch.
    clip_coef : float, optional
        Clipping coefficient for PPO surrogate objective.
    norm_adv : bool, optional
        Whether to normalize advantages.
    clip_vloss : bool, optional
        Whether to use clipped value loss.
    vf_coef : float, optional
        Coefficient for value function loss.
    ent_coef : float, optional
        Coefficient for entropy bonus.
    max_grad_norm : float, optional
        Maximum norm for gradient clipping.
    learning_rate : float, optional
        Learning rate for optimizer.
    critic_pretrain_steps : int, optional
        Number of steps to pretrain the critic before policy updates.
    anneal_lr : bool, optional
        Whether to linearly anneal the learning rate.
    log_action_id : int, optional
        Action ID to log statistics for (e.g., expert action).
    """

    cls: str = "PPOAlgorithm"
    log_freq: int = 10
    save_freq: int = 0
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
    critic_pretrain_steps: int = 0
    anneal_lr: bool = False
    log_action_id: int = 1


class PPOAlgorithm(Algorithm):
    def __init__(self, config):
        """
        Initialize the PPOAlgorithm.

        Parameters
        ----------
        config : PPOAlgorithmConfig
            Configuration object containing PPO hyperparameters.
        """
        self.config = config

    def _initialize(self):
        config = self.config
        env = self.env
        policy = self.policy

        self.num_envs = env.num_envs
        self.batch_size = int(self.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

        self.save_dir = get_global_variable("experiment_dir")

        self.buffer = TrainBuffer.new(env, config.num_steps)
        self.global_step = 0
        self.summarizer = PPOTrainSummarizer(config)

        self.optim = optim.Adam(
            policy.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        # NOTE: weird bug, torch.optim messes up logging, so we need to reconfigure
        configure_logging(get_global_variable("log_file"))
        self.wandb_logger = WandbLogger()

        env.reset()

    def train(
        self,
        policy: "yrc.policies.PPOPolicy",
        env: "gym.Env",
        validators: Dict[str, "yrc.core.Evaluator"],
    ):
        """
        Train the PPO algorithm on the specified environment(s) using the provided policy.

        This method performs multiple training iterations, periodically evaluates the policy
        on specified splits, logs statistics, and saves checkpoints for the best and last models.

        Parameters
        ----------
        policy : yrc.policies.PPOPolicy
            The policy to be trained. Must implement act(), train(), and eval() methods.
        env : gym.Env
            The environment instance for training.
        validators : dict of str to yrc.core.Evaluator
            Dictionary mapping split names to evaluator instances for evaluation.

        Returns
        -------
        None

        Examples
        --------
        >>> algorithm.train(policy, env, validators)
        """

        config = self.config
        self.env = env
        self.policy = policy

        self._initialize()

        best_result = {split: {"reward_mean": -float("inf")} for split in validators}

        for iteration in range(self.num_iterations):
            # save checkpoint
            if config.save_freq > 0 and iteration % config.save_freq == 0:
                self.save_checkpoint(policy, f"iter_{iteration}")

            # evaluation
            if iteration % config.log_freq == 0:
                if iteration > 0:
                    logging.info(f"Iteration {iteration}")
                    logging.info(f"Train {self.global_step} steps:")
                    train_summary = self.summarizer.write()

                self.save_checkpoint(policy, "last")

                eval_result = {}
                for split, validator in validators.items():
                    logging.info(f"Evaluating on {split} split")
                    eval_result[split] = validator.evaluate(policy)
                    if (
                        eval_result[split]["reward_mean"]
                        > best_result[split]["reward_mean"]
                    ):
                        best_result[split] = eval_result[split]
                        self.save_checkpoint(policy, f"best_{split}")

                for split, validator in validators.items():
                    logging.info(f"BEST {split} so far")
                    validator.summarizer.write(best_result[split])

                # wandb logging
                self.wandb_logger.clear()
                self.wandb_logger.log["step"] = self.global_step
                if iteration > 0:
                    self.wandb_logger.add("train", train_summary)
                for split in validators:
                    self.wandb_logger.add(split, eval_result[split])
                    self.wandb_logger.add(f"best_{split}", best_result[split])
                wandb.log(self.wandb_logger.get())

            # training
            self._train_once()

        # close env after training
        env.close()

    def _train_once(self):
        """
        Perform a single training iteration of PPO, including trajectory collection,
        advantage computation, and policy/value updates.

        Returns
        -------
        None
        """
        config = self.config
        env = self.env
        policy = self.policy
        buffer = self.buffer
        device = get_global_variable("device")

        self.summarizer.initialize_iteration(env)

        # NOTE: set policy to eval mode when collecting trajectories
        policy.eval()

        next_done = np.zeros((self.num_envs,))
        next_obs = env.get_obs()

        for step in range(config.num_steps):
            self.global_step += self.num_envs

            done = torch.from_numpy(next_done).to(device).float()
            obs = TensorDict.from_numpy(next_obs).to(device)

            with torch.no_grad():
                action, cur_model_output = policy.act(
                    obs.data, return_model_output=True
                )
                log_prob = Categorical(logits=cur_model_output.logits).log_prob(action)

            next_obs, reward, next_done, info = env.step(action.cpu().numpy())

            # Correctly handling boostrapping for truncation
            # Issue: https://github.com/DLR-RM/stable-baselines3/issues/633
            # Solution: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L234
            for i, d in enumerate(done):
                if (
                    d
                    and info[i].get("terminal_observation") is not None
                    and info[i].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = TensorDict.from_numpy(
                        info[i]["terminal_observation"]
                    ).to(device)
                    with torch.no_grad():
                        terminal_value = policy.model(terminal_obs).value
                    reward[i] += config.gamma * terminal_value

            buffer.add(
                step,
                {
                    "obs": obs,
                    "actions": action,
                    "dones": done,
                    "values": cur_model_output.value,
                    "log_probs": log_prob,
                    "rewards": torch.from_numpy(reward).to(device).float(),
                },
            )

            self.summarizer.add_episode_step(
                action,
                log_prob,
                reward,
                next_done,
                info,
            )

        # NOTE: don't forget to add done and value of last step
        done = torch.from_numpy(next_done).to(device).float()
        obs = TensorDict.from_numpy(next_obs).to(device)
        with torch.no_grad():
            buffer.add(step, {"dones": done, "values": policy.model(obs.data).value})

        # conpute returns and advantages
        buffer["advantages"], buffer["returns"] = self._compute_advantages_and_returns()

        # flatten buffer
        buffer = buffer.flatten()

        # NOTE: set policy to training mode
        policy.train()

        self._update_learning_rate()

        for mb in buffer.generate_minibatches(
            config.update_epochs, self.minibatch_size
        ):
            cur_model_output = policy.model(mb.obs.data)
            cur_dist = Categorical(logits=cur_model_output.logits)

            ref_log_prob = mb.log_probs
            cur_log_prob = cur_dist.log_prob(mb.actions)
            ratio = (cur_log_prob - ref_log_prob).exp()

            adv = mb.advantages
            if config.norm_adv:
                adv = (mb.advantages - mb.advantages.mean()) / (
                    mb.advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(
                ratio, 1 - config.clip_coef, 1 + config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            ref_value = mb.values
            cur_value = cur_model_output.value
            if config.clip_vloss:
                v_loss_unclipped = (cur_value - mb.returns) ** 2
                v_clipped = ref_value + torch.clamp(
                    cur_value - ref_value,
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb.returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((cur_value - mb.returns) ** 2).mean()

            entropy_loss = -cur_dist.entropy().mean()

            if self.global_step < config.critic_pretrain_steps:
                loss = v_loss
            else:
                loss = (
                    pg_loss + config.vf_coef * v_loss + config.ent_coef * entropy_loss
                )

            loss.backward()

            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    policy.model.parameters(), config.max_grad_norm
                )

            self.optim.step()
            self.optim.zero_grad()

            self.summarizer.add_training_iteration(
                cur_value,
                adv,
                pg_loss,
                v_loss,
                entropy_loss,
                loss,
            )

        self.summarizer.finalize_iteration()

    def _update_learning_rate(self):
        """
        Update the learning rate for the optimizer, optionally annealing it over time.

        Returns
        -------
        None
        """
        config = self.config
        lrnow = config.learning_rate
        if config.anneal_lr:
            lrnow *= 1 - self.global_step / config.total_timesteps
        self.optim.param_groups[0]["lr"] = lrnow
        # kepp track of the learning rate in the summarizer
        self.summarizer.log["lr"] = lrnow

    def _compute_advantages_and_returns(self):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).

        Returns
        -------
        advantages : torch.Tensor
            Advantage estimates for each step.
        returns : torch.Tensor
            Computed returns for each step.
        """
        buffer = self.buffer
        config = self.config

        advantages = torch.zeros_like(buffer.rewards)

        last_gaelam = 0
        for t in reversed(range(config.num_steps)):
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
        returns = advantages + buffer.values[:-1]
        return advantages, returns

    def save_checkpoint(self, policy, name):
        """
        Save the current policy and optimizer state to a checkpoint file.

        Parameters
        ----------
        policy : PPOPolicy
            The policy to save.
        name : str
            Name for the checkpoint file.

        Returns
        -------
        None
        """
        save_path = f"{self.save_dir}/{name}.ckpt"
        torch.save(
            {
                "policy_config": policy.config,
                "model_state_dict": policy.get_params(),
                "optim_state_dict": self.optim.state_dict(),
                "global_step": self.global_step,
            },
            save_path,
        )
        logging.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, policy, load_path):
        """
        Load policy and optimizer state from a checkpoint file.

        Parameters
        ----------
        policy : PPOPolicy
            The policy to load parameters into.
        load_path : str
            Path to the checkpoint file.

        Returns
        -------
        None
        """
        ckpt = torch.load(load_path, map_location=get_global_variable("device"))
        policy.set_params(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optim_state_dict"])
        self.global_step = ckpt["global_step"]
        logging.info(
            f"Loaded checkpoint from {load_path}, global step: {self.global_step}"
        )


@dataclass
class PPOBatch:
    """
    Data structure for a batch of PPO training data.

    Attributes
    ----------
    obs : TensorDict
        Batch of observations.
    actions : torch.Tensor
        Batch of actions taken.
    log_probs : torch.Tensor
        Log probabilities of the actions.
    advantages : torch.Tensor
        Advantage estimates for the batch.
    returns : torch.Tensor
        Computed returns for the batch.
    values : torch.Tensor
        Value function predictions for the batch.
    """

    obs: "TensorDict"
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class TrainBuffer:
    """
    Buffer for storing trajectories and training data for PPO.

    Parameters
    ----------
    data : dict
        Dictionary containing buffer arrays for each key (e.g., obs, actions, rewards).

    Attributes
    ----------
    data : dict
        Dictionary containing buffer arrays for each key.

    Examples
    --------
    >>> buffer = TrainBuffer.new(env, num_steps=128)
    """

    def __init__(self, data):
        """
        Initialize the TrainBuffer.

        Parameters
        ----------
        data : dict
            Dictionary containing buffer arrays for each key.
        """
        self.data = data

    def __getattr__(self, name):
        """
        Retrieve a buffer attribute by key.

        Parameters
        ----------
        name : str
            Name of the buffer key to retrieve.

        Returns
        -------
        value : Any
            The buffer value for the given key.

        Raises
        ------
        AttributeError
            If the key is not found in the buffer.
        """
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'TrainBuffer' object has no attribute '{name}'")

    @classmethod
    def new(cls, env, num_steps):
        """
        Create a new TrainBuffer with zero-initialized arrays for the given environment and number of steps.

        Parameters
        ----------
        env : gym.Env
            The environment instance.
        num_steps : int
            Number of steps to allocate in the buffer.

        Returns
        -------
        TrainBuffer
            A new buffer instance with allocated arrays.
        """
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

        new_data = {}
        new_data["obs"] = TensorDict.zeros(obs_buffer_shape).to(device)
        new_data["actions"] = torch.zeros((num_steps, num_envs) + action_shape).to(
            device
        )
        new_data["log_probs"] = torch.zeros((num_steps, num_envs)).to(device)
        new_data["rewards"] = torch.zeros((num_steps, num_envs)).to(device)
        new_data["dones"] = torch.zeros((num_steps + 1, num_envs)).to(device)
        new_data["values"] = torch.zeros((num_steps + 1, num_envs)).to(device)

        return cls(new_data)

    def add(self, step, new_data):
        """
        Add new data for a given step to the buffer.

        Parameters
        ----------
        step : int
            The step index to add data to.
        new_data : dict
            Dictionary of new data to add for this step.

        Returns
        -------
        None
        """
        for k, v in new_data.items():
            assert k in self.data, f"Key {k} not found in buffer"
            self.data[k][step] = v

    def flatten(self):
        """
        Flatten the buffer for minibatch training.

        Returns
        -------
        TrainBuffer
            A new buffer with flattened arrays.
        """
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = v.flatten(0, 1)
        return TrainBuffer(new_data)

    def __setitem__(self, name, value):
        """
        Set a buffer value by key.

        Parameters
        ----------
        name : str
            Name of the buffer key to set.
        value : Any
            Value to assign to the buffer key.

        Returns
        -------
        None
        """
        self.data[name] = value

    def generate_minibatches(self, num_epochs, minibatch_size):
        """
        Yield minibatches for training.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to iterate over the buffer.
        minibatch_size : int
            Size of each minibatch.

        Yields
        ------
        PPOBatch
            A minibatch of PPO training data.
        """
        batch_size = self.actions.shape[0]
        b_inds = np.arange(batch_size)
        for _ in range(num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                yield PPOBatch(
                    obs=self.obs[mb_inds],
                    actions=self.actions[mb_inds],
                    log_probs=self.log_probs[mb_inds],
                    advantages=self.advantages[mb_inds],
                    returns=self.returns[mb_inds],
                    values=self.values[mb_inds],
                )


class TensorDict:
    """
    Utility class for handling dictionary-structured tensors, supporting batch operations.

    Parameters
    ----------
    data : dict or torch.Tensor
        Dictionary of tensors or a single tensor.

    Attributes
    ----------
    data : dict or torch.Tensor
        The underlying tensor data.

    Examples
    --------
    >>> td = TensorDict({'obs': torch.zeros(4, 3)})
    """

    def __init__(self, data):
        """
        Initialize a TensorDict.

        Parameters
        ----------
        data : dict or torch.Tensor
            Dictionary of tensors or a single tensor.
        """
        self.data = data

    @classmethod
    def zeros(cls, shape):
        """
        Create a TensorDict of zeros with the given shape.

        Parameters
        ----------
        shape : dict or tuple
            Shape for each tensor or the single tensor.

        Returns
        -------
        TensorDict
            A TensorDict of zeros.
        """
        if isinstance(shape, dict):
            data = {}
            for k, shape in shape.items():
                data[k] = torch.zeros(shape)
        else:
            data = torch.zeros(shape)
        return TensorDict(data)

    def to(self, device):
        """
        Move all tensors in the TensorDict to the specified device.

        Parameters
        ----------
        device : torch.device or str
            The device to move tensors to.

        Returns
        -------
        TensorDict
            A new TensorDict with tensors on the specified device.
        """
        if isinstance(self.data, dict):
            data = {}
            for k in self.data:
                data[k] = self.data[k].to(device)
        else:
            data = self.data.to(device)
        return TensorDict(data)

    def __setitem__(self, indices, other):
        """
        Set values in the TensorDict at the given indices.

        Parameters
        ----------
        indices : Any
            Indices to set.
        other : TensorDict
            TensorDict containing values to set.

        Returns
        -------
        None
        """
        if isinstance(self.data, dict):
            for k in self.data:
                self.data[k][indices] = other.data[k]
        else:
            self.data[indices] = other.data

    def __getitem__(self, indices):
        """
        Retrieve values from the TensorDict at the given indices.

        Parameters
        ----------
        indices : Any
            Indices to retrieve.

        Returns
        -------
        TensorDict
            A new TensorDict with the selected values.
        """
        if isinstance(self.data, dict):
            data = {}
            for k in self.data:
                data[k] = self.data[k][indices]
        else:
            data = self.data[indices]
        return TensorDict(data)

    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten tensors in the TensorDict along specified dimensions.

        Parameters
        ----------
        start_dim : int, optional
            The first dimension to flatten. Default is 0.
        end_dim : int, optional
            The last dimension to flatten. Default is -1.

        Returns
        -------
        TensorDict
            A new TensorDict with flattened tensors.
        """
        if isinstance(self.data, dict):
            data = {}
            for k in self.data:
                data[k] = self.data[k].flatten(start_dim, end_dim)
        else:
            data = self.data.flatten(start_dim, end_dim)
        return TensorDict(data)

    @classmethod
    def from_numpy(cls, data):
        """
        Convert numpy arrays to a TensorDict.

        Parameters
        ----------
        data : dict or np.ndarray
            Dictionary of numpy arrays or a single numpy array.

        Returns
        -------
        TensorDict
            A TensorDict with tensors converted from numpy arrays.
        """
        if isinstance(data, dict):
            data = data.copy()
            for k in data:
                data[k] = torch.from_numpy(data[k]).float()
        else:
            data = torch.from_numpy(data).float()
        return TensorDict(data)


class PPOTrainSummarizer:
    """
    Summarizer for PPO training statistics and logging.

    Parameters
    ----------
    config : PPOAlgorithmConfig
        Configuration object for the summarizer.

    Attributes
    ----------
    log_action_id : int
        Action ID to log statistics for.
    log : dict
        Dictionary for storing summary statistics.

    Examples
    --------
    >>> summarizer = PPOTrainSummarizer(config)
    """

    def __init__(self, config):
        """
        Initialize the PPOTrainSummarizer.

        Parameters
        ----------
        config : PPOAlgorithmConfig
            Configuration object for the summarizer.
        """
        self.log_action_id = config.log_action_id
        self.clear()

    def clear(self):
        """
        Clear the summary statistics log.

        Returns
        -------
        None
        """
        self.log = {}

    def initialize_iteration(self, env):
        """
        Initialize logging for a new training iteration.

        Parameters
        ----------
        env : gym.Env
            The environment instance for the iteration.

        Returns
        -------
        None
        """
        keys = [
            "reward",
            "base_reward",
            f"action_{self.log_action_id}",
            "action_prob",
            "pg_loss",
            "v_loss",
            "ent_loss",
            "loss",
            "advantage",
            "value",
        ]

        self.iter_log = {k: [] for k in keys}

        self.episode_total_reward = {
            "reward": [0.0] * env.num_envs,
            "base_reward": [0.0] * env.num_envs,
        }

    def finalize_iteration(self):
        """
        Finalize and aggregate statistics for the iteration.

        Returns
        -------
        None
        """
        for k, v in self.iter_log.items():
            if isinstance(v, list):
                self.log.setdefault(k, []).extend(v)
            else:
                raise NotImplementedError

    def add_episode_step(self, action, log_prob, reward, done, info):
        """
        Log statistics for each episode step.

        Parameters
        ----------
        action : torch.Tensor
            Actions taken at this step.
        log_prob : torch.Tensor
            Log probabilities of the actions.
        reward : np.ndarray or torch.Tensor
            Rewards received at this step.
        done : np.ndarray or torch.Tensor
            Done flags for each environment.
        info : list of dict
            Additional info for each environment.

        Returns
        -------
        None
        """
        self.iter_log[f"action_{self.log_action_id}"].extend(
            (action == self.log_action_id).long().tolist()
        )
        self.iter_log["action_prob"].extend(log_prob.exp().tolist())
        for i in range(action.shape[0]):
            self.episode_total_reward["reward"][i] += reward[i]
            if "base_reward" in info[i]:
                self.episode_total_reward["base_reward"][i] += info[i]["base_reward"]
            if done[i]:
                self.iter_log["reward"].append(self.episode_total_reward["reward"][i])
                self.iter_log["base_reward"].append(
                    self.episode_total_reward["base_reward"][i]
                )
                self.episode_total_reward["reward"][i] = 0
                self.episode_total_reward["base_reward"][i] = 0

    def add_training_iteration(
        self, value, advantage, pg_loss, v_loss, entropy_loss, loss
    ):
        """
        Log statistics for each training minibatch.

        Parameters
        ----------
        value : torch.Tensor
            Value function predictions.
        advantage : torch.Tensor
            Advantage estimates.
        pg_loss : torch.Tensor
            Policy gradient loss.
        v_loss : torch.Tensor
            Value loss.
        entropy_loss : torch.Tensor
            Entropy loss.
        loss : torch.Tensor
            Total loss.

        Returns
        -------
        None
        """
        self.iter_log["value"].extend(value.tolist())
        self.iter_log["advantage"].extend(advantage.tolist())
        self.iter_log["pg_loss"].append(pg_loss.item())
        self.iter_log["v_loss"].append(v_loss.item())
        self.iter_log["ent_loss"].append(entropy_loss.item())
        self.iter_log["loss"].append(loss.item())

    def summarize(self):
        """
        Compute summary statistics for the current log.

        Returns
        -------
        dict
            Dictionary of summary statistics.
        """
        log = self.log
        return {
            "lr": log["lr"],
            "reward_mean": float(np.mean(log["reward"])),
            "reward_std": float(np.std(log["reward"])),
            "base_reward_mean": float(np.mean(log["base_reward"])),
            "base_reward_std": float(np.std(log["base_reward"])),
            "pg_loss": float(np.mean(log["pg_loss"])),
            "v_loss": float(np.mean(log["v_loss"])),
            "ent_loss": float(np.mean(log["ent_loss"])),
            "loss": float(np.mean(log["loss"])),
            "advantage_mean": float(np.mean(log["advantage"])),
            "advantage_std": float(np.std(log["advantage"])),
            "value_mean": float(np.mean(log["value"])),
            "value_std": float(np.std(log["value"])),
            f"action_{self.log_action_id}": float(
                np.mean(log[f"action_{self.log_action_id}"])
            ),
            "action_prob": float(np.mean(log["action_prob"])),
        }

    def write(self, summary=None):
        """
        Pretty-print and log the summary statistics.

        Parameters
        ----------
        summary : dict, optional
            Precomputed summary statistics. If None, will compute from log.

        Returns
        -------
        dict
            The summary statistics that were logged.
        """
        if summary is None:
            summary = self.summarize()

        log_str = (
            "\n"
            f"   Reward:      mean {summary['reward_mean']:7.2f} ± {summary['reward_std']:7.2f}\n"
            f"   Base Reward: mean {summary['base_reward_mean']:7.2f} ± {summary['base_reward_std']:7.2f}\n"
            f"   Loss:        pg_loss {summary['pg_loss']:7.4f}  "
            f"v_loss {summary['v_loss']:7.4f}  "
            f"ent_loss {summary['ent_loss']:7.4f}  "
            f"loss {summary['loss']:7.4f}\n"
            f"   Others:      advantage {summary['advantage_mean']:7.4f} ± {summary['advantage_std']:7.4f}  "
            f"value {summary['value_mean']:7.4f} ± {summary['value_std']:7.4f}\n"
            f"   Action {self.log_action_id} frac: {summary[f'action_{self.log_action_id}']:7.2f}\n"
            f"   Action prob: {summary['action_prob']:7.2f}"
        )
        logging.info(log_str)
        return summary
