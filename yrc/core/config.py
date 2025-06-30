import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

import wandb
import yrc.algorithms as algorithm_factory
import yrc.environments as env_factory
import yrc.policies as policy_factory
from yrc.core.environment import CoordinationConfig
from yrc.core.evaluator import EvaluatorConfig
from yrc.utils.global_variables import set_global_variable
from yrc.utils.logging import configure_logging


@dataclass
class YRCConfig:
    """
    Main configuration class for the YRC framework.

    This class holds all experiment-level configuration, including environment, policy, algorithm, evaluation, and coordination settings.

    Parameters
    ----------
    name : str, optional
        Name of the experiment. Default is "default".
    device : int, optional
        Device index for CUDA. Default is 0.
    seed : int, optional
        Random seed for reproducibility. Default is 10.
    env : Any, optional
        Environment configuration or name. Default is "procgen".
    policy : Any, optional
        Policy configuration or name. Default is "PPOPolicy".
    algorithm : Any, optional
        Algorithm configuration or name. Default is "PPOAlgorithm".
    evaluation : Any, optional
        Evaluation configuration. Default is None.
    eval_name : str, optional
        Name for evaluation run. Default is None.
    overwrite : bool, optional
        Whether to overwrite existing experiment directory. Default is False.
    use_wandb : bool, optional
        Whether to use Weights & Biases logging. Default is False.
    experiment_dir : str, optional
        Path to the experiment directory. Default is "".
    train_novice : str, optional
        Path to novice training checkpoint. Default is None.
    train_expert : str, optional
        Path to expert training checkpoint. Default is None.
    test_novice : str, optional
        Path to novice test checkpoint. Default is None.
    test_expert : str, optional
        Path to expert test checkpoint. Default is None.
    coordination : Any, optional
        Coordination configuration. Default is None.

    Examples
    --------
    >>> config = YRCConfig(name="my_experiment", env="procgen", policy="PPOPolicy")
    """

    name: str = "default"
    device: int = 0
    seed: int = 10
    env: Any = "procgen"
    policy: Any = "PPOPolicy"
    algorithm: Any = "PPOAlgorithm"
    evaluation: Any = None
    eval_name: Optional[str] = None
    overwrite: bool = False
    use_wandb: bool = False
    experiment_dir: str = ""

    train_novice: Optional[str] = None
    train_expert: Optional[str] = None
    test_novice: Optional[str] = None
    test_expert: Optional[str] = None

    evaluation: Any = None
    coordination: Any = None

    def __post_init__(self) -> None:
        """
        Post-initialization logic for YRCConfig.

        Converts string or dictionary fields for env, policy, algorithm, evaluation, and coordination
        into their respective configuration objects.

        Raises
        ------
        IndexError
            If required keys are missing in configuration dictionaries.
        ValueError
            If configuration fields are not of expected types.

        Examples
        --------
        >>> config = YRCConfig(env={"name": "procgen"})
        >>> config.__post_init__()
        """
        if isinstance(self.env, str):
            self.env = env_factory.registry[self.env]()
        elif isinstance(self.env, dict):
            if "name" not in self.env:
                raise IndexError("Please specify env.name through YAML file or flag")
            self.env = env_factory.registry[self.env["name"]](**self.env)
        else:
            raise ValueError("env must be a string or a dictionary")

        if isinstance(self.policy, str):
            self.policy = policy_factory.registry[self.policy].config_cls()
        elif isinstance(self.policy, dict):
            if "name" not in self.policy:
                raise IndexError("Please specify policy.name through YAML file or flag")
            self.policy = policy_factory.registry[self.policy["name"]].config_cls(
                **self.policy
            )
        else:
            raise ValueError("policy must be a string or a dictionary")

        if isinstance(self.algorithm, str):
            self.algorithm = algorithm_factory.registry[self.algorithm].config_cls()
        elif isinstance(self.algorithm, dict):
            if "name" not in self.algorithm:
                raise IndexError(
                    "Please specify algorithm.name through YAML file or flag"
                )
            self.algorithm = algorithm_factory.registry[
                self.algorithm["name"]
            ].config_cls(**self.algorithm)
        else:
            raise ValueError("algorithm must be a string or a dictionary")

        if self.evaluation is None:
            self.evaluation = EvaluatorConfig()
        elif isinstance(self.evaluation, dict):
            self.evaluation = EvaluatorConfig(**self.evaluation)
        else:
            raise ValueError("evaluation must be a dictionary or None")

        if self.coordination is None:
            self.coordination = CoordinationConfig()
        elif isinstance(self.coordination, dict):
            self.coordination = CoordinationConfig(**self.coordination)
        else:
            raise ValueError("coordination must be a dictionary or None")


def configure(config: YRCConfig) -> None:
    """
    Set up experiment directory, logging, random seeds, and global variables for the experiment.

    Parameters
    ----------
    config : YRCConfig
        The experiment configuration object.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the experiment directory exists and overwrite is not set.

    Examples
    --------
    >>> configure(config)
    """
    # set up experiment directory
    config.experiment_dir = "experiments/%s" % config.name

    if os.path.exists(config.experiment_dir):
        if config.eval_name is None and not config.overwrite:
            raise FileExistsError(
                f"Experiment directory {config.experiment_dir} exists! "
                "Set `overwrite=1` to overwrite it."
            )
    else:
        os.makedirs(config.experiment_dir)

    # reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config.eval_name is not None:
        log_file = os.path.join(config.experiment_dir, f"{config.eval_name}.eval.log")
    else:
        log_file = os.path.join(config.experiment_dir, f"{config.name}.train.log")

    if os.path.isfile(log_file):
        os.remove(log_file)

    # logging
    configure_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info("python -u " + " ".join(sys.argv))
    logging.info("Write log to %s" % log_file)
    logging.info(str(OmegaConf.to_yaml(config)))

    # configure wandb
    wandb.init(
        project="YRC",
        name=f"{config.name}_{str(int(time.time()))}",
        mode="online" if config.use_wandb else "disabled",
    )
    wandb.config.update(config)

    device = (
        torch.device(f"cuda:{config.device}") if torch.cuda.is_available() else "cpu"
    )

    # some useful global variables
    set_global_variable("device", device)
    set_global_variable("experiment_dir", config.experiment_dir)
    set_global_variable("seed", config.seed)
    set_global_variable("log_file", log_file)
