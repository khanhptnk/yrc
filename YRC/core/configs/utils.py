import logging
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import yaml

import wandb
from YRC.core.configs import ConfigDict
from YRC.core.configs.global_configs import set_global_variable


def load(yaml_file_or_str, flags_args=None):
    if yaml_file_or_str.endswith(".yaml"):
        with open(yaml_file_or_str) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = yaml.safe_load(yaml_file_or_str)

    with open("configs/common.yaml") as f:
        common_config = yaml.safe_load(f)
        common_config = ConfigDict(**common_config)

    config = ConfigDict(**config_dict)

    # fill missing arguments with common config
    algorithm = config.general.algorithm
    benchmark = config.general.benchmark
    if config.coord_policy is None:
        config.coord_policy = getattr(common_config.coord_policy, algorithm)
    if config.coord_env is None:
        config.coord_env = getattr(common_config.coord_env, benchmark)
    if config.evaluation is None:
        config.evaluation = getattr(common_config.evaluation, benchmark)
    if config.algorithm is None:
        config.algorithm = getattr(common_config.algorithm, algorithm)
    if config.environment is None:
        config.environment = getattr(common_config.environment, benchmark)

    # overwrite config with flags
    if flags_args is not None:
        config_dict = config.as_dict()
        update_config(flags_args.as_dict(), config_dict)
        config = ConfigDict(**config_dict)

    # config.environment.val_sim.env_name_suffix = (
    #     config.environment.train.env_name_suffix
    # )
    # config.environment.val_true.env_name_suffix = (
    #     config.environment.test.env_name_suffix
    # )

    # config.data_dir = os.getenv("SM_DATA_DIR", config.data_dir)
    # output_dir = os.getenv("SM_OUTPUT_DIR", "experiments")

    config.experiment_dir = "experiments/%s" % config.name

    if not config.eval_mode and not config.overwrite:
        try:
            os.makedirs(config.experiment_dir)
        except FileExistsError:
            raise FileExistsError(
                f"Experiment directory {config.experiment_dir} exists!"
            )

    # if not os.path.exists(config.experiment_dir):
    #     os.makedirs(config.experiment_dir)

    seed = config.general.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # config.random = random.Random(seed)

    config.general.device = torch.device("cuda", config.general.device)

    # set global variables
    set_global_variable("device", config.general.device)
    set_global_variable("benchmark", config.general.benchmark)
    set_global_variable("experiment_dir", config.experiment_dir)
    set_global_variable("seed", config.general.seed)

    config.start_time = time.time()

    # TODO: this is ugly :(
    if config.eval_mode:
        if config.file_name is None:
            log_file = os.path.join(config.experiment_dir, f"eval_seed_{seed}.log")
        elif config.file_name.__contains__("sim"):
            log_file = os.path.join(config.experiment_dir, f"eval_sim_seed_{seed}.log")
        elif config.file_name.__contains__("true"):
            log_file = os.path.join(config.experiment_dir, f"eval_true_seed_{seed}.log")
        assert not os.path.exists(log_file), f"Eval log file {log_file} already exists!"
    else:
        log_file = os.path.join(config.experiment_dir, "run.log")

    set_global_variable("log_file", log_file)

    if os.path.isfile(log_file):
        os.remove(log_file)

    # configure logging
    config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info("python -u " + " ".join(sys.argv))
    logging.info("Write log to %s" % log_file)
    logging.info(str(config))

    # configure wandb
    wandb.init(
        project="YRC",
        name=f"{config.name}_{str(int(time.time()))}",
        mode="online" if config.use_wandb else "disabled",
    )
    wandb.config.update(config)

    return config


def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]


def config_logging(log_file):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(
        level=logging.INFO, handlers=[file_handler, stream_handler], force=True
    )

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler


class ElapsedFormatter:
    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "[%s %s]: %s" % (elapsed_time, record.levelname, record.getMessage())
        return log_str
