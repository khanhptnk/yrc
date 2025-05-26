import importlib
import logging

import wandb
from yrc.utils.global_variables import get_global_variable


def make(config, env):
    algorithm = getattr(importlib.import_module("yrc.algorithms"), config.cls)(
        config, env
    )
    return algorithm


class Algorithm:
    def init(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwarg):
        raise NotImplementedError

    def _train_one_iteration(self, *args, **kwargs):
        raise NotImplementedError
