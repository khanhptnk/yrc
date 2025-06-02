import importlib
from abc import ABC, abstractmethod


def make(config):
    algorithm = getattr(importlib.import_module("yrc.algorithms"), config.cls)(config)
    return algorithm


class Algorithm(ABC):
    @abstractmethod
    def train(self, *args, **kwarg):
        """
        Trains the model or algorithm using the provided arguments.

        Parameters
        ----------
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        pass
