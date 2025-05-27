import importlib
from abc import ABC, abstractmethod


def make(config, env):
    algorithm = getattr(importlib.import_module("yrc.algorithms"), config.cls)(
        config, env
    )
    return algorithm


class Algorithm:
    @abstractmethod
    def init(self, *args, **kwargs):
        """
        Initializes the object with the given arguments.

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
