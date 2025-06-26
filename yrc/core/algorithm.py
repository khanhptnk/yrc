import importlib
from abc import ABC, abstractmethod


def make(config):
    """
    Instantiate and return an algorithm object based on the provided configuration.

    Parameters
    ----------
    config : object
        Configuration object with a 'cls' attribute specifying the algorithm class name.

    Returns
    -------
    Algorithm
        Instantiated algorithm object.

    Examples
    --------
    >>> algo = make(config)
    """
    algorithm = getattr(importlib.import_module("yrc.algorithms"), config.cls)(config)
    return algorithm


class Algorithm(ABC):
    """
    Abstract base class for all algorithms in the YRC framework.

    This class defines the interface that all algorithm implementations must follow.

    Examples
    --------
    >>> class MyAlgorithm(Algorithm):
    ...     def train(self, *args, **kwargs):
    ...         pass
    """

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model or algorithm using the provided arguments.

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
