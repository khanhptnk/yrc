from abc import ABC, abstractmethod


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
    def train(self, *args, **kwargs) -> None:
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

        Examples
        --------
        >>> algo.train(data)
        """
        pass
