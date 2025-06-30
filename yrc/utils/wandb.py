from typing import Any, Dict


class WandbLogger:
    """
    Simple logger for aggregating statistics and logging to Weights & Biases (wandb).

    Examples
    --------
    >>> logger = WandbLogger()
    >>> logger.clear()
    >>> logger['step'] = 1
    >>> logger.add('train', {'loss': 0.1, 'acc': 0.9})
    >>> stats = logger.get()
    """

    def clear(self) -> None:
        """
        Clear the logger's internal statistics dictionary.

        Returns
        -------
        None

        Examples
        --------
        >>> logger = WandbLogger()
        >>> logger.clear()
        """
        self.log = {}

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a statistic in the logger by key.

        Parameters
        ----------
        key : str
            The key for the statistic.
        value : Any
            The value to store.

        Returns
        -------
        None

        Examples
        --------
        >>> logger = WandbLogger()
        >>> logger['step'] = 1
        """
        self.log[key] = value

    def add(self, split: str, stats: Dict[str, Any]) -> None:
        """
        Add multiple statistics for a given split, prefixing keys with the split name.

        Parameters
        ----------
        split : str
            The split name (e.g., 'train', 'test').
        stats : dict
            Dictionary of statistics to add.

        Returns
        -------
        None

        Examples
        --------
        >>> logger = WandbLogger()
        >>> logger.add('train', {'loss': 0.1, 'acc': 0.9})
        """
        for k, v in stats.items():
            self.log[f"{split}/{k}"] = v

    def get(self) -> Dict[str, Any]:
        """
        Retrieve the current statistics dictionary.

        Returns
        -------
        dict
            The current statistics log.

        Examples
        --------
        >>> logger = WandbLogger()
        >>> logger.clear()
        >>> logger['step'] = 1
        >>> logger.add('train', {'loss': 0.1, 'acc': 0.9})
        >>> stats = logger.get()
        """
        return self.log
