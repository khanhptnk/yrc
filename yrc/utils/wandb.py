class WandbLogger:
    """
    Simple logger for aggregating statistics and logging to Weights & Biases (wandb).

    Attributes
    ----------
    log : dict
        Dictionary storing logged statistics.

    Examples
    --------
    >>> logger = WandbLogger()
    >>> logger.clear()
    >>> logger['step'] = 1
    >>> logger.add('train', {'loss': 0.1, 'acc': 0.9})
    >>> stats = logger.get()
    """

    def clear(self):
        """
        Clear the logger's internal statistics dictionary.

        Returns
        -------
        None
        """
        self.log = {}

    def __setitem__(self, key, value):
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
        """
        self.log[key] = value

    def add(self, split, stats):
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
        """
        for k, v in stats.items():
            self.log[f"{split}/{k}"] = v

    def get(self):
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
