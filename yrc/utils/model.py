import torch.nn as nn


def orthogonal_init(module, gain=nn.init.calculate_gain("relu")):
    """
    Apply orthogonal initialization to a module's weights and zero to biases.

    Parameters
    ----------
    module : nn.Module
        The module to initialize (e.g., nn.Linear, nn.Conv2d).
    gain : float, optional
        Scaling factor for the orthogonal initialization. Default is relu gain.

    Returns
    -------
    nn.Module
        The initialized module.

    Examples
    --------
    >>> layer = nn.Linear(10, 5)
    >>> orthogonal_init(layer)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    """
    Apply Xavier uniform initialization to a module's weights and zero to biases.

    Parameters
    ----------
    module : nn.Module
        The module to initialize (e.g., nn.Linear, nn.Conv2d).
    gain : float, optional
        Scaling factor for the Xavier initialization. Default is 1.0.

    Returns
    -------
    nn.Module
        The initialized module.

    Examples
    --------
    >>> layer = nn.Linear(10, 5)
    >>> xavier_uniform_init(layer)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module
