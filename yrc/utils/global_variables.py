GLOBAL_VARIABLES = {}


def set_global_variable(key, value):
    """
    Set a global variable by key.

    Parameters
    ----------
    key : str
        The key for the global variable.
    value : Any
        The value to set for the global variable.

    Returns
    -------
    None

    Examples
    --------
    >>> set_global_variable('device', 'cuda')
    """
    global GLOBAL_VARIABLES
    GLOBAL_VARIABLES[key] = value


def get_global_variable(key):
    """
    Retrieve the value of a global variable by key.

    Parameters
    ----------
    key : str
        The key for the global variable.

    Returns
    -------
    Any or None
        The value of the global variable, or None if not set.

    Examples
    --------
    >>> get_global_variable('device')
    'cuda'
    """
    return GLOBAL_VARIABLES.get(key)


def get_all_global_variables():
    """
    Get the dictionary of all global variables.

    Returns
    -------
    dict
        Dictionary of all global variables.

    Examples
    --------
    >>> get_all_global_variables()
    {'device': 'cuda', 'seed': 42}
    """
    return GLOBAL_VARIABLES
