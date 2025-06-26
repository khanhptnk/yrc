import json
import logging
import pprint


def get_test_eval_info(env_suite, env_name, test_env, evaluator):
    """
    Retrieve or compute evaluation statistics for a test environment and expert agent.

    If statistics for the given environment are missing, this function evaluates the expert
    agent on the test environment, saves the results, and returns the summary.

    Parameters
    ----------
    env_suite : str
        Name of the environment suite (e.g., 'procgen').
    env_name : str
        Name of the specific environment (e.g., 'coinrun').
    test_env : object
        Test environment instance, expected to have 'expert', 'base_env', and 'num_envs' attributes.
    evaluator : object
        Evaluator instance with an 'eval' method for running evaluation.

    Returns
    -------
    info : dict
        Dictionary of evaluation statistics for the test environment and expert agent.

    Examples
    --------
    >>> info = get_test_eval_info('procgen', 'coinrun', test_env, evaluator)
    >>> print(info['reward_mean'])
    """
    with open("yrc/metadata/test_eval_info.json") as f:
        data = json.load(f)

    if env_name not in data[env_suite]:
        logging.info(f"Missing info about {env_suite}-{env_name}!")
        logging.info("Calculating missing info (taking a few minutes)...")
        # eval expert agent on test environment to get statistics
        summary = evaluator.eval(
            test_env.expert,
            {"test": test_env.base_env},
            ["test"],
            num_episodes=test_env.num_envs,
        )["test"]
        data[env_suite][env_name] = summary

        with open("metadata/test_eval_info.json", "w") as f:
            json.dump(data, f, indent=2)
        logging.info("Saved info!")

    info = data[env_suite][env_name]

    logging.info(f"{pprint.pformat(info, indent=2)}")
    return info
