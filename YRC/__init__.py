import flags
import YRC.core.algorithm as algo_factory
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator


def load_config():
    args = flags.make()
    return config_utils.load(args.config, flags_args=args)


def make(config, eval=False):
    envs = make_environments(config)
    policy = make_policy(config, envs["train"])
    evaluator = Evaluator(config.evaluation)

    if eval:
        return envs, policy, eval

    if config.general.algorithm == "always":
        algorithm = None
    else:
        algorithm = make_algorithm(config, envs["train"])

    return envs, policy, algorithm, evaluator


def make_algorithm(config, env):
    return algo_factory.make(config, env)


def make_environments(config):
    return env_factory.make(config)


def make_policy(config, env):
    return policy_factory.make(config, env)
