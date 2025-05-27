import json
import logging
import pprint


def get_test_eval_info(env_suite, env_name, test_env, evaluator):
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
