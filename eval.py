import os

import yrc

if __name__ == "__main__":
    config = yrc.load_config()

    envs, policy, evaluator = yrc.make(config, eval=True)

    if config.general.algorithm != "always" and config.filename is not None:
        policy.load_model(os.path.join(config.experiment_dir, config.file_name))

    evaluator.eval(policy, envs, eval_splits=["test"])
