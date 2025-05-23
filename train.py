import YRC

if __name__ == "__main__":
    config = YRC.load_config()

    envs, policy, algorithm, evaluator = YRC.make(config)

    if config.general.algorithm == "always":
        evaluator.eval(policy, envs, eval_splits=config.evaluation.val_splits)
    else:
        algorithm.train(
            policy,
            envs,
            evaluator,
            train_split="train",
            eval_splits=config.evaluation.val_splits,
        )
