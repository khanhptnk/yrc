Main classes
===============

There are three main types of classes in YRC: **Environment**, **Policy**, and **Algorithm**.

- An **Environment** defines a (PO)MDP—the task to be solved.
- A **Policy** *acts* within an Environment.
- An **Algorithm** defines a procedure to find the best Policy for a given Environment.

YRC can be used as a reinforcement learning package, as demonstrated in the following script for training novice and expert agents:

.. code-block:: python

    envs = {
        split: environments.procgen.make_env(split, config.env)
        for split in ["train"] + args.eval_splits
    }
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)
    evaluator = yrc.Evaluator(config.evaluation)
    algorithm.train(
        policy=policy,
        envs=envs,
        evaluator=evaluator,
        train_split="train",
        eval_splits=args.eval_splits,
    )

In this example, we first create training, validation, and test environments. Then, we create the policy using ``yrc.make_policy`` and the algorithm using ``yrc.make_algorithm``. We also need an evaluator to assess the policy. Finally, we pass the policy, environments, and evaluator to the ``train()`` function of the algorithm.

A more complex example, for training a coordinator, is shown below:

.. code-block:: python

    base_envs = {}
    for split in splits:
        base_envs[split] = environments.procgen.make_env(split, config.env)

    train_novice = yrc.load_policy(config.train_novice, base_envs["train"])
    train_expert = yrc.load_policy(config.train_expert, base_envs["train"])
    test_novice = yrc.load_policy(config.test_novice, base_envs[eval_splits[0]])
    test_expert = yrc.load_policy(config.test_expert, base_envs[eval_splits[0]])

    reward_per_action = compute_reward_per_action(config.env)

    envs = {}
    for split in splits:
        if split in ["train", "val_sim"]:
            novice, expert = train_novice, train_expert
        else:
            novice, expert = test_novice, test_expert
        envs[split] = yrc.CoordEnv(
            config.coordination, base_envs[split], novice, expert
        )

    # Set costs for the coordination environment
    reward_per_action = compute_reward_per_action(config.env)
    for split in splits:
        envs[split].set_costs(reward_per_action)

    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)
    evaluator = yrc.Evaluator(config.evaluation)

    algorithm.train(
        policy,
        envs,
        evaluator,
        train_split="train",
        eval_splits=["val_sim", "val_true"],
    )

In this example, it is important to distinguish between a *base environment* and a *CoordEnv*. A CoordEnv takes as input a base environment, a novice policy, and an expert policy. The coordinator is also a policy. CoordEnv has the same API as a base environment, so the remainder of the code is similar to the previous example: we create an evaluator and call ``algorithm.train()`` with the policy, environments, and evaluator.

