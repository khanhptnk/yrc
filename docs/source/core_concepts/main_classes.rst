Main classes
===============

There are three main types of classes in YRC: **Environment**, **Policy**, and **Algorithm**.

- An **Environment** defines a (PO)MDP—the task to be solved.
- A **Policy** *acts* within an Environment.
- An **Algorithm** defines a procedure to find the best Policy for a given Environment.

We demonstrate how these classes interact with one another using the following example.
In this example, YRC serves a reinforcement learning package that trains an agent in a Gym environment:

.. code-block:: python
 
    envs = {
        split: environments.procgen.make_env(split, config.env)
        for split in ["train", "test"] 
    }

    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)
    validator = yrc.Evaluator(config.evaluation, envs["test"])

    algorithm.train(policy, envs["train"], [validator])


In this example, we first create the training and test environments. Then, we create the policy using ``yrc.make_policy`` and the algorithm using ``yrc.make_algorithm``. We also need an validator for policy selection during training. Finally, we call ``algorithm.train()`` with the policy, the training environment, and the validator.

Here is a more complex example, which trains a coordinator for a YRC-0 problem:

.. code-block:: python

    def make_envs(splits, config):

        # Create base environments
        base_envs = {}
        for split in splits:
            base_envs[split] = environments.procgen.make_env(split, config.env)

        # Load novice and expert agents
        some_base_env = list(base_envs.values())[0]
        train_novice = yrc.load_policy(config.train_novice, some_base_env)
        train_expert = yrc.load_policy(config.train_expert, some_base_env)
        test_novice = yrc.load_policy(config.test_novice, some_base_env)
        test_expert = yrc.load_policy(config.test_expert, some_base_env)

        # Create coordination environments using base environments and agents
        envs = {}
        for split in splits:
            if split in ["train", "val_sim"]:
                novice, expert = train_novice, train_expert
            else:
                novice, expert = test_novice, test_expert
            envs[split] = yrc.CoordEnv(
                config.coordination, base_envs[split], novice, expert
            )

        # Finally, we need to set the expert-query cost for the coordination environment
        reward_per_action = compute_reward_per_action(config.env)
        for split in splits:
            envs[split].set_costs(reward_per_action)

        return envs

    eval_splits = ["val_sim", "val_true"]
    splits = ["train"] + eval_splits

    envs = make_envs(splits, config)
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in eval_splits:
        validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)


In this example, it is important to distinguish between a *base environment* and a *coordination environment* (*CoordEnv*). A CoordEnv takes as input a base environment, a novice policy, and an expert policy, and constructs the POMDP for the coordinator. The novice, expert, and coordinator are instances of the Policy class. The CoordEnv follows the Gym API, so the remainder of the code is similar to the previous example: we create a list of validators and call ``algorithm.train()`` with the policy, the training environment, and the validators.

