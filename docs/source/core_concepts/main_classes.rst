Main classes
===============

There are three main types of classes in YRC: **Environment**, **Policy**, and **Algorithm**.

- An **Environment** defines a (PO)MDP—the task to be solved.
- A **Policy** acts within an Environment.
- An **Algorithm** defines a procedure to find the best Policy for a given Environment.

We demonstrate how these classes interact with one another using the following example:

.. code-block:: python
 
    base_envs = make_base_envs(config)
    envs = make_coord_envs(config, base_envs) if args.type == "coord" else base_envs
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)
    validators = { "test" : yrc.Evaluator(config.evaluation, envs["test"]) }

    algorithm.train(policy, envs["train"], validators)


In this example, we first create the base environments. Optionally, if we are training a coordinator, we wrap the base environments inside coordination environments. Then, we create a policy and an algorithm. We also need validators for policy selection. Finally, we call ``algorithm.train()`` with the policy, the training environment, and the validators.

