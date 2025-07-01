Train a Coordination Policy
===========================

In this tutorial, you will learn how to train a coordination policy to enable effective collaboration between two agents.

Training a coordination policy is very similar to training a single agent, thanks to YRC's standardized abstractions for algorithms, policies, and environments. The main differences are the configuration arguments and the need to wrap the base environments with `CoordEnv`.

0. Refresher: What is CoordEnv?
-------------------------------

A `CoordEnv` represents the POMDP presented to the coordination policy.  
It is implemented as a Gym environment and consists of a base environment, a novice, and an expert policy.

The action space contains two actions: ``NOVICE`` and ``EXPERT``, corresponding to querying the novice or the expert for the next decision.  
When an action is chosen, the corresponding agent is queried for a base environment action. This action is then fed into the base environment to obtain the next state and reward.

1. Configuration
----------------

Compared to training an agent, training a coordination policy differs in:

- The algorithm, policy, and policy model

- The coordination configuration

- The paths to load the novice and expert agents

Let's look at an example at ``configs/procgen_skyline.yaml``, which uses the PPO algorithm:

.. code-block:: yaml

    name: "procgen_skyline"

    env:
      name: "procgen"
      train:
        distribution_mode: "hard"
      val_sim:
        distribution_mode: "hard"

    algorithm: "ppo"

    policy:
      name: "ppo"
      model:
        name: "impala_coord_ppo"
        feature_type: obs

    coordination:
      expert_query_cost_weight: 0.4
      switch_agent_cost_weight: 0.0
      temperature: 1.0

    train_novice: "experiments/procgen_ppo_novice/best_test.ckpt"
    train_expert: "experiments/procgen_ppo_expert/best_test.ckpt"
    test_novice: "experiments/procgen_ppo_novice/best_test.ckpt"
    test_expert: "experiments/procgen_ppo_expert/best_test.ckpt"

The ``Random`` algorithm requires a simpler policy configuration of the algorithm and the policy:

.. code-block:: yaml

    algorithm:
      name: "random"

    policy:
      cls: "RandomPolicy"

2. Training Script
------------------

There is a new step in the training script, which creates the `CoordEnv`:

.. code-block:: python

    base_envs = make_base_envs(config)
    # NEW STEP: create CoordEnv
    envs = make_coord_envs(config, base_envs)
    policy = yrc.make_policy(config.policy, envs["train"])
    algorithm = yrc.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = yrc.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)

The ``make_coord_envs`` function is defined as follows:

.. code-block:: python

    def make_coord_envs(config, base_envs):
        # 1) Load novice and expert
        some_base_env = list(base_envs.values())[0]
        train_novice = yrc.load_policy(config.train_novice, some_base_env)
        train_expert = yrc.load_policy(config.train_expert, some_base_env)
        test_novice = yrc.load_policy(config.test_novice, some_base_env)
        test_expert = yrc.load_policy(config.test_expert, some_base_env)

        # 2) Create CoordEnv
        # We use train_novice and train_expert for training and validation
        # and test_novice and test_expert for testing
        envs = {}
        for split in splits:
            if split in ["train", "val_sim"]:
                novice, expert = train_novice, train_expert
            else:
                novice, expert = test_novice, test_expert
            envs[split] = yrc.CoordEnv(
                config.coordination, base_envs[split], novice, expert
            )

        # 3) Set coordination costs 
        # The cost can depend on the reward structure of the base env
        base_penalty = compute_reward_per_action(config.env)
        for split in splits:
            envs[split].set_costs(base_penalty) 
        return envs


