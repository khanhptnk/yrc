Train an Agent Using PPO
========================

In this tutorial, you will learn how to use YRC to train an agent on the ``Procgen-coinrun`` task using the PPO algorithm.

The Procgen environments are included in our GitHub repository.  
Remember to add ``--recurse-submodules`` when cloning the repository:


TODO: change repo link

.. code-block:: bash

    git clone --recurse-submodules git@github.com:khanhptnk/YRC-Bench.git

1. Run the Training Script
--------------------------

A script for training PPO agents is provided at ``examples/procgen_yrc.py``.  
The following command trains an agent on the ``easy`` version of the ``coinrun`` task:

.. code-block:: bash

    python examples/procgen_yrc.py \
      --config configs/procgen_agent.yaml \
      --mode train \
      --type agent \
      name=procgen_novice \
      overwrite=1 \
      device=0

Let's analyze this command:

- There are two types of flags: those with ``--`` and those without.  
  The flags with ``--`` are defined in the script (``examples/procgen_yrc.py``), while the others are defined by YRC.

- The ``--config`` flag is especially important because it specifies the YAML file that contains the configuration for training.  
  You will need to provide this flag in almost every use case of YRC.

- The ``name`` flag specifies the name of the output folder for this run.  
  When evaluating a policy, use ``eval_name`` instead. (Setting ``eval_name`` to a non-``None`` value activates evaluation mode.)

- ``overwrite=1`` indicates that you want to overwrite the contents inside the ``name`` folder if it already exists.  
  Be careful: this may overwrite previous checkpoints. In this case, we use it for convenience so we don't have to manually delete the folder each time.

2. Configuration
----------------

Take a look at the configuration file at ``configs/procgen_agent.yaml``:

.. code-block:: yaml

    name: "procgen_agent_novice"

    env:
      name: "procgen"
      train:
        distribution_mode: "easy"
        seed: 0
        start_level: 0
        num_levels: 100000
      test:
        distribution_mode: "easy"
        seed: 0
        start_level: 0
        num_levels: 100000

    algorithm:
      name: "ppo"
      total_timesteps: 15000000

    policy:
      name: "ppo"
      model: "impala_ppo"

The purpose of this file is not to specify all configuration arguments, but to overwrite default values as needed.  
For example, the default values for the PPO algorithm are defined in ``yrc/algorithms/ppo.py``:

.. code-block:: python

    @dataclass
    class PPOAlgorithmConfig:
        name: str = "ppo"
        log_freq: int = 10
        save_freq: int = 0
        num_steps: int = 256
        total_timesteps: int = 1_500_000
        update_epochs: int = 3
        gamma: float = 0.999
        gae_lambda: float = 0.95
        num_minibatches: int = 8
        clip_coef: float = 0.2
        norm_adv: bool = True
        clip_vloss: bool = True
        vf_coef: float = 0.5
        ent_coef: float = 0.01
        max_grad_norm: float = 0.5
        learning_rate: float = 0.0005
        critic_pretrain_steps: int = 0
        anneal_lr: bool = False
        log_action_id: int = 1

In this YAML file, we only override the ``total_timesteps`` argument.

.. note::

   When you specify environment, algorithm, or policy arguments through YAML or command-line flags, you must provide the ``name`` of the environment, algorithm, or policy, or you will get an error.  
   The list of valid names is provided in the corresponding registry (e.g., ``yrc.environments.registry``).  
   You can also register a new name to add a new environment, algorithm, or policy, which will be covered in the next tutorials.

You can also override arguments in the YAML file using command-line flags.  
For example, to train an agent on the ``hard`` version of ``coinrun``, run:

.. code-block:: bash

    python examples/procgen_yrc.py \
      --config configs/procgen_agent.yaml \
      --mode train \
      --type agent \
      name=procgen_expert \
      overwrite=1 \
      device=0 \
      env.train.distribution_mode=hard \
      env.test.distribution_mode=hard \
      algorithm.total_timesteps=25000000

3. Training Script
------------------

Now let's look more closely at the training script ``examples/procgen_yrc.py``.

YRC separates the base environment code from the main package to enhance extensibility.  
The environment code defines the configuration arguments for the environment.  
To combine these arguments with YRC's arguments, you must register the environment configuration with YRC before parsing all arguments.

.. code-block:: python

    yrc.register_environment("procgen", ProcgenConfig)
    args, config = parse_args()

This allows you to specify environment arguments using YAML or command-line flags, like ``env.train.distribution_mode`` in the previous example.

Training follows these steps:

.. code-block:: python

    # 1) Create environments
    envs = make_base_envs(config)
    # 2) Create the learning policy
    policy = yrc.make_policy(config.policy, envs["train"])
    # 3) Create PPO algorithm
    algorithm = yrc.make_algorithm(config.algorithm)

    # 4) Create validators for policy selection
    validators = {}
    validators["test"] = yrc.Evaluator(config.evaluation, envs["test"])

    # 5) Run the algorithm
    algorithm.train(policy, envs["train"], validators)


