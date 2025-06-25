Quickstart
==========

This page walks you through the steps to install and train a simple PPO coordinator on the Procgen-CoinRun "hard" tasks using the `yrc` package.

Step 0: Clone the Repository
----------------------------

.. code-block:: bash

   git clone <your-yrc-repo-url>
   cd yrc

Step 1: Install the Package Locally
-----------------------------------

.. code-block:: bash

   pip install -e .

Step 2: Train and Evaluate the Novice Agent
-------------------------------------------

**Train the novice on the `easy` distribution:**

.. code-block:: bash

   python examples/procgen_agent.py \
       --config configs/procgen_ppo.yaml \
       name=procgen_novice \
       env.train.distribution_mode=easy \
       env.test.distribution_mode=easy

**Evaluate the novice on the `hard` distribution:**

.. code-block:: bash

   python examples/procgen_agent.py \
       --config configs/procgen_ppo.yaml \
       eval_name=procgen_novice \
       env.test.distribution_mode=hard \
       policy.load_path=experiments/procgen_novice/best_test.ckpt

Step 3: Train and Evaluate the Expert Agent
-------------------------------------------

**Train the expert on the `hard` distribution:**

.. code-block:: bash

   python examples/procgen_agent.py \
       --config configs/procgen_ppo.yaml \
       name=procgen_expert \
       env.train.distribution_mode=hard \
       env.test.distribution_mode=hard \
       algorithm.total_timesteps=25000000

.. note::

   We increase the number of training steps since the tasks are harder. 
   Any command-line flag in `yrc` will overwrite the corresponding value in the YAML config file.

**Evaluate the expert on the `hard` distribution:**

.. code-block:: bash

   python examples/procgen_agent.py \
       --config configs/procgen_ppo.yaml \
       eval_name=procgen_expert \
       env.test.distribution_mode=hard \
       policy.load_path=experiments/procgen_expert/best_test.ckpt

Step 4: Train a Coordinator on the `hard` Distribution
------------------------------------------------------

.. code-block:: bash

   python examples/procgen_coord_policy.py \
       --config configs/procgen_skyline.yaml






