Reinforcement Learning
======================

We provide a full re-implementation of the PPO algorithm [1]_. The policy supports three types of input features:

- ``obs``: the observation from the base environment.
- ``hidden``: the hidden representations of the novice.
- ``dist``: the output softmax distribution of the novice.

Users can specify any non-empty combination of these feature types (e.g., ``obs_hidden``, ``obs_hidden_dist``).

References
----------

.. [1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.  
   "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017.

