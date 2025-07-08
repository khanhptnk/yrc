Skyline
=======

To track progress on a :ref:`YRC-0 problem <core-concepts-yrc0>`, it is important to establish a skyline result—an estimate of the best possible performance. Unlike many machine learning problems, humans may perform poorly on a YRC-0 problem. Due to mismatches in cognitive states, it is not easy for a human to determine when an AI agent needs help. AI agents may not perceive the world as humans do, so situations that are difficult for AI agents may not be difficult for humans, and vice versa.

Instead of using humans, we assume access to the test tasks and the expert, and use PPO to train a coordinator under this setting. This provides a respectable upper-bound for performance.


