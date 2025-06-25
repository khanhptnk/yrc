Core Concepts
=============

1. The YRC Problem
------------------

YRC stands for *Yield-and-Request-Control*. The term refers to an agent's decision of whether to let another agent make the next decision or to make the decision on its own. It is the atomic decision that an agent must make to efficiently leverage assistance from another agent.

The YRC problem considers a weak agent, called a **novice**, who is asked to perform a novel task. Since the task is novel, this agent alone can't achieve a high score on the task. Fortunately, an **expert** who has mastered the task is present to help. Nevertheless, each help request made to the expert incurs a cost. The problem is to achieve the highest score on the task, where the score includes penalties for the help requests made.

2. The Coordinator
------------------

The **coordinator** is a policy that decides at each step whether the novice or the expert will make the decision. The coordinator is a component of the novice in this problem, so it has access to the environment observation given to the novice and can observe the novice's internal computations while this agent is making a decision. Meanwhile, the expert is a black box to the coordinator. The YRC problem is essentially a POMDP for the coordinator, where the action is binary (request help from the expert or not).

3. The Coordination Environment (CoordEnv)
------------------------------------------

**CoordEnv** is our implementation of the POMDP presented to the coordinator. It follows the OpenGym API with some special requirements (see documentation for more details). This familiar interface allows easy integration of existing methods to tackle YRC.

A CoordEnv is made up of a base environment, a novice, and an expert. When a coordinator makes a (binary) decision, the CoordEnv queries the novice or the expert to get an environment action. This action is fed into the base environment to change the current environment state and obtain the environment reward. An expert-request cost is subtracted from the environment reward to yield the final reward of the step.

4. Expert-request Cost
----------------------

Our package allows flexible specification of this cost. Our default way of computing this cost is to roll out the expert on the test tasks to compute an average reward :math:`\bar G` and an average episode length :math:`\bar L`. The reward at each time step is computed as:

.. math::

   r_t = R(s_t, a_t) - \mathbf{1}\{ x_t = e \} \cdot \alpha \cdot \frac{\bar G_e}{\bar T_e}

where :math:`R(s, a)` is the base environment's reward function and :math:`\alpha` is a hyperparameter (specified through the `coordination.expert_query_cost_weight` flag).

.. _core-concepts-yrc0:
5. The YRC-0 Problem
--------------------

YRC-0 assumes that for training the coordinator, the test tasks (or the test task distribution) and the expert will not be available. YRC-0 tests for out-of-distribution generalization and zero-shot coordination.

YRC-0 motivates the development of unsupervised methods for YRC. Even when one cannot solve YRC-0 perfectly, solutions to this problem can significantly reduce the amount of supervision required to solve YRC.

YRC-0 introduces the new problem of policy validation. In a standard supervised learning setting, one can evaluate a policy under conditions similar to the test conditions to select the best policy for testing. That is impossible with YRC-0. Our research paper proposes a method to simulate the test conditions. We train a **simulated novice** for a limited number of training iterations. This creates an agent that performs poorly on the *full* training task distribution, similar to how the novice falters on the test tasks. We then use the simulated novice as the expert on the training tasks, as this agent is assumed to perform well in-distribution. In the end, the triple (simulated novice, novice, training task distribution) simulates the test conditions involving (novice, expert, test task distribution).




