Problem setting
=============

1. The YRC Problem
------------------

YRC stands for *Yield-and-Request-Control*. The term refers to an agent's decision of whether to let another agent make the next decision or to make the decision on its own. It is the atomic decision that an agent must make to efficiently leverage assistance from another agent.

The YRC problem considers a weak agent, called a **novice**, who is asked to perform a novel task. Since the task is novel, this agent alone can't achieve a high score on the task. Fortunately, an **expert** who has mastered the task is present to help. Nevertheless, each help request made to the expert incurs a cost. The problem is to achieve the highest score on the task, where the score includes penalties for the help requests made.

In the context of the a dual cognitive system, the two agents represent the two component systems. But YRC is fundamentally a general problem that concerns the coordination of any two agents. 

2. The Coordinator
------------------

The **coordinator** is a policy that decides at each step whether the novice or the expert will make the decision. In our standard setting, the coordinator is a component of the novice, so it has access to the environment observation given to the novice and can observe the novice's internal computations while this agent is making a decision. Meanwhile, the expert is a black box to the coordinator. But our package supports other settings such as an transparent expert. 

The YRC problem is essentially a POMDP for the coordinator, where the action is binary (request help from the expert or not).

3. The Coordination Environment (CoordEnv)
------------------------------------------

**CoordEnv** is our implementation of the POMDP presented to the coordinator. It follows the OpenGym API with some special requirements (see "Tutorials" for more details). This familiar interface allows easy integration of existing methods to tackle YRC.

A CoordEnv is made up of a base environment, a novice, and an expert. When a coordinator makes a (binary) decision, the CoordEnv queries the novice or the expert to get an environment action. This action is fed into the base environment to change the current environment state and obtain the environment reward. An expert-request cost is subtracted from the environment reward to yield the final reward of the step.

4. Coordination Cost
----------------------

We considers two types of cost: the cost of providing assistance (expert-assistance cost) and the cost of switching from one agent to the other (context-switching cost). Our package allows users to flexibly specify these costs. The reward at each time step of a CoordEnv is computed as:

.. math::

   r_t = R(s_t, a_t) - \mathbf{1}\{ x_t = expert \} \cdot \alpha \cdot  BP - \mathbf{1}\{ x_t \neq x_{t - 1} \} \cdot \beta \cdot BP


where :math:`R(s, a)` is the base environment's reward function, :math:`x_t` denotes the agent currently taking control, :math:`BP` is a user-specified base per-step penalty, and :math:`\alpha`, :math:`\beta` are user-specified cost weights.

.. _core-concepts-yrc0:
5. The YRC-0 Problem
--------------------

YRC-0 is the most difficult setting of YRC. YRC-0 assumes that for training the coordinator, the test tasks (or the test task distribution) and the expert will *not* be available. Moreover, the training and test distriubtions differ. The problem essentially requires generalization to novel environments and collaborators. 

YRC-0 motivates the development of unsupervised methods for YRC. Even when one cannot solve YRC-0 perfectly, solutions to this problem can significantly reduce the amount of supervision required to solve YRC.

YRC-0 introduces the challenge of policy validation. In a setting where the training and test conditions are similar, one can evaluate a policy on the training conditions and the performance provides a reliable signal to select the best policy for testing. That is impossible with YRC-0. An agent must somehow be anticipate how good it is at collaborating with another agent, despite being trained in isolation in different environments. 

Our research paper proposes a method to simulate the test conditions. We train a *weakened novice* that performs poorly on the training task distribution. We then use the weakened novice and the novice as the novice and the expert on the training task distribution. We hope that the triple (simulated novice, novice, training task distribution) decently simulates the test conditions involving (novice, expert, test task distribution).




