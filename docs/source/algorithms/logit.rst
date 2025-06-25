Logit Algorithms
=============

The novice computes a confidence score based on its output logits. It makes a help request when the score is below a threshold.  

**Notation:**

- :math:`z = (z_1, \dots, z_{|A|})` are the logits computed by the novice.

- :math:`p = \mathrm{Softmax}(z)` is the probability vector.

- :math:`p^{\downarrow}` denotes the elements of :math:`p` sorted in descending order.

**Supported metrics:**

- ``max_logit``:  
  The maximum logit value.
  :math:`\max_i z_i`

- ``max_prob`` [1]_:
  The maximum probability.
  :math:`\max_i p_i` 

- ``margin`` [2]_:
  The difference between the highest and second-highest probabilities.
  :math:`p_1^{\downarrow} - p_2^{\downarrow}`

- ``entropy`` [3]_:
  The negative entropy of the action distribution.
  :math:`\sum_i p_i \ln p_i`

- ``energy`` [4]_:
  The log-sum-exp of the logits.
  :math:`\ln \sum_i \exp(z_i)`


A challenge in this approach is determining the appropriate threshold.
We address this by proposing the following adaptive procedure:

1. **Exploration:**  
   Use the novice to explore the training environment, generating a set of states :math:`\mathcal{S}_{\text{train}}`.

2. **Score Computation:**  
   For each state :math:`s \in \mathcal{S}_{\text{train}}`, compute its confidence score :math:`c(s)`.  
   This results in a pool of confidence scores  
   :math:`\mathcal{C} = \{c(s) \mid s \in \mathcal{S}_{\text{train}}\}`.

3. **Threshold Selection:**  
   Consider the :math:`n`-th percentiles of :math:`\mathcal{C}` as candidate thresholds (:math:`n = 0, 10,..., 100`).

4. **Validation:**  
   For each candidate threshold, evaluate the performance on the validation tasks.

5. **Test-Time Selection:**  
   Select the threshold :math:`\tau^*` that yields the best validation performance and use it during testing.


References
----------

.. [1] David D. Lewis. "A sequential algorithm for training text classifiers: Corrigendum and additional data." *Acm Sigir Forum*, 29:13–19, 1995.

.. [2] Burr Settles. "Active learning literature survey." Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.

.. [3] Tobias Scheffer, Christian Decomain, and Stefan Wrobel. "Active hidden markov models for information extraction." In *International symposium on intelligent data analysis*, pages 309–318. Springer, 2001.

.. [4] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. "Energy-based out-of-distribution detection." *Advances in Neural Information Processing Systems*, 33:21464–21475, 2020.
