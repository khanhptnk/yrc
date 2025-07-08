Configuration
=============

YRC provides two ways to specify arguments: a YAML file or command-line flags. YAML arguments override default arguments, and command-line flags override YAML arguments. Each algorithm or policy has a `*Config` dataclass that specifies its arguments. For example, the config class for the LogitAlgorithm is:

.. code-block:: python

    @dataclass
    class LogitAlgorithmConfig:
        name: str = "logit"
        num_rollouts: int = 128
        percentiles: List[float] = field(default_factory=lambda: list(range(0, 101, 10)))
        explore_temps: List[float] = field(default_factory=lambda: [1.0])
        score_temps: List[float] = field(default_factory=lambda: [1.0])

Suppose you want to change ``num_rollouts``. You can do so in a YAML file:

.. code-block:: yaml

    algorithm:
      name: "logit"
      num_rollouts: 128

Or by specifying flags ``algorithm.name=logit algorithm.num_rollouts=128`` (we use OmegaConf, so dashes are not needed).

.. note::

   When you specify environment, algorithm, or policy arguments through YAML or command-line flags, you must provide the ``name`` of the environment, algorithm, or policy, or you will get an error.  
   The list of valid names is provided in the corresponding registry (e.g., ``yrc.environments.registry``).  
   You can also register a new name to add a new environment, algorithm, or policy, which will be covered in the next tutorials.

