Configuration
=============

YRC provides two ways to specify arguments: a YAML file or command-line flags. YAML arguments overwrite default arguments, and command-line flags overwrite YAML arguments. Each algorithm or policy has a `*Config` dataclass that specifies its arguments. For example, the config class for the LogitAlgorithm is:

.. code-block:: python

    @dataclass
    class LogitAlgorithmConfig:
        cls: str = "LogitAlgorithm"
        num_rollouts: int = 128
        percentiles: List[float] = field(default_factory=lambda: list(range(0, 101, 10)))
        explore_temps: List[float] = field(default_factory=lambda: [1.0])
        score_temps: List[float] = field(default_factory=lambda: [1.0])

Suppose you want to change ``num_rollouts``. You can do so in a YAML file:

.. code-block:: yaml

    algorithm:
      cls: "LogitAlgorithm"
      num_rollouts: 128

Or by specifying a flag ``algorithm.num_rollouts=128`` (we use OmegaConf, so dashes are not needed).




