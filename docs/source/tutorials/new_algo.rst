Add a New Algorithm
===================

In this tutorial, you will learn how to add a new algorithm.

We will implement a simple algorithm called ``AskEveryK``, which learns a policy that asks for help every K steps. The algorithm searches for the best value of K from a set of candidates.

The code for this tutorial is provided at ``examples/procgen_ask_every_k.py``. Try running it with:

.. code-block:: bash

    python examples/procgen_ask_every_k.py --config configs/procgen_ask_every_k.yaml overwrite=1

1. Implement the Algorithm
--------------------------

We implement the ``AskEveryKAlgorithm`` class, which is a subclass of ``yrc.core.Algorithm``, along with its configuration dataclass, as follows:

.. code-block:: python

    from yrc.core import Algorithm

    @dataclass
    class AskEveryKAlgorithmConfig:
        name: str = "ask_every_k"
        candidates: List[int] = field(default_factory=lambda: [5, 10, 15, 20])

    class AskEveryKAlgorithm(Algorithm):
        config_cls = AskEveryKAlgorithmConfig

        def __init__(self, config):
            self.config = config

        def train(self, policy, env, validators):
            config = self.config
            self.save_dir = get_global_variable("experiment_dir")

            best_k = None
            best_result = {}
            for split in validators:
                best_result[split] = {"reward_mean": -float("inf")}

            for k in config.candidates:
                logging.info(f"Evaluating k={k}")
                policy.set_params({"k": k})
                for split, validator in validators.items():
                    result = validator.evaluate(policy)
                    if result["reward_mean"] > best_result[split]["reward_mean"]:
                        best_result[split] = result
                        best_k = k
                        self.save_checkpoint(policy, f"best_{split}")

            for split, validator in validators.items():
                logging.info(f"BEST result for {split} (k={best_k}):")
                validator.summarizer.write(best_result[split])

        def save_checkpoint(self, policy, name):
            save_path = f"{self.save_dir}/{name}.ckpt"
            torch.save(
                {
                    "policy_config": policy.config,
                    "model_state_dict": policy.get_params(),
                },
                save_path,
            )
            logging.info(f"Saved checkpoint to {save_path}")

2. Implement the Policy
-----------------------

Next, we implement a policy with a parameter K, which queries the expert every K steps.

.. code-block:: python

    from yrc.core import Policy

    @dataclass
    class AskEveryKPolicyConfig:
        name: str = "ask_every_k"
        load_path: Optional[str] = None

    class AskEveryKPolicy(Policy):
        config_cls = AskEveryKPolicyConfig

        def __init__(self, config, env):
            self.config = config
            self.EXPERT = env.EXPERT
            self.k = None
            self.step = np.array([0] * env.num_envs)
            self.device = get_global_variable("device")

        def reset(self, done):
            self.batch_size = len(done)
            if self.batch_size < len(self.step):
                self.step = self.step[: self.batch_size]
            self.step[done] = 0

        def act(self, obs, temperature=None):
            batch_size = self.batch_size
            assert obs["base_obs"].shape[0] == batch_size
            action = torch.zeros(batch_size).long().to(self.device)
            for i in range(batch_size):
                if self.step[i] % self.k == 0:
                    action[i] = self.EXPERT
                else:
                    action[i] = 1 - self.EXPERT
                self.step[i] += 1
            return action

        def set_params(self, params):
            self.k = params["k"]

        def get_params(self):
            return {"k": self.k}

        def train(self):
            pass

        def eval(self):
            pass

3. Register the Algorithm and Policy
------------------------------------

Finally, we register the algorithm and the policy with YRC so that their configuration arguments are included in YRC’s argument list.

.. code-block:: python

    yrc.register_algorithm("ask_every_k", AskEveryKAlgorithm)
    yrc.register_policy("ask_every_k", AskEveryKPolicy)

That covers all the major steps. The rest of the code follows the standard process for training a coordination policy.

