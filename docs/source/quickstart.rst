Quickstart
==========

0. Requirements
------------

- Python 3.8 or higher

- (Optional) CUDA-compatible GPU for faster training

1. Installation
---------------

You can install ``yrc`` using pip:

.. code-block:: bash

    pip install yrc

Alternatively, install from source:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/khanhptnk/yrc.git
    pip install -e .

Check that yrc was installed correctly by running:

.. code-block:: bash

    python -c "import yrc; print(yrc.__version__)"

2. Training a Random Coordination Policy for Procgen-Coinrun
------------------------------------------------------------

If you have not already cloned the repository in the previous step, do so now. The repository already integrates the Procgen environments:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/khanhptnk/yrc.git

Next, train the coordination policy using the following command:

.. code-block:: bash

    python examples/procgen_yrc.py --config configs/procgen_random.yaml --mode train --type coord

Training takes about 16 minutes to complete on an RTX 6000 GPU.  
You should expect a reward of around 5.73:

.. code-block:: none

    [0:16:12 INFO]: BEST test so far
    [0:16:12 INFO]: Parameters: {'temperature': 1.0, 'threshold': -1.9764122247695923}
    [0:16:12 INFO]:    Steps:         18,792
       Episode length: mean   73.41  min   17.00  max  256.00
       Reward:         mean 5.73 ± 0.58
       Base Reward:    mean 6.88 ± 0.57
       Action 1 fraction:    0.36

Wandb logs of all methods on Procgen-Coinrun are available `here <https://wandb.ai/kxnguyen/YRC-public?nw=nwuserkxnguyen>`_.

