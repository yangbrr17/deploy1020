"""Offline RL on logged transitions (RL doc 7): the second step, when multi-step effects matter.

    behaviour.py   pi_beta(a | s): a LightGBM multiclass model of the logging policy (support estimate)
    fqi.py         batch-constrained fitted Q iteration with LightGBM (recommended)
    fqe.py         fitted Q evaluation of any fixed policy (the offline evaluator)
    iql.py         discrete implicit Q-learning in PyTorch (neural alternative)
"""
