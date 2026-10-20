"""execrl: execution-slice decisions with contextual bandits, offline RL and (simulator-only) PPO.

Layout (read in this order; every module starts with a docstring that explains its contract):

    config.py            all hyper-parameters as dataclasses; one ExperimentConfig serialises to JSON
    compute.py           CPU detection (12th-gen Intel hybrid P/E cores) and thread settings
    actions.py           the 6 atomic actions, macro actions, and the progress-band action mask
    reward.py            slice cost, per-step reward, potential-based shaping, macro-action reward
    data.py              BanditLog / Transitions containers, validation, group-aware splits
    simulator.py         a vectorised TOY slice environment (gym-like reset/step) + data generators
    logging_policy.py    epsilon-randomised rule that records propensities (needed by every offline method)
    baselines.py         TWAP, Almgren-Chriss, a rule policy and the one-step cost rule
    bandit/              reward model (cross-fitted), off-policy evaluation, policy tree, Thompson sampling
    offline/             behaviour model, constrained FQI, FQE, discrete IQL
    online/              masked PPO for the simulator and distillation to a decision tree
    evaluate.py          sample size, CUPED, cluster bootstrap, design effect
    train.py             the training entry points (train_bandit / train_fqi / train_iql / train_ppo) and the CLI

Quick start (repository root):

    PYTHONPATH=src python -m execrl.train bandit --out runs/bandit
    PYTHONPATH=src python -m execrl.train fqi    --out runs/fqi
    PYTHONPATH=src python -m execrl.train ppo    --out runs/ppo --iterations 50
    PYTHONPATH=src python -m pytest -q src/tests

The maths behind each module is on docs/signal-model/reinforcement-learning.md (section numbers are
quoted in the docstrings, e.g. "RL doc 2.4").
"""

__version__ = "0.1.0"
