"""All hyper-parameters in one place, as dataclasses.

Why dataclasses and not dicts: typos fail loudly, defaults are documented next to the field, and
ExperimentConfig.to_json / from_json give a reproducible record of every run (train.py writes it
next to the model). Change defaults here, not inside the algorithms.

Units: prices in currency, quantities in lots, costs and rewards in basis points (bp) of the
reference mid, time in decision steps unless a field says seconds.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass


@dataclass
class SliceConfig:
    """One execution slice (RL doc 2.1): buy q_star lots within n_steps decisions."""
    n_steps: int = 10              # decisions per slice (N)
    q_star: float = 10.0           # slice target, lots
    tick: float = 0.01             # price increment
    mid0: float = 20.0             # starting mid (only scales bp: 1 tick = 5 bp here)
    take_lots: float = 3.0         # max lots one TAKE buys at the touch
    fill_lots: float = 3.0         # max lots one passive fill gives
    fee_take_ticks: float = 0.1    # taker fee, in ticks per share
    fee_make_ticks: float = -0.05  # maker fee (negative = rebate), in ticks per share
    clean_extra_ticks: float = 1.0 # clean-up premium above the ask at the end (walks the book)
    band_width: float = 0.25       # progress band half-width around the linear schedule
    band_eps: float = 0.05         # buffer inside the band (epsilon_band in RL doc 2.3)
    p_wide_spread: float = 0.3     # probability the spread is 2 ticks at a step (else 1)
    signal_noise: float = 0.03     # noise on the observed up/down probabilities
    drift_ar: float = 0.8          # AR(1) coefficient of the latent drift
    shaping: bool = False          # potential-based shaping (RL doc 2.4); does not change the optimum


@dataclass
class LoggingConfig:
    """Randomised logging policy (RL doc 5)."""
    epsilon: float = 0.10          # exploration share inside the safe set
    seed: int = 0


@dataclass
class LGBMParams:
    """LightGBM settings shared by the reward model, FQI and the behaviour model.
    num_threads=0 means "use compute.recommended_threads()" (resolved at run time)."""
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    num_threads: int = 0
    verbose: int = -1


@dataclass
class BanditConfig:
    """Macro-action contextual bandit + DR (RL doc 6)."""
    n_samples: int = 40_000        # logged decisions to generate in the simulator
    test_frac: float = 0.3         # share of GROUPS (parent orders) held out for evaluation
    n_folds: int = 5               # cross-fitting folds for the reward model (DR needs out-of-fold mu)
    weight_clip: float = 20.0      # M in w = min(pi/pi0, M)
    tree_depth: int = 2            # policy tree depth (0 = no tree)
    tree_grid: int = 16            # candidate thresholds per feature in the tree search
    gate_z: float = 1.645          # one-sided 95%: deploy only if DR(new) - DR(log) - z*se > 0
    lgbm: LGBMParams = field(default_factory=LGBMParams)


@dataclass
class FQIConfig:
    """Batch-constrained fitted Q iteration (RL doc 7.1)."""
    n_episodes: int = 6_000        # logged slices to generate
    # keep a' only if pi_beta(a'|s') >= tau * max_b pi_beta(b|s'). Must sit BELOW the exploration
    # ratio of the logs, (eps/k) / (1 - eps + eps/k) ~ 0.02 for eps = 0.1 and k = 5 allowed actions;
    # with tau = 0.1 every explored action is cut and FQI can only copy the rule (seen in testing).
    support_tau: float = 0.01
    n_iter: int = 0                # 0 = slice horizon (exact backward induction)
    test_frac: float = 0.3
    lgbm: LGBMParams = field(default_factory=lambda: LGBMParams(n_estimators=200))


@dataclass
class IQLConfig:
    """Discrete implicit Q-learning (RL doc 7.2)."""
    expectile: float = 0.8         # kappa_e
    beta: float = 5.0              # advantage temperature
    adv_clip: float = 100.0        # max exp(beta * A)
    hidden: int = 256
    lr: float = 3e-4
    batch_size: int = 512
    steps: int = 20_000
    tau_target: float = 0.005      # Polyak rate for the target Q
    seed: int = 0


@dataclass
class PPOConfig:
    """Masked PPO in the simulator (RL doc 8)."""
    n_envs: int = 256              # vectorised envs (numpy), not processes
    iterations: int = 200
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 10
    minibatch: int = 512
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    hidden: int = 64
    max_grad_norm: float = 0.5
    seed: int = 0


@dataclass
class ComputeConfig:
    """Threads. 0 = auto (see compute.py). On a 12th-gen i9 with 8 P-cores the auto value is 8."""
    threads: int = 0
    workers: int = 0


@dataclass
class ExperimentConfig:
    slice: SliceConfig = field(default_factory=SliceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    fqi: FQIConfig = field(default_factory=FQIConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    seed: int = 0

    def to_json(self, path=None) -> str:
        s = json.dumps(asdict(self), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, path_or_str: str) -> "ExperimentConfig":
        """Load a (possibly partial) JSON config; missing keys keep their defaults."""
        try:
            with open(path_or_str) as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = json.loads(path_or_str)
        return _from_dict(cls, d)


def _from_dict(cls, d: dict):
    obj = cls()
    for f in fields(cls):
        if f.name not in d:
            continue
        cur = getattr(obj, f.name)
        setattr(obj, f.name, _from_dict(type(cur), d[f.name]) if is_dataclass(cur) else d[f.name])
    return obj
