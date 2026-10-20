"""Non-learned baselines every learned policy must beat (RL doc 9).

- twap_schedule / almgren_chriss_trajectory: the scheduling layer (how much per slice).
- slice_rule: the deterministic in-slice rule used as the logging policy's base ("join, take when behind").
- one_step_rule: the supervised-signal decision of ml-pipeline 2.8 = the one-step approximation of Q
  (RL doc 2.5): take if the expected cost of waiting exceeds the cost of crossing now.
- macro_rule: the bandit's base rule (join_hold2 unless behind or the signal is strongly against us).
"""
from __future__ import annotations

import numpy as np

from .actions import Action

# observation column indices (simulator.OBS_NAMES)
REM, TLEFT, BAND, WIDE, SUP, SDN, HAS, DIST = range(8)


def twap_schedule(total: float, n_slices: int) -> np.ndarray:
    return np.full(n_slices, total / n_slices)


def almgren_chriss_trajectory(X: float, T: float, n: int, sigma: float, eta: float, lam: float) -> np.ndarray:
    """Remaining holdings x(t_k) = X sinh(kappa (T - t)) / sinh(kappa T), kappa = sqrt(lam sigma^2 / eta);
    lam -> 0 gives the straight TWAP line."""
    t = np.linspace(0, T, n + 1)
    kappa = np.sqrt(lam * sigma ** 2 / eta)
    if kappa * T < 1e-8:
        return X * (1 - t / T)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


def slice_rule(x, mask):
    """Join the bid; keep an existing order; take when only TAKE/IMPROVE are allowed (behind)."""
    a = np.where(x[:, HAS] > 0, Action.HOLD, Action.JOIN).astype(int)
    behind = ~mask[:, Action.JOIN]
    a[behind] = np.where(mask[behind, Action.IMPROVE], Action.IMPROVE, Action.TAKE)
    return np.where(mask[np.arange(len(a)), a], a, mask.argmax(1))


def one_step_rule(threshold: float = 0.05):
    """Take when the signal says the price is about to rise (sig_up - sig_dn > threshold), else join."""
    def rule(x, mask):
        a = np.where(x[:, SUP] - x[:, SDN] > threshold, Action.TAKE, Action.JOIN).astype(int)
        a = np.where((a == Action.JOIN) & (x[:, HAS] > 0), Action.HOLD, a)
        return np.where(mask[np.arange(len(a)), a], a, mask.argmax(1))
    return rule


def macro_rule(x, mask):
    """Bandit base rule over DEFAULT_MACROS: join_hold2 (index 1); take_now (0) if not allowed."""
    a = np.full(len(x), 1)
    return np.where(mask[np.arange(len(a)), a], a, mask.argmax(1))
