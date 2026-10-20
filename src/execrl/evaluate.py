"""Evaluation utilities (RL doc 11): sample size, design effect, CUPED, cluster bootstrap."""
from __future__ import annotations

import numpy as np
from scipy import stats


def sample_size(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm n for a two-sided test: n = 2 (z_{1-alpha/2} + z_power)^2 sigma^2 / delta^2."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * z ** 2 * sigma ** 2 / delta ** 2))


def effective_n(n: int, m_bar: float, icc: float) -> float:
    """Design effect for clustered decisions: n_eff = n / (1 + (m_bar - 1) rho)."""
    return n / (1 + (m_bar - 1) * icc)


def cuped(y, x):
    """Y_cuped = Y - theta (X - mean X), theta = Cov(Y, X) / Var(X). Returns (adjusted y, theta)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def cluster_bootstrap(stat_fn, group, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of stat_fn(row_indices) resampling whole groups with replacement."""
    rng = np.random.default_rng(seed)
    g, inv = np.unique(group, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(g))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(g), len(g))
        vals.append(stat_fn(np.concatenate([rows[k] for k in pick])))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(vals)), float(lo), float(hi)


def simulate_value(policy, slice_cfg, n_episodes: int = 10_000, seed: int = 123) -> tuple[float, float]:
    """Ground-truth value of a policy in the (toy) simulator: mean slice return and its standard error."""
    from .simulator import episode_returns, rollout
    ret = episode_returns(rollout(slice_cfg, policy, n_episodes, seed=seed))
    return float(ret.mean()), float(ret.std(ddof=1) / np.sqrt(len(ret)))
