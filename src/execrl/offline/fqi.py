"""Batch-constrained fitted Q iteration with LightGBM (RL doc 7.1). Finite horizon, gamma = 1.

Iteration j:
    y_i = r_i + (1 - done_i) * max_{a' in A_beta(s'_i)} Q_j(s'_i, a')
    A_beta(s') = { a' allowed at s' : pi_beta(a'|s') >= tau * max_b pi_beta(b|s') }
    Q_{j+1} = LightGBM regression of y on [s, one-hot(a)]
With time-to-go in the state, n_iter = horizon iterations is exact backward induction.
Policy extraction uses the SAME support: argmax over allowed AND well-logged actions; otherwise the
policy can still pick actions the logs barely contain (the extrapolation error offline RL is about).
tau -> 0: plain FQI (over-optimistic); tau -> 1: imitate the most frequent logged action.
tau must be below the exploration ratio of the logs (config.FQIConfig.support_tau explains why).
"""
from __future__ import annotations

from dataclasses import asdict

import lightgbm as lgb
import numpy as np

from ..compute import resolve
from ..config import LGBMParams


def _xa(S, a, K):
    return np.hstack([S, np.eye(K)[a]])


def q_all(model, S, K: int) -> np.ndarray:
    """Q(s, a) for every action: (n, K)."""
    n = len(S)
    return np.column_stack([model.predict(_xa(S, np.full(n, k), K)) for k in range(K)])


def support(pb, tau: float) -> np.ndarray:
    return pb >= tau * pb.max(axis=1, keepdims=True)


def constrained_fqi(S, A, R, S2, done, mask2, pb2, K: int, n_iter: int, tau: float = 0.01,
                    params: LGBMParams | None = None, verbose: bool = False):
    """Returns the final Q model. pb2 = behaviour probabilities at S2 (n, K)."""
    p = asdict(params or LGBMParams(n_estimators=200))
    p["num_threads"] = resolve(p["num_threads"])
    allowed2 = mask2 & support(pb2, tau)
    X = _xa(S, A, K)
    q = None
    for j in range(n_iter):
        if q is None:
            target = R.copy()
        else:
            q2 = np.where(allowed2, q_all(q, S2, K), -np.inf).max(axis=1)
            q2 = np.where(np.isfinite(q2), q2, 0.0)            # no supported action: treat as terminal
            target = R + (1 - done) * q2
        q = lgb.LGBMRegressor(**p).fit(X, target)
        if verbose:
            print(f"FQI iter {j + 1}/{n_iter}: mean target {target.mean():.4f}")
    return q


def constrained_greedy(q, S, mask, pb, K: int, tau: float = 0.01) -> np.ndarray:
    """argmax over allowed AND supported actions (falls back to allowed if the intersection is empty)."""
    ok = mask & support(pb, tau)
    ok = np.where(ok.any(1, keepdims=True), ok, mask)
    return np.where(ok, q_all(q, S, K), -np.inf).argmax(axis=1)
