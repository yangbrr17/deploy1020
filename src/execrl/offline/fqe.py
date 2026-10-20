"""Fitted Q evaluation (RL doc 7.4; Le, Voloshin, Yue 2019).

For a FIXED policy pi (a function of state and mask), regress
    y_i = r_i + (1 - done_i) * Q_pi(s'_i, pi(s'_i))
for n_iter = horizon iterations, then V(pi) = mean over initial states s_0 of Q_pi(s_0, pi(s_0)).
Use a held-out set of slices. FQE and the simulator must both show an improvement before shadow mode.
Standard error: bootstrap over groups (evaluate.cluster_bootstrap) - FQE itself gives none.
"""
from __future__ import annotations

from dataclasses import asdict

import lightgbm as lgb
import numpy as np

from ..compute import resolve
from ..config import LGBMParams
from .fqi import _xa


def fqe(tr, policy_actions, K: int, n_iter: int, params: LGBMParams | None = None) -> float:
    """tr: Transitions (held out). policy_actions(S, mask) -> actions. Returns V_hat(pi)."""
    p = asdict(params or LGBMParams(n_estimators=150))
    p["num_threads"] = resolve(p["num_threads"])
    X = _xa(tr.S, tr.A, K)
    a2 = policy_actions(tr.S2, tr.mask2)
    X2 = _xa(tr.S2, a2, K)
    q = None
    for _ in range(n_iter):
        target = tr.R if q is None else tr.R + (1 - tr.done) * q.predict(X2)
        q = lgb.LGBMRegressor(**p).fit(X, target)
    s0 = tr.step == 0
    a0 = policy_actions(tr.S[s0], tr.mask[s0])
    return float(q.predict(_xa(tr.S[s0], a0, K)).mean())
