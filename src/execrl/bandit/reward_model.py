"""Direct-method reward model mu_hat(x, a) ~ E[r | x, a] (RL doc 6.2).

One LightGBM regressor on [x, one-hot(a)]. Alternative: one model per arm (better when arms behave
very differently, worse when some arms are rare). Keep the interface: fit -> model, predict_all -> (n, K).

Cross-fitting (RL doc 6.3 condition 3): DR is only unbiased if mu_hat for a row was fitted WITHOUT
that row. cross_fit() returns out-of-fold predictions, folds formed by group so a parent order's
decisions never sit on both sides.
"""
from __future__ import annotations

from dataclasses import asdict

import lightgbm as lgb
import numpy as np

from ..compute import resolve
from ..config import LGBMParams
from ..data import group_folds


def _design(X, a, K):
    return np.hstack([X, np.eye(K)[a]])


def fit_reward_model(X, a, r, K: int, params: LGBMParams | None = None, sample_weight=None):
    """Fit mu_hat. sample_weight must not depend on the outcome (e.g. time-decay weights are fine)."""
    p = asdict(params or LGBMParams())
    p["num_threads"] = resolve(p["num_threads"])
    model = lgb.LGBMRegressor(**p)
    model.fit(_design(X, a, K), r, sample_weight=sample_weight)
    return model


def predict_all(model, X, K: int) -> np.ndarray:
    """mu_hat[i, a] for every arm a: shape (n, K)."""
    n = len(X)
    return np.column_stack([model.predict(_design(X, np.full(n, k), K)) for k in range(K)])


def cross_fit(X, a, r, group, K: int, n_folds: int = 5, params: LGBMParams | None = None, seed: int = 0):
    """Out-of-fold mu_hat (n, K) and the list of fold models (average them for deployment)."""
    fold = group_folds(group, n_folds, seed)
    mu = np.empty((len(X), K))
    models = []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        m = fit_reward_model(X[tr], a[tr], r[tr], K, params)
        mu[te] = predict_all(m, X[te], K)
        models.append(m)
    return mu, models


def predict_ensemble(models, X, K: int) -> np.ndarray:
    return np.mean([predict_all(m, X, K) for m in models], axis=0)
