"""Behaviour-policy model pi_beta(a | s) (RL doc 7.1).

Used only to define the SUPPORT of the logs: which actions were taken often enough in states like s
to trust Q(s, a). When the logs contain recorded propensities for exactly these states you can use
them instead, but a fitted model also covers the next states S2 (where nothing was logged yet).
Output always has K columns, even if some action never appears in the training data.
"""
from __future__ import annotations

from dataclasses import asdict

import lightgbm as lgb
import numpy as np

from ..compute import resolve
from ..config import LGBMParams


class BehaviourModel:
    def __init__(self, K: int, params: LGBMParams | None = None):
        p = asdict(params or LGBMParams(n_estimators=150))
        p["num_threads"] = resolve(p["num_threads"])
        self.K, self.params = K, p

    def fit(self, S, A, mask=None):
        self.classes_ = np.unique(A)
        if len(self.classes_) == 1:
            self.model = None
            return self
        self.model = lgb.LGBMClassifier(**self.params).fit(S, A)
        return self

    def predict_proba(self, S, mask=None) -> np.ndarray:
        """(n, K) probabilities; masked actions set to 0 and rows renormalised."""
        P = np.zeros((len(S), self.K))
        if self.model is None:
            P[:, self.classes_[0]] = 1.0
        else:
            P[:, self.model.classes_] = self.model.predict_proba(S)
        if mask is not None:
            P = P * mask
            s = P.sum(1, keepdims=True)
            P = np.where(s > 0, P / np.where(s > 0, s, 1), mask / mask.sum(1, keepdims=True))
        return P
