"""Containers for logged data, validation and leakage-free splits.

BanditLog: one row per logged decision (RL doc 5-6).
    X      (n, d)  context features at decision time (only information available at that time!)
    A      (n,)    chosen arm / action index
    P      (n,)    propensity pi_0(A | X) as RECORDED by the logging policy (not re-estimated)
    R      (n,)    reward in bp (= -cost)
    mask   (n, K)  allowed arms at X (the same mask the logging policy used)
    group  (n,)    cluster id: the parent order (or day) the decision belongs to

Transitions: one row per step of a slice (RL doc 7).
    S, A, R, S2, done, mask (allowed at S), mask2 (allowed at S2), prop (logged propensity),
    group (episode / parent-order id), step (decision index within the slice)

Real logs: build these from your decision log table with `from_frame`; the column names are
arguments so nothing here assumes a particular storage format.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


def _check(obj, two_d=("X", "S", "S2", "mask", "mask2")):
    n = None
    for f in fields(obj):
        v = getattr(obj, f.name)
        if v is None:
            continue
        if n is None:
            n = len(v)
        if len(v) != n:
            raise ValueError(f"{type(obj).__name__}.{f.name}: length {len(v)} != {n}")
        if f.name in two_d and np.ndim(v) != 2:
            raise ValueError(f"{type(obj).__name__}.{f.name} must be 2-D")
    return n


@dataclass
class BanditLog:
    X: np.ndarray
    A: np.ndarray
    P: np.ndarray
    R: np.ndarray
    mask: np.ndarray
    group: np.ndarray

    def __post_init__(self):
        n = _check(self)
        self.A = self.A.astype(int)
        if np.any(self.P <= 0) or np.any(self.P > 1):
            raise ValueError("propensities must be in (0, 1]; zero means the logged arm was impossible")
        if not self.mask[np.arange(n), self.A].all():
            raise ValueError("a logged arm is outside its own mask: mask and log are inconsistent")

    def __len__(self):
        return len(self.R)

    def subset(self, idx) -> "BanditLog":
        return BanditLog(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @classmethod
    def from_frame(cls, df, feature_cols, arm_col, prop_col, reward_col, mask_cols, group_col):
        return cls(X=df[feature_cols].to_numpy(float), A=df[arm_col].to_numpy(int),
                   P=df[prop_col].to_numpy(float), R=df[reward_col].to_numpy(float),
                   mask=df[mask_cols].to_numpy(bool), group=df[group_col].to_numpy())


@dataclass
class Transitions:
    S: np.ndarray
    A: np.ndarray
    R: np.ndarray
    S2: np.ndarray
    done: np.ndarray
    mask: np.ndarray
    mask2: np.ndarray
    prop: np.ndarray
    group: np.ndarray
    step: np.ndarray

    def __post_init__(self):
        _check(self)
        self.A = self.A.astype(int)

    def __len__(self):
        return len(self.R)

    def subset(self, idx) -> "Transitions":
        return Transitions(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def initial(self) -> "Transitions":
        """Rows at the first decision of each slice (D_0 in FQE)."""
        return self.subset(self.step == 0)


def split_by_group(group, test_frac: float, seed: int = 0):
    """Train / test row indices with whole groups on one side (no leakage across a parent order).
    For time series use split_by_time instead when groups are days."""
    g = np.unique(group)
    rng = np.random.default_rng(seed)
    test_g = rng.choice(g, int(round(test_frac * len(g))), replace=False)
    is_test = np.isin(group, test_g)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def split_by_time(day, test_days: int):
    """The last `test_days` distinct days go to test (walk-forward). `day` sortable (YYYY-MM-DD)."""
    days = np.unique(day)
    is_test = np.isin(day, days[-test_days:])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def group_folds(group, n_folds: int, seed: int = 0):
    """Fold id per row such that each group sits in one fold (for cross-fitting)."""
    g, inv = np.unique(group, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(g))
    return (perm % n_folds)[inv]
