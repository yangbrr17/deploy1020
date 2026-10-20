"""Randomised logging policy with recorded propensities (RL doc 5).

pi_0(a | x) = 1 - eps + eps / |S(x)|   if a is the rule's action
            = eps / |S(x)|             if a is another action in the safe set S(x)
where S(x) is the allowed set (mask) intersected with an optional "safe" subset. The rule's action
must itself be allowed; if it is not, the policy falls back to uniform over S(x).

Every policy in this package has the signature
    policy(x: (n, d) array, mask: (n, K) bool array, rng) -> (actions: (n,) int, propensities: (n,) float)
so logging policies, learned greedy policies and Thompson samplers are interchangeable.
"""
from __future__ import annotations

import numpy as np


def epsilon_rule(rule, eps: float, safe=None):
    """Wrap a deterministic rule(x, mask) -> actions into an epsilon-randomised logging policy.
    safe: optional (n, K) bool function safe(x, mask) restricting where exploration may go."""
    def policy(x, mask, rng):
        allowed = mask if safe is None else (mask & safe(x, mask))
        allowed = np.where(allowed.any(1, keepdims=True), allowed, mask)
        base = rule(x, mask)
        ok = allowed[np.arange(len(x)), base]
        k = allowed.sum(1)
        probs = allowed * (eps / k)[:, None]
        probs[np.arange(len(x))[ok], base[ok]] += 1 - eps
        probs[~ok] = allowed[~ok] / k[~ok, None]              # rule action not allowed -> uniform
        a = (probs.cumsum(1) > rng.random((len(x), 1))).argmax(1)
        return a, probs[np.arange(len(x)), a]
    return policy


def greedy(scores_fn):
    """Deterministic policy from a score function scores_fn(x) -> (n, K); propensity 1."""
    def policy(x, mask, rng):
        s = np.where(mask, scores_fn(x), -np.inf)
        a = s.argmax(1)
        return a, np.ones(len(x))
    return policy


def uniform(x, mask, rng):
    """Uniform over allowed actions (useful for coverage checks)."""
    probs = mask / mask.sum(1, keepdims=True)
    a = (probs.cumsum(1) > rng.random((len(x), 1))).argmax(1)
    return a, probs[np.arange(len(x)), a]


def greedy_rule(rule):
    """A deterministic rule(x, mask) -> actions as a package policy (propensity 1)."""
    def policy(x, mask, rng):
        return rule(x, mask), np.ones(len(x))
    return policy
