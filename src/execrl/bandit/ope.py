"""Off-policy evaluation (RL doc 6.3): IPW, self-normalised IPW and doubly robust (DR).

For a target policy with action probabilities pi[i, a] (deterministic policy: one-hot) and logged
(a_i, r_i, p_i = pi_0(a_i | x_i)):

    w_i      = min(pi[i, a_i] / p_i, M)
    IPW      = mean(w_i r_i)
    SNIPW    = sum(w_i r_i) / sum(w_i)
    Gamma_i  = sum_a pi[i, a] mu_hat[i, a] + w_i (r_i - mu_hat[i, a_i])        DR = mean(Gamma_i)

Standard errors cluster by group (decisions of one parent order are correlated):
    se = sqrt( sum_g (sum_{i in g} (Gamma_i - DR))^2 ) / n
Deployment gate: DR(new) - DR(logging) has a one-sided 95% lower bound > 0, with the difference
computed per row (paired) so the common noise cancels.
"""
from __future__ import annotations

import numpy as np


def one_hot(actions, K: int) -> np.ndarray:
    return np.eye(K)[np.asarray(actions, int)]


def cluster_se(terms, group) -> float:
    """Cluster-robust standard error of mean(terms)."""
    terms = np.asarray(terms, float)
    dev = terms - terms.mean()
    _, inv = np.unique(group, return_inverse=True)
    s = np.bincount(inv, weights=dev)
    return float(np.sqrt(np.sum(s ** 2)) / len(terms))


def dr_terms(pi, a, r, p, mu, clip: float = 20.0) -> np.ndarray:
    """Per-row DR contributions Gamma_i (their mean is the DR value)."""
    n = len(a)
    w = np.minimum(pi[np.arange(n), a] / p, clip)
    return np.sum(pi * mu, axis=1) + w * (r - mu[np.arange(n), a])


def evaluate(pi, a, r, p, mu, group, clip: float = 20.0) -> dict:
    """All three estimators for one target policy. pi: (n, K) probabilities."""
    n = len(a)
    w = np.minimum(pi[np.arange(n), a] / p, clip)
    g = dr_terms(pi, a, r, p, mu, clip)
    return {"ipw": float(np.mean(w * r)), "snipw": float(np.sum(w * r) / np.sum(w)),
            "dr": float(g.mean()), "dr_se": cluster_se(g, group),
            "eff_n": float(w.sum() ** 2 / np.sum(w ** 2)),     # effective sample size of the weights
            "match_rate": float(np.mean(pi[np.arange(n), a] > 0))}


def logging_policy_value(r, group) -> dict:
    """On-policy value of the logging policy itself: the plain mean reward."""
    return {"dr": float(np.mean(r)), "dr_se": cluster_se(r, group)}


def gate(pi_new, a, r, p, mu, group, clip: float = 20.0, z: float = 1.645) -> dict:
    """Paired comparison new vs logging policy. Logging value per row is r_i itself (on-policy)."""
    diff = dr_terms(pi_new, a, r, p, mu, clip) - r
    d, se = float(diff.mean()), cluster_se(diff, group)
    return {"diff": d, "se": se, "lower": d - z * se, "pass": bool(d - z * se > 0)}


def clip_sensitivity(pi, a, r, p, mu, group, clips=(5, 10, 20, 50, 1e9)) -> list[dict]:
    """DR as a function of the weight clip M (RL doc 6.7): pick M where the curve flattens."""
    return [{"clip": c, **evaluate(pi, a, r, p, mu, group, c)} for c in clips]
