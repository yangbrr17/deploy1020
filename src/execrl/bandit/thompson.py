"""Linear Thompson sampling inside the safe set (RL doc 6.5), for the online phase.

Per arm a: r = theta_a' x + noise, noise ~ N(0, s2), prior theta_a ~ N(0, lam^-1 I).
Posterior after data: A_a = lam I + sum x x' / s2,  b_a = sum r x / s2,  theta_a ~ N(A_a^-1 b_a, A_a^-1).
Act: draw theta_a for each allowed arm, pick argmax theta_a' x.

The propensity of the chosen arm is NOT available in closed form; act() estimates it by
Monte Carlo (n_mc posterior draws) and returns it, because the next offline evaluation needs it.
Log it with the decision. Add a bias column (x with a trailing 1) yourself if you want an intercept.
"""
from __future__ import annotations

import numpy as np


class LinearThompson:
    def __init__(self, n_arms: int, dim: int, lam: float = 1.0, noise_var: float = 1.0, seed: int = 0):
        self.A = np.stack([lam * np.eye(dim) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))
        self.s2 = noise_var
        self.rng = np.random.default_rng(seed)

    def _draw(self, k, size):
        cov = np.linalg.inv(self.A[k])
        return self.rng.multivariate_normal(cov @ self.b[k], cov, size=size)   # (size, dim)

    def act(self, x, allowed, n_mc: int = 200):
        """One decision. Returns (arm, estimated propensity of that arm)."""
        arms = np.flatnonzero(allowed)
        scores = np.stack([self._draw(k, n_mc) @ x for k in arms], axis=1)       # (n_mc, |arms|)
        winners = arms[scores.argmax(1)]
        arm = int(winners[0])                       # the first draw is the actual decision
        prop = max(np.mean(winners == arm), 1.0 / n_mc)
        return arm, float(prop)

    def update(self, x, a, r):
        self.A[a] += np.outer(x, x) / self.s2
        self.b[a] += r * x / self.s2
