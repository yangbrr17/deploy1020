"""Policy tree on DR scores (RL doc 6.4; Athey and Wager 2021).

DR score of arm a for row i:
    Gamma_i(a) = mu_hat(x_i, a) + 1[a_i = a] / p_i * (r_i - mu_hat(x_i, a_i))
Policy tree of depth D: argmax over trees pi in Pi_D of sum_i Gamma_i(pi(x_i)).

This implementation is EXACT for depth 1 and 2 over a finite threshold grid (quantiles of each
feature): every leaf picks the arm with the largest summed score over its rows.
Masks: an arm that is illegal for a row gets a large penalty on that row, so a leaf picks an arm that
is legal for (almost) all of its rows; at deployment apply the mask again and fall back to the best
legal arm by mu_hat (train.py does this).
Cost: depth 2 ~ (d * G)^2 * K cumulative sums; with d = 8 features and G = 16 thresholds it takes
seconds for 1e5 rows. For deeper trees use the `policytree` R package or a greedy CART on regrets.
"""
from __future__ import annotations

import numpy as np

PENALTY = 1e6       # score subtracted for a masked arm, so leaves never pick an arm that is often illegal


def dr_scores(mu, a, r, p, mask=None, clip: float = 20.0) -> np.ndarray:
    n, K = mu.shape
    G = mu.copy()
    w = np.minimum(1.0 / p, clip)
    G[np.arange(n), a] += w * (r - mu[np.arange(n), a])
    if mask is not None:
        G = np.where(mask, G, G - PENALTY)
    return G


def _best_leaf(G):
    s = G.sum(0)
    k = int(s.argmax())
    return k, float(s[k])


def _best_split(X, G, grid):
    """Best depth-1 split: returns (value, feature, threshold, left_arm, right_arm)."""
    best = (_best_leaf(G)[1], None, None, _best_leaf(G)[0], None)
    total = G.sum(0)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, cum = X[order, j], np.cumsum(G[order], axis=0)
        for c in grid[j]:
            i = np.searchsorted(xs, c, side="right")          # rows with x_j <= c go left
            if i == 0 or i == len(xs):
                continue
            left, right = cum[i - 1], total - cum[i - 1]
            v = left.max() + right.max()
            if v > best[0]:
                best = (float(v), j, float(c), int(left.argmax()), int(right.argmax()))
    return best


class PolicyTree:
    """Depth 0, 1 or 2. tree = {'leaf': arm} or {'j', 'c', 'left', 'right'}."""

    def __init__(self, depth: int = 2, n_grid: int = 16):
        self.depth, self.n_grid = depth, n_grid

    def fit(self, X, G):
        qs = np.linspace(0, 1, self.n_grid + 2)[1:-1]
        self.grid = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.tree = self._fit(X, G, self.depth)
        return self

    def _fit(self, X, G, depth):
        if depth == 0 or len(X) < 2:
            return {"leaf": _best_leaf(G)[0]}
        if depth == 1:
            v, j, c, la, ra = _best_split(X, G, self.grid)
            return {"leaf": la} if j is None else {"j": j, "c": c, "left": {"leaf": la}, "right": {"leaf": ra}}
        best_v, best = _best_leaf(G)[1], {"leaf": _best_leaf(G)[0]}
        for j in range(X.shape[1]):
            for c in self.grid[j]:
                L = X[:, j] <= c
                if L.all() or not L.any():
                    continue
                vl, *restl = _best_split(X[L], G[L], self.grid)
                vr, *restr = _best_split(X[~L], G[~L], self.grid)
                if vl + vr > best_v:
                    best_v = vl + vr
                    best = {"j": j, "c": float(c), "left": self._leafify(restl), "right": self._leafify(restr)}
        return best

    @staticmethod
    def _leafify(rest):
        j, c, la, ra = rest
        return {"leaf": la} if j is None else {"j": j, "c": c, "left": {"leaf": la}, "right": {"leaf": ra}}

    def predict(self, X) -> np.ndarray:
        out = np.empty(len(X), int)
        for i, x in enumerate(X):
            node = self.tree
            while "leaf" not in node:
                node = node["left"] if x[node["j"]] <= node["c"] else node["right"]
            out[i] = node["leaf"]
        return out

    def rules(self, feature_names, arm_names, node=None, indent="") -> str:
        """Human-readable if/else text for model review."""
        node = node or self.tree
        if "leaf" in node:
            return f"{indent}-> {arm_names[node['leaf']]}\n"
        f = feature_names[node["j"]]
        return (f"{indent}if {f} <= {node['c']:.4g}:\n" + self.rules(feature_names, arm_names, node["left"], indent + "    ")
                + f"{indent}else:\n" + self.rules(feature_names, arm_names, node["right"], indent + "    "))
