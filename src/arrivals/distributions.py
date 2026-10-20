"""Marginal distribution of durations: which family fits?

Candidate families (densities for x > 0, all with a free scale; durations are first normalised to
unit mean so the scale is ~1 and the shape parameters are comparable across days):

    exponential          f = (1/s) e^{-x/s}                                   1 parameter
    Weibull              f = (k/s)(x/s)^{k-1} e^{-(x/s)^k}                    2   (k < 1: more short AND long gaps)
    gamma                f = x^{a-1} e^{-x/s} / (Gamma(a) s^a)                2
    log-normal           log x ~ N(m, sigma^2)                                2
    generalised gamma    f = |c| x^{ca-1} e^{-(x/s)^c} / (Gamma(a) s^{ca})    3   (Stacy 1962)
    Burr XII             F = 1 - (1 + (x/s)^c)^{-d}                           3   (heavy tail; Burr-ACD)

Nesting (so LR tests apply): exponential = Weibull(k=1) = gamma(a=1); Weibull = GG(a=1); gamma = GG(c=1).
Log-normal is a limit of GG (a -> inf) and Burr is not nested with GG, so use Vuong for those.

IMPORTANT: these fits treat durations as iid. Raw durations are autocorrelated (H1 rejects that), so
use this module on ACD residuals (H4) to choose the error distribution, and only descriptively on raw
durations. Standard errors that ignore the dependence are too small.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from . import stats as st


def _fit(dist, x, *start):
    """MLE with location fixed at 0; `start` are optional starting shape values (positional, scipy API)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = dist.fit(x, *start, floc=0)
    ll_i = dist.logpdf(x, *params)
    return params, ll_i


def fit_all(x, max_n: int | None = 200_000, rng=None) -> dict:
    """Fit every family by MLE. Returns {name: {params, ll, k, aic, bic, ll_i}}.
    `max_n` subsamples very long series for speed (the fits are O(n) per likelihood evaluation)."""
    x = np.asarray(x, float)
    x = x[x > 0]
    if max_n and len(x) > max_n:
        x = (rng or np.random.default_rng(0)).choice(x, max_n, replace=False)
    x = x / x.mean()
    n = len(x)
    out = {}

    lam = 1.0 / x.mean()
    out["exponential"] = {"params": (lam,), "ll_i": np.log(lam) - lam * x, "k": 1}

    p, ll = _fit(stats.weibull_min, x)
    out["weibull"] = {"params": p, "ll_i": ll, "k": 2}
    p, ll = _fit(stats.gamma, x)
    out["gamma"] = {"params": p, "ll_i": ll, "k": 2}
    p, ll = _fit(stats.lognorm, x)
    out["lognormal"] = {"params": p, "ll_i": ll, "k": 2}
    # generalised gamma: start from the Weibull fit (a = 1, c = k_weibull) to help the optimiser.
    c0 = out["weibull"]["params"][0]
    p, ll = _fit(stats.gengamma, x, 1.0, c0)
    out["gengamma"] = {"params": p, "ll_i": ll, "k": 3}
    try:
        p, ll = _fit(stats.burr12, x)
        out["burr12"] = {"params": p, "ll_i": ll, "k": 3}
    except Exception:  # scipy's generic fit sometimes fails on very clean exponential data
        pass

    for v in out.values():
        v["ll"] = float(np.sum(v["ll_i"]))
        v["aic"] = st.aic(v["ll"], v["k"])
        v["bic"] = st.bic(v["ll"], v["k"], n)
    return out


NESTED = [("exponential", "weibull"), ("exponential", "gamma"), ("weibull", "gengamma"), ("gamma", "gengamma")]
NON_NESTED = [("weibull", "lognormal"), ("gengamma", "burr12"), ("gamma", "lognormal")]


def compare(fits: dict) -> list[dict]:
    """LR tests for nested pairs, Vuong tests for non-nested pairs, ranked by BIC."""
    rows = []
    for a, b in NESTED:
        if a in fits and b in fits:
            r = st.lr_test(fits[a]["ll"], fits[b]["ll"], fits[b]["k"] - fits[a]["k"])
            rows.append({"test": f"LR {a} vs {b}", **r})
    for a, b in NON_NESTED:
        if a in fits and b in fits:
            r = st.vuong_test(fits[a]["ll_i"], fits[b]["ll_i"], fits[a]["k"], fits[b]["k"])
            rows.append({"test": f"Vuong {a} vs {b} (z>0 favours {a})", **r})
    best = min(fits, key=lambda k: fits[k]["bic"])
    rows.append({"test": "BIC ranking", "stat": fits[best]["bic"], "p_value": np.nan,
                 "effect": np.nan, "effect_name": "best=" + best,
                 "note": ", ".join(sorted(fits, key=lambda k: fits[k]["bic"]))})
    return rows
