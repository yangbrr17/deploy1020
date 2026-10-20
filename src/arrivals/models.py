"""Seasonality, ACD and Hawkes: the models whose residuals the tests check.

Every model here produces a sequence that is iid Exp(1) when the model is right:
    NHPP (deterministic diurnal rate)   e_i = Lambda(t_i) - Lambda(t_{i-1}),  Lambda(t) = int_0^t lambda(u) du
    Weibull-ACD                         e_i = (Gamma(1+1/k) x_i / psi_i)^k     (generalised residuals)
    Hawkes                              e_i = int_{t_{i-1}}^{t_i} lambda*(u) du  (compensator increments)
That is the time-rescaling theorem (Brown et al. 2002); stats.ks_exponential / dispersion_test /
ljung_box are then applied to e.

The functions are the same as on docs/signal-model/event-arrivals.md, with seasonality added.
Speed note: the Hawkes recursion is a Python loop. Measured: a ~17k-event synthetic day fits in a few
seconds; 1e5 events should take about a minute (estimate). If that is too slow, decorate `_hawkes_A`
with numba.njit (TODO).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special
from scipy.signal import lfilter


# ------------------------------------------------------------------ utilities

def numerical_hessian(f, p, rel: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function at p (for MLE standard errors)."""
    p = np.asarray(p, float)
    k = len(p)
    H = np.empty((k, k))
    h = rel * np.maximum(np.abs(p), 1e-3)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)) / (4 * h[i] * h[j])
    return H


def mle_se(nll, p_hat) -> np.ndarray:
    """Standard errors from the inverse Hessian of the negative log-likelihood (NaN if singular)."""
    try:
        return np.sqrt(np.diag(np.linalg.inv(numerical_hessian(nll, p_hat))))
    except np.linalg.LinAlgError:
        return np.full(len(p_hat), np.nan)


# ------------------------------------------------------------------ seasonality (H2)

def diurnal_rate(times_by_day: list, day_len: float, n_bins: int = 26) -> tuple[np.ndarray, np.ndarray]:
    """Average event rate per time-of-day bin across days: lambda_hat(bin) = mean_d N_d(bin) / width.

    Pass OTHER days than the one being tested (leave-one-day-out) so the test day's own clustering is
    not absorbed into the seasonal curve. With 26 bins a 6.5 h session has 15-minute bins.
    Bins with zero events get a floor of 0.5 event so Lambda stays strictly increasing."""
    edges = np.linspace(0.0, day_len, n_bins + 1)
    counts = np.mean([np.histogram(t, bins=edges)[0] for t in times_by_day], axis=0)
    width = edges[1] - edges[0]
    return edges, np.maximum(counts, 0.5) / width


def cumulative_rate(t, edges: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Lambda(t) for a piecewise-constant rate: exact, piecewise linear in t."""
    cum = np.concatenate([[0.0], np.cumsum(rates * np.diff(edges))])
    return np.interp(t, edges, cum)


def nhpp_residuals(times, edges, rates) -> np.ndarray:
    """e_i = Lambda(t_i) - Lambda(t_{i-1}); iid Exp(1) iff the events are an NHPP with this rate.
    If the seasonal curve was estimated on other days, rescale so the test day's mean residual is 1
    (the level of activity varies across days; only the SHAPE is assumed stable)."""
    lam = cumulative_rate(np.asarray(times), edges, rates)
    e = np.diff(lam)
    return e / e.mean()


def business_time(times, edges, rates) -> np.ndarray:
    """Map clock time to business time u = Lambda(t) * T / Lambda(T): same units as seconds but with
    the intraday seasonality removed. Fit Hawkes / ACD on u (or on diurnally adjusted durations)."""
    lam = cumulative_rate(np.asarray(times), edges, rates)
    total = cumulative_rate(edges[-1], edges, rates)
    return lam * edges[-1] / total


def diurnal_adjust(x, tod, day_len: float, n_bins: int = 26):
    """Engle-Russell adjustment for durations: x_i / s(tod_i), s = E[x | time of day] from bin means
    smoothed by a natural cubic spline. Returns (adjusted durations, s)."""
    from scipy.interpolate import CubicSpline
    edges = np.linspace(0, day_len, n_bins + 1)
    idx = np.clip(np.digitize(tod, edges) - 1, 0, n_bins - 1)
    means = np.array([x[idx == j].mean() if np.any(idx == j) else np.nan for j in range(n_bins)])
    centres = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(means)
    s = CubicSpline(centres[ok], means[ok], bc_type="natural")(tod)
    s = np.maximum(s, 1e-3 * np.nanmean(means))
    return x / s, s


# ------------------------------------------------------------------ ACD (H4)

def acd_psi(x, omega, a, b):
    """psi_i = omega + a x_{i-1} + b psi_{i-1}; psi_0 = sample mean. Vectorised with lfilter."""
    u = np.empty_like(x)
    u[0] = np.mean(x)
    u[1:] = omega + a * x[:-1]
    psi = lfilter([1.0], [1.0, -b], u[1:], zi=[b * u[0]])[0]
    return np.concatenate([[u[0]], psi])


def wacd_loglik(params, x, per_obs: bool = False):
    """Weibull-ACD(1,1): x_i = psi_i eps_i, eps ~ Weibull(k) scaled to unit mean. k = 1 is EACD."""
    omega, a, b, k = params
    psi = acd_psi(x, omega, a, b)
    g = special.gamma(1.0 + 1.0 / k)
    z = g * x / psi
    ll_i = np.log(k) - np.log(x) + k * np.log(z) - z ** k
    return ll_i if per_obs else (np.sum(ll_i), psi)


def fit_wacd(x, fix_k: float | None = None) -> dict:
    """MLE over log-parameters with three starting points (the surface has flat ridges; a single
    start can stop at omega ~ 0, b ~ 1). fix_k=1.0 fits the exponential ACD (EACD).
    Returns params, loglik, standard errors and persistence a + b."""
    def unpack(th):
        omega, a, b = np.exp(th[:3])
        k = fix_k if fix_k is not None else np.exp(th[3])
        return omega, a, b, k

    def nll(th):
        omega, a, b, k = unpack(th)
        if a + b >= 0.9999:
            return 1e12
        return -wacd_loglik((omega, a, b, k), x)[0]

    best = None
    for a0, b0 in [(0.1, 0.8), (0.05, 0.9), (0.2, 0.6)]:
        x0 = [np.log((1 - a0 - b0) * np.mean(x)), np.log(a0), np.log(b0)] + ([] if fix_k else [0.0])
        res = optimize.minimize(nll, x0, method="Nelder-Mead",
                                options=dict(xatol=1e-7, fatol=1e-7, maxiter=8000))
        if best is None or res.fun < best.fun:
            best = res
    p = np.array(unpack(best.x))
    free = p if fix_k is None else p[:3]
    nll_p = (lambda q: -wacd_loglik(tuple(q), x)[0]) if fix_k is None else \
            (lambda q: -wacd_loglik((q[0], q[1], q[2], fix_k), x)[0])
    return {"params": p, "ll": -best.fun, "se": mle_se(nll_p, free), "persistence": p[1] + p[2],
            "k_free": len(free)}


def wacd_residuals(params, x) -> np.ndarray:
    """Generalised residuals (Gamma(1+1/k) x_i / psi_i)^k, iid Exp(1) if the model is right.
    For choosing the error law, also look at eps_i = x_i / psi_i (unit mean) with distributions.py."""
    omega, a, b, k = params
    psi = acd_psi(x, omega, a, b)
    return (special.gamma(1.0 + 1.0 / k) * x / psi) ** k


def acd_standardised(params, x) -> np.ndarray:
    """eps_i = x_i / psi_i (unit mean under the model): input for distributions.fit_all."""
    return x / acd_psi(x, *params[:3])


# ------------------------------------------------------------------ Hawkes (H5)

def _hawkes_A(t, beta):
    """A_i = sum_{j<i} exp(-beta (t_i - t_j)) by the Ozaki (1979) recursion A_i = e^{-beta dt}(1 + A_{i-1})."""
    decay = np.exp(-beta * np.diff(t))
    A = np.zeros_like(t)
    a = 0.0
    for i in range(1, len(t)):
        a = decay[i - 1] * (1.0 + a)
        A[i] = a
    return A


def hawkes_exp_loglik(params, t, T):
    """Exact log-likelihood of lambda*(t) = mu + alpha sum_{t_j<t} exp(-beta (t - t_j)) on [0, T]."""
    mu, alpha, beta = params
    A = _hawkes_A(t, beta)
    comp = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    return np.sum(np.log(mu + alpha * A)) - comp, A


def fit_hawkes_exp(t, T, beta0: float | None = None) -> dict:
    """MLE over log-parameters; rejects branching ratio alpha/beta >= 1 (non-stationary).
    beta0: starting decay rate (1/s); default 1 / median gap so the start is on the right time scale."""
    t = np.asarray(t, float)
    beta0 = beta0 or 1.0 / np.median(np.diff(t))

    def nll(theta):
        p = np.exp(theta)
        if p[1] / p[2] >= 0.999:
            return 1e12
        return -hawkes_exp_loglik(p, t, T)[0]

    best = None
    for n0 in (0.3, 0.7):                             # two starts on the branching ratio
        x0 = np.log([(1 - n0) * len(t) / T, n0 * beta0, beta0])
        res = optimize.minimize(nll, x0, method="Nelder-Mead",
                                options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    p = np.exp(best.x)
    se = mle_se(lambda q: -hawkes_exp_loglik(q, t, T)[0], p)
    return {"params": p, "ll": -best.fun, "se": se, "branching": p[1] / p[2],
            "half_life": np.log(2) / p[2]}


def hawkes_residuals(params, t) -> np.ndarray:
    """Compensator increments Lambda(t_{i-1}, t_i); iid Exp(1) under the true model."""
    mu, alpha, beta = params
    A = _hawkes_A(t, beta)
    dt = np.diff(t)
    return mu * dt + (alpha / beta) * (1.0 + A[:-1]) * (1.0 - np.exp(-beta * dt))


def poisson_loglik(t, T) -> float:
    """Homogeneous Poisson log-likelihood at the MLE rate N/T (the H5 null model)."""
    lam = len(t) / T
    return len(t) * np.log(lam) - lam * T
