"""Test statistics used by the arrival hypotheses. Pure numpy/scipy, no state.

Every function returns a plain dict so results can go straight into the results table.
Conventions: `x` are durations (seconds or any positive unit), `e` are residuals that should be
iid Exp(1) under the null (generalised residuals / compensator increments), `times` are event times.

Large-sample warning: with 10^5 events a day, every p-value below will be ~0 for any real deviation.
Always read the effect size next to the p-value (CV, KS distance D, first autocorrelations, Fano factor);
the report applies practical thresholds to the effect sizes, not to p alone.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ exponential / Poisson null

def dispersion_test(x) -> dict:
    """Engle-Russell (1998) excess-dispersion test.

    Normalise to unit mean, e = x / mean(x). Under iid Exp(1): E[e] = 1, Var(e) = 1 and
    Var((e - 1)^2) = mu_4 - 1 = 9 - 1 = 8, so  z = sqrt(n / 8) * (s^2 - 1)  ->  N(0, 1).
    z > 0 means over-dispersion (clustering / heavier tail than exponential), the usual finding.
    Effect size: CV = s / mean (exponential: 1).
    """
    e = np.asarray(x, float)
    e = e / e.mean()
    n = len(e)
    s2 = e.var(ddof=1)
    z = np.sqrt(n / 8.0) * (s2 - 1.0)
    return {"stat": z, "p_value": 2 * stats.norm.sf(abs(z)), "p_over": stats.norm.sf(z),
            "effect": np.sqrt(s2), "effect_name": "CV", "n": n}


def ks_exponential(x, n_boot: int = 199, rng=None) -> dict:
    """KS distance to the exponential with the rate ESTIMATED from the data (Lilliefors problem).

    The textbook KS p-value assumes a known rate and is too conservative here, so the p-value comes
    from a parametric bootstrap: the null law of D does not depend on the rate (scale invariance),
    so simulate Exp(1) samples of the same n, re-estimate the rate, recompute D.
    Effect size: D itself (max vertical gap between empirical and fitted CDF).
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    n = len(x)

    def d_stat(v):
        v = np.sort(v / v.mean())
        f = 1.0 - np.exp(-v)
        i = np.arange(1, len(v) + 1)
        return max(np.max(i / len(v) - f), np.max(f - (i - 1) / len(v)))

    d = d_stat(x)
    boot = np.array([d_stat(rng.exponential(size=n)) for _ in range(n_boot)])
    p = (1 + np.sum(boot >= d)) / (n_boot + 1)
    return {"stat": d, "p_value": p, "effect": d, "effect_name": "KS D", "n": n}


def anderson_exponential(x) -> dict:
    """Anderson-Darling with estimated scale (scipy's tables account for the estimation).
    More weight on the tails than KS."""
    x = np.asarray(x, float)
    try:                                          # SciPy >= 1.17: p-value interpolated from the tables
        r = stats.anderson(x, dist="expon", method="interpolate")
        p = float(r.pvalue)
        # The tables stop at 1% and 15%: p is clipped there. p_censored marks "p <= 0.01".
        return {"stat": r.statistic, "p_value": p, "effect": r.statistic, "effect_name": "A2",
                "n": len(x), "p_censored": p <= 0.01,
                "note": "p <= 0.01 (table floor)" if p <= 0.01 else "p interpolated from AD tables"}
    except TypeError:                             # older SciPy: bracket the p-value by the critical values
        r = stats.anderson(x, dist="expon")
        levels = r.significance_level / 100.0     # [0.15, 0.10, 0.05, 0.025, 0.01]
        passed = r.statistic < r.critical_values
        p_upper = levels[passed].min() if passed.any() else levels.min()
        return {"stat": r.statistic, "p_value": float(p_upper), "effect": r.statistic,
                "effect_name": "A2", "n": len(x), "p_censored": not passed.any(),
                "note": "p is an upper bracket from AD tables"}


def ljung_box(z, lags: int = 20) -> dict:
    """Ljung-Box Q = n(n+2) sum_{h<=L} rho_h^2 / (n-h) ~ chi2(L) under no autocorrelation.
    Use on durations (H1), on ACD residuals and on compensator increments (H4/H5).
    Effect size: rho_1, the first autocorrelation."""
    z = np.asarray(z, float) - np.mean(z)
    n = len(z)
    denom = np.sum(z * z)
    acf = np.array([np.sum(z[h:] * z[:-h]) for h in range(1, lags + 1)]) / denom
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return {"stat": q, "p_value": stats.chi2.sf(q, lags), "effect": acf[0], "effect_name": "rho_1",
            "n": n, "acf": acf}


def count_dispersion(times, t_end: float, window: float) -> dict:
    """Index-of-dispersion test on counts in non-overlapping windows of length `window`.

    Homogeneous Poisson: counts N_1..N_k iid Poisson, so  I = (k-1) s^2 / mean ~ chi2(k-1).
    Effect size: Fano factor F = s^2 / mean (Poisson: 1 at every window length)."""
    edges = np.arange(0.0, t_end + 1e-12, window)
    counts = np.histogram(times, bins=edges)[0]
    k = len(counts)
    m, s2 = counts.mean(), counts.var(ddof=1)
    i_stat = (k - 1) * s2 / m
    return {"stat": i_stat, "p_value": stats.chi2.sf(i_stat, k - 1), "effect": s2 / m,
            "effect_name": "Fano", "n": k}


def fano_curve(times, t_end: float, windows) -> np.ndarray:
    """Fano factor as a function of window length. Poisson: flat at 1. Self-exciting / long memory:
    rises with the window. Deterministic seasonality alone also makes it rise at long windows, so
    compute it again after time-rescaling (H2)."""
    return np.array([count_dispersion(times, t_end, w)["effect"] for w in windows])


# ------------------------------------------------------------------ model comparison

def lr_test(ll_null: float, ll_alt: float, df: int, boundary: bool = False) -> dict:
    """Likelihood-ratio test of nested models. LR = 2 (ll_alt - ll_null) ~ chi2(df).

    boundary=True: the null puts ONE parameter on the edge of its space (e.g. Hawkes alpha = 0, or a
    variance = 0). Then LR ~ 0.5 chi2(0) + 0.5 chi2(1) (Self and Liang 1987), i.e. halve the p-value.
    Caveat for Hawkes: under alpha = 0 the decay beta is not identified (Davies problem); treat the
    p-value as indicative and rely on AIC plus the time-rescaling tests."""
    lr = max(2.0 * (ll_alt - ll_null), 0.0)
    p = stats.chi2.sf(lr, df)
    if boundary and df == 1:
        p *= 0.5
    return {"stat": lr, "p_value": p, "effect": ll_alt - ll_null, "effect_name": "delta loglik"}


def vuong_test(ll1_i, ll2_i, k1: int, k2: int) -> dict:
    """Vuong (1989) test for NON-nested models from per-observation log-likelihoods.

    m_i = l1_i - l2_i,  z = (sum m_i - (k1 - k2) log(n) / 2) / (sqrt(n) sd(m))  ~ N(0,1).
    z > 0 favours model 1. Requires iid observations: apply to residuals of a dependence model
    (e.g. ACD residuals) rather than raw durations."""
    m = np.asarray(ll1_i) - np.asarray(ll2_i)
    n = len(m)
    z = (m.sum() - 0.5 * (k1 - k2) * np.log(n)) / (np.sqrt(n) * m.std(ddof=1))
    return {"stat": z, "p_value": 2 * stats.norm.sf(abs(z)), "effect": m.mean(),
            "effect_name": "mean loglik diff"}


def aic(ll: float, k: int) -> float:
    return 2 * k - 2 * ll


def bic(ll: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * ll


# ------------------------------------------------------------------ tails, heterogeneity, calibration

def hill_estimator(x, k: int | None = None) -> dict:
    """Hill (1975) tail index from the k largest observations:
    alpha_hat = [ (1/k) sum_{i=1..k} log X_(n-i+1) - log X_(n-k) ]^(-1),  se ~ alpha_hat / sqrt(k).
    P(X > t) ~ t^(-alpha). Cont-de Larrard, balanced flow: alpha = 1 for the time to the next mid
    change. Default k = 2% of n (check stability over k with a Hill plot)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    k = k or max(int(0.02 * n), 20)
    tail = np.log(x[n - k:]) - np.log(x[n - k - 1])
    a = 1.0 / tail.mean()
    return {"stat": a, "se": a / np.sqrt(k), "effect": a, "effect_name": "tail index", "n": k}


def heterogeneity(est, se) -> dict:
    """Cochran's Q across days (or symbols): are daily estimates equal up to their standard errors?
    Q = sum w_d (theta_d - theta_bar)^2, w_d = 1/se_d^2, Q ~ chi2(D - 1).
    I^2 = max(0, (Q - (D-1)) / Q) is the share of variation that is real heterogeneity."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1.0 / se ** 2
    bar = np.sum(w * est) / w.sum()
    q = np.sum(w * (est - bar) ** 2)
    d = len(est)
    i2 = max(0.0, (q - (d - 1)) / q) if q > 0 else 0.0
    return {"stat": q, "p_value": stats.chi2.sf(q, d - 1), "effect": i2, "effect_name": "I2",
            "pooled": bar, "n": d}


def calibration_test(p, y) -> dict:
    """Is a predicted probability p calibrated for binary outcomes y?

    Logistic recalibration (Cox 1958): y ~ Bernoulli(sigmoid(a + b * logit(p))). Perfect calibration
    is a = 0, b = 1; Wald test of both jointly ~ chi2(2). Also returns the Brier score.
    Fitted by Newton-Raphson on the 2-parameter log-likelihood."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    z = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(z), z])
    th = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ th))
        grad = X.T @ (y - mu)
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, grad)
        th += step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(H)
    d = th - np.array([0.0, 1.0])
    w = float(d @ np.linalg.solve(cov, d))
    return {"stat": w, "p_value": stats.chi2.sf(w, 2), "intercept": th[0], "slope": th[1],
            "effect": th[1], "effect_name": "calibration slope", "brier": np.mean((p - y) ** 2),
            "n": len(y)}


def benjamini_hochberg(pvals, q: float = 0.05) -> np.ndarray:
    """BH step-up: boolean array of rejections controlling the false discovery rate at q.
    Use across symbols x days x hypotheses in the final summary."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej
