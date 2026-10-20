"""Cont-de Larrard (2013) queue-depletion formulas and their empirical inputs (H6-H8).

Model (one tick spread, large-tick stock): each best queue is a birth-death process in units of
one average order. Births (limit orders) at rate lam, deaths (market orders + cancellations) at rate
c = mu + theta. The mid moves when either best queue hits 0. Starting from x orders:

    Laplace transform  E[e^{-s sigma}] = ( (lam + c + s - sqrt((lam + c + s)^2 - 4 lam c)) / (2 lam) )^x
    density            f_x(t) = (x / t) (c / lam)^{x/2} I_x(2 sqrt(lam c) t) e^{-(lam + c) t}
    time to next mid change   tau = min(sigma_bid, sigma_ask),   P(tau > t) = S_bid(t) S_ask(t)

Balanced flow (lam = c): S_x(t) ~ x / sqrt(pi lam t), so P(tau > t) ~ t^{-1} (tail index 1, infinite
mean). Imbalanced (lam < c): exponential cut-off at rate (sqrt(c) - sqrt(lam))^2.
Direction (balanced): P(next move up | bid = n, ask = p) = phi(n, p), an integral (see up_probability).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import integrate, special, stats
from scipy.integrate import cumulative_trapezoid

from .schema import Session, l1_queue_flows


def depletion_density(t, x, lam, c):
    """f_x(t) above; special.ive(x, z) = I_x(z) e^{-z} keeps it finite for large z."""
    t = np.asarray(t, float)
    z = 2.0 * np.sqrt(lam * c) * t
    return (x / t) * (c / lam) ** (x / 2.0) * special.ive(x, z) * np.exp(z - (lam + c) * t)


def survival_on_grid(x: int, lam: float, c: float, t_grid: np.ndarray) -> np.ndarray:
    """S_x(t) = 1 - int_0^t f_x on a (log-spaced) grid by the trapezoid rule. Fast enough to tabulate
    every queue size 1..x_max once per symbol-day. For lam > c it tends to 1 - (c/lam)^x, not 0."""
    f = depletion_density(t_grid, x, lam, c)
    F = cumulative_trapezoid(f, t_grid, initial=0.0)
    return np.clip(1.0 - F, 0.0, 1.0)


@lru_cache(maxsize=4096)
def up_probability(n_bid: int, p_ask: int) -> float:
    """phi(n, p) = P(ask queue depletes first) = P(next mid move is UP), balanced flow:
    phi(n, p) = (1/pi) int_0^pi (2 - cos t - sqrt((2 - cos t)^2 - 1))^p  sin(n t) cos(t/2) / sin(t/2) dt.
    Checked against simulation: phi(5, 2) = 0.756 (sim 0.759), phi(10, 1) = 0.936 (sim 0.931)."""
    def f(t):
        r = 2 - np.cos(t)
        return (r - np.sqrt(r * r - 1)) ** p_ask * np.sin(n_bid * t) * np.cos(t / 2) / np.sin(t / 2)
    val, _ = integrate.quad(f, 0, np.pi, limit=400)
    return float(val / np.pi)


def estimate_rates(quotes, trades, s: Session) -> dict:
    """lam (adds / s) and c (removals / s) per side in units of the average added order size.
    Uses schema.l1_queue_flows (L1 approximation); replace with L2 counts when available."""
    fl = l1_queue_flows(quotes, trades, s)
    out = {}
    for side in ("bid", "ask"):
        d = fl[side]
        unit = d["added"] / max(d["n_adds"], 1)            # average order size, shares
        out[side] = {"lam": d["added"] / unit / fl["seconds"],
                     "c": (d["traded"] + d["cancelled"]) / unit / fl["seconds"],
                     "unit": unit}
    return out


def first_passage_sample(mids: dict, t_end: float, unit_bid: float, unit_ask: float, q_max: int = 50) -> dict:
    """From the mid-change events (schema.mid_change_events): for each change k, the waiting time to
    change k+1, the queues right after change k in order units (rounded, clipped to [1, q_max]) and
    whether change k+1 was up. The last change of the day has no successor and is dropped."""
    t = mids["times"]
    qb = np.clip(np.rint(mids["bid_sz"] / unit_bid), 1, q_max).astype(int)
    qa = np.clip(np.rint(mids["ask_sz"] / unit_ask), 1, q_max).astype(int)
    return {"tau": np.diff(t), "q_bid": qb[:-1], "q_ask": qa[:-1],
            "up": (mids["direction"][1:] > 0).astype(int), "spread": mids["spread"][:-1]}


def pit_first_passage(fp: dict, rates: dict, n_grid: int = 4000) -> np.ndarray:
    """Probability integral transform U_k = P_model(tau > tau_k | q_bid_k, q_ask_k).
    Under the model U ~ Uniform(0, 1); test with KS. Survival curves are tabulated per queue size."""
    tau = fp["tau"]
    lo = 1e-6 / max(rates["bid"]["lam"] + rates["bid"]["c"], 1e-9)
    t_grid = np.geomspace(lo, max(tau.max(), 1.0) * 1.01, n_grid)
    sb = {q: survival_on_grid(q, rates["bid"]["lam"], rates["bid"]["c"], t_grid) for q in np.unique(fp["q_bid"])}
    sa = {q: survival_on_grid(q, rates["ask"]["lam"], rates["ask"]["c"], t_grid) for q in np.unique(fp["q_ask"])}
    u = np.empty(len(tau))
    for i, (tt, b, a) in enumerate(zip(tau, fp["q_bid"], fp["q_ask"])):
        u[i] = np.interp(tt, t_grid, sb[b]) * np.interp(tt, t_grid, sa[a])
    return u


def uniform_ks(u) -> dict:
    """KS of the PIT values against Uniform(0,1). Effect size D."""
    r = stats.kstest(u, "uniform")
    return {"stat": r.statistic, "p_value": r.pvalue, "effect": r.statistic, "effect_name": "KS D", "n": len(u)}


def queue_reactive_lr(quotes, s: Session, n_bins: int = 5, to_business=None) -> list[dict]:
    """H6: do add / removal intensities at the best queue depend on the queue size?

    From L1 snapshots, per side: each interval between two snapshots with the SAME best price has a
    queue size q_k (displayed size), an exposure dt_k and ends with an add (size up) or a removal
    (size down). Intervals that end with a price change are censored (exposure, no event).
    Pooling both sides, bin q by time-weighted quantiles into B bins, and test
        H0: lambda_e(q) = lambda_e      vs      H1: lambda_e(q) = lambda_{e,b} for q in bin b
    with the Poisson-process likelihood l = sum_b (N_b log lambda_b - lambda_b T_b), LR ~ chi2(B - 1).
    Effect size: max_b lambda_b / min_b lambda_b. Queue-reactive (Huang et al. 2015) predicts strong
    dependence (removals per unit time rise with q; adds often fall with q).

    Seasonality confounds this test (busy hours have both higher rates and different queue sizes),
    so pass `to_business` (seconds -> business time, e.g. models.business_time with the intraday
    curve) and exposures are measured in business time."""
    from .schema import clean_quotes, clip_session
    from . import stats as st
    q = clean_quotes(clip_session(quotes, s))
    t = (q["ts"].to_numpy() - s.t_open) / 1e9
    if to_business is not None:
        t = to_business(t)
    dt = np.diff(t)
    sizes, exp_, add_, rem_ = [], [], [], []
    for px, sz in (("bid_px", "bid_sz"), ("ask_px", "ask_sz")):
        p, v = q[px].to_numpy(), q[sz].to_numpy()
        same = p[1:] == p[:-1]
        dv = np.diff(v)
        sizes.append(v[:-1]); exp_.append(dt)
        add_.append(same & (dv > 0)); rem_.append(same & (dv < 0))
    size, expo = np.concatenate(sizes), np.concatenate(exp_)
    is_add, is_rem = np.concatenate(add_), np.concatenate(rem_)
    order = np.argsort(size)
    cum = np.cumsum(expo[order]) / expo.sum()
    edges = np.interp(np.linspace(0, 1, n_bins + 1)[1:-1], cum, size[order])
    b = np.digitize(size, np.unique(edges))
    rows = []
    for name, ev in (("add", is_add), ("remove", is_rem)):
        T_b = np.bincount(b, weights=expo)
        N_b = np.bincount(b, weights=ev.astype(float), minlength=len(T_b))
        ok = T_b > 0
        lam_b = N_b[ok] / T_b[ok]
        lam0 = N_b.sum() / T_b.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            ll1 = np.sum(np.where(N_b[ok] > 0, N_b[ok] * np.log(lam_b), 0.0) - lam_b * T_b[ok])
        ll0 = N_b.sum() * np.log(lam0) - lam0 * T_b.sum()
        r = st.lr_test(ll0, ll1, int(ok.sum()) - 1)
        ratio = lam_b.max() / max(lam_b.min(), 1e-12)
        rows.append({"test": f"LR rate({name}) constant in queue size, {ok.sum()} bins", **r,
                     "effect": ratio, "effect_name": "max/min rate",
                     "note": "rates by bin: " + ", ".join(f"{x:.3g}" for x in lam_b)})
    return rows


def phi_for_sample(fp: dict, q_cap: int = 30) -> np.ndarray:
    """phi(q_bid, q_ask) for every first passage (cached per pair). Queues are capped at q_cap to bound
    the number of distinct integrals; check the share of capped queues and raise q_cap if it is large."""
    return np.array([up_probability(int(min(b, q_cap)), int(min(a, q_cap)))
                     for b, a in zip(fp["q_bid"], fp["q_ask"])])
