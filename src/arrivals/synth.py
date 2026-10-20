"""Synthetic trades/quotes in the input schema, for a dry run of the whole report before real data.

What it builds into the data (so every hypothesis has something to find):
- intraday U-shaped activity (all rates multiplied by s(tod));
- market orders from an exponential Hawkes process in business time (clustering, branching ~0.6);
- limit-order adds and cancellations as Poisson at each best queue (Cont-de Larrard style), balanced
  by default (add rate = market + cancel rate per side);
- when a best queue empties the mid moves one tick and both best queues are redrawn.

It is NOT a realistic market simulator; it is a test fixture. Timestamps are rounded to `res_ns`
to mimic a feed's resolution, so tie handling in schema.py is exercised too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import NS, Session


def seasonal(u: np.ndarray, day_len: float) -> np.ndarray:
    """U-shape: about 2.2x the midday rate at the open and close (normalised so its mean is ~1)."""
    x = u / day_len - 0.5
    return (0.6 + 6.4 * x ** 2) / (0.6 + 6.4 / 12)


def _hawkes_times(mu, alpha, beta, T, rng):
    """Ogata thinning, exponential kernel (same as on the event-arrivals page)."""
    t, ev, excite = 0.0, [], 0.0
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1.0 / lam_bar)
        if t + w > T:
            return np.asarray(ev)
        excite *= np.exp(-beta * w)
        t += w
        if rng.uniform() * lam_bar <= mu + excite:
            ev.append(t)
            excite += alpha


def _to_clock(u, day_len, n=4096):
    """Invert business time: clock t with Lambda_s(t) = u, Lambda_s = integral of seasonal()."""
    grid = np.linspace(0, day_len, n)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (seasonal(grid[1:], day_len) + seasonal(grid[:-1], day_len)) * np.diff(grid))])
    cum *= day_len / cum[-1]
    return np.interp(u, cum, grid)


def make_day(seed: int = 0, day_len: float = 23400.0, t_open_ns: int = 0, tick: float = 0.01,
             lam_add: float = 1.2, theta_cancel: float = 0.8, hawkes=(0.3, 0.9, 1.5),
             lot: float = 100.0, res_ns: int = 1_000, level: float = 1.0, season: bool = True):
    """One synthetic day. Returns (trades, quotes, session).

    Rates are per side per second at average activity: adds lam_add, cancels theta_cancel; market
    orders come from Hawkes(mu, alpha, beta) in business time, split 50/50 between sides, so the
    removal rate per side is theta_cancel + mean_market_rate / 2 (balanced if it equals lam_add;
    mean market rate = mu / (1 - alpha / beta)). `level` scales all rates (day-to-day differences).
    hawkes=(rate, 0, 1) and season=False give the exact Cont-de Larrard model (Poisson everything),
    which is how tests/test_arrivals.py checks the H7/H8 code."""
    rng = np.random.default_rng(seed)
    shape = (lambda t: seasonal(t, day_len)) if season else (lambda t: np.ones_like(np.asarray(t, float)))
    mu, alpha, beta = hawkes
    mkt_u = _hawkes_times(mu * level, alpha, beta, day_len, rng)
    mkt_t = _to_clock(mkt_u, day_len) if season else mkt_u
    mkt_side = rng.choice([1, -1], size=len(mkt_t))               # +1 buy hits the ask

    # Poisson adds/cancels per side with seasonal rate, by thinning a homogeneous process.
    peak = float(shape(np.array([0.0]))[0])
    evs = []
    for kind, rate in (("add", lam_add), ("cancel", theta_cancel)):
        for book in (1, -1):
            n = rng.poisson(rate * level * peak * day_len)
            t = np.sort(rng.uniform(0, day_len, n))
            keep = rng.uniform(size=n) < shape(t) / peak
            evs.append(pd.DataFrame({"t": t[keep], "kind": kind, "book": book}))
    evs.append(pd.DataFrame({"t": mkt_t, "kind": "market", "book": -mkt_side}))  # buy (+1) hits ask (-1)
    ev = pd.concat(evs).sort_values("t", kind="stable").to_numpy()

    def draw_q():
        return int(rng.integers(1, 6))                             # new queue after a price change, in lots

    bid_px, qb, qa = 100.0, draw_q(), draw_q()
    q_rows, t_rows = [], []
    for t, kind, book in ev:
        if kind == "add":
            if book == 1: qb += 1
            else: qa += 1
        else:
            if kind == "market":                                    # buy (book=-1) at the ask, sell at the bid
                t_rows.append((t, bid_px + tick if book == -1 else bid_px, lot, 1 if book == -1 else -1))
            if book == 1:
                qb -= 1
            else:
                qa -= 1
            if qa == 0:                                             # ask depleted (trade or cancel): mid up
                bid_px += tick; qb, qa = draw_q(), draw_q()
            elif qb == 0:                                           # bid depleted: mid down
                bid_px -= tick; qb, qa = draw_q(), draw_q()
        q_rows.append((t, bid_px, qb * lot, bid_px + tick, qa * lot))

    def to_ns(t):
        return t_open_ns + (np.floor(np.asarray(t, float) * NS / res_ns) * res_ns).astype(np.int64)

    trades = pd.DataFrame(t_rows, columns=["t", "price", "size", "side"])
    trades.insert(0, "ts", to_ns(trades.pop("t")))
    quotes = pd.DataFrame(q_rows, columns=["t", "bid_px", "bid_sz", "ask_px", "ask_sz"])
    quotes.insert(0, "ts", to_ns(quotes.pop("t")))
    return trades.astype({"side": "int8"}), quotes, Session(t_open_ns, t_open_ns + int(day_len * NS))


def make_days(n_days: int = 3, seed: int = 0, **kw):
    """Several days with different activity levels; t_open_ns spaced one calendar day apart."""
    out = []
    for d in range(n_days):
        lvl = float(np.random.default_rng(seed + 1000 + d).uniform(0.8, 1.25))
        out.append(make_day(seed=seed + d, t_open_ns=d * 86_400 * NS, level=lvl, **kw))
    return out
