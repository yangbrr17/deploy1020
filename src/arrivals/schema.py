"""Input tables, validation, cleaning and event-series extraction.

Two tables are required, one optional. All timestamps are int64 nanoseconds (UTC or exchange local time;
just be consistent). One row per message, already filtered to ONE symbol and ONE trading day.

TRADES (required)                          QUOTES (required, top of book; one row per change)
    ts      int64  ns, exchange time           ts      int64  ns
    price   float                              bid_px  float
    size    float  shares / contracts          bid_sz  float  displayed size at best bid
    side    int8   +1 buyer-initiated,         ask_px  float
                   -1 seller-initiated,        ask_sz  float
                    0 unknown

LEVEL2 (optional, only for the queue-reactive test H6; one row per order-book event at the best levels)
    ts      int64  ns
    book    int8   +1 bid side, -1 ask side
    event   str    'add' | 'cancel' | 'trade'
    size    float
    q_after float  queue size at that level right after the event

Session: (t_open, t_close) in the same ns clock; only continuous trading (no auctions, no halts).

Cleaning rules, in order (each one is a function below so it can be unit-tested and changed):
1. keep rows inside the continuous session;
2. drop crossed / locked quotes (ask <= bid) and non-positive sizes;
3. aggregate trades with the same timestamp and side into one "trade event" (one aggressive order that
   swept several resting orders prints several fills with one timestamp);
4. if the timestamp resolution is coarse (e.g. 1 ms) and ties remain, spread tied events uniformly
   inside the resolution interval (randomised jitter) so continuous-time tests are valid;
5. durations are computed within the day only (no overnight gap); the first duration after the open
   is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NS = 1_000_000_000  # nanoseconds per second

TRADE_COLS = {"ts": "int64", "price": "float64", "size": "float64", "side": "int8"}
QUOTE_COLS = {"ts": "int64", "bid_px": "float64", "bid_sz": "float64", "ask_px": "float64", "ask_sz": "float64"}
L2_COLS = {"ts": "int64", "book": "int8", "event": "object", "size": "float64", "q_after": "float64"}


@dataclass
class Session:
    """Continuous-trading window of one day, in the same ns clock as the tables."""
    t_open: int
    t_close: int

    @property
    def seconds(self) -> float:
        return (self.t_close - self.t_open) / NS


def validate(df: pd.DataFrame, cols: dict, name: str) -> pd.DataFrame:
    """Check the required columns exist, cast dtypes, sort by ts (stable). Raises with a clear message."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; required {list(cols)}")
    out = df[list(cols)].astype(cols).sort_values("ts", kind="stable").reset_index(drop=True)
    if out["ts"].isna().any():
        raise ValueError(f"{name}: null timestamps")
    return out


def clip_session(df: pd.DataFrame, s: Session) -> pd.DataFrame:
    """Rule 1: keep only rows inside the continuous session [t_open, t_close)."""
    return df[(df["ts"] >= s.t_open) & (df["ts"] < s.t_close)].reset_index(drop=True)


def clean_quotes(q: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: drop crossed/locked books and non-positive sizes; drop rows that change nothing."""
    q = q[(q["ask_px"] > q["bid_px"]) & (q["bid_sz"] > 0) & (q["ask_sz"] > 0)]
    vals = q[["bid_px", "bid_sz", "ask_px", "ask_sz"]].to_numpy()
    changed = np.ones(len(q), bool)
    changed[1:] = (np.diff(vals, axis=0) != 0).any(axis=1)
    return q[changed].reset_index(drop=True)


def aggregate_trades(t: pd.DataFrame) -> pd.DataFrame:
    """Rule 3: one row per (ts, side): total size, VWAP price, number of fills (n_fills)."""
    g = t.assign(pv=t["price"] * t["size"]).groupby(["ts", "side"], sort=True)
    out = g.agg(size=("size", "sum"), pv=("pv", "sum"), n_fills=("size", "size")).reset_index()
    out["price"] = out["pv"] / out["size"]
    return out.drop(columns="pv")


def jitter_ties(ts_ns: np.ndarray, resolution_ns: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rule 4: event times in seconds, with tied timestamps spread uniformly inside [ts, ts + resolution).

    With 1 ms stamps a liquid name has many ties; zero durations break every continuous-time test
    (the exponential has no atom at 0). Every event gets jitter so the treatment is uniform.
    Returns (sorted times, order): apply `order` to the marks so they stay aligned with the times.
    resolution_ns=0 (true ns stamps): no jitter; remaining exact ties give zero gaps that
    durations() drops.
    """
    t = ts_ns.astype(np.float64) / NS
    if resolution_ns > 0:
        t = t + rng.uniform(0.0, resolution_ns / NS, size=len(t))
    order = np.argsort(t, kind="stable")
    return t[order], order


def durations(times_s: np.ndarray) -> np.ndarray:
    """Rule 5: positive gaps between consecutive events of one day (the first gap after the open is
    already excluded because we only difference events, not the open itself)."""
    d = np.diff(np.asarray(times_s, float))
    return d[d > 0]


# ---------------------------------------------------------------------------------------------
# Event series. Each returns event times in SECONDS since the open, plus optional marks.
# ---------------------------------------------------------------------------------------------

def trade_events(trades: pd.DataFrame, s: Session, resolution_ns: int, rng) -> dict:
    """E1: aggressive orders (aggregated trades). Marks: side, size."""
    t = aggregate_trades(clip_session(trades, s))
    times, o = jitter_ties(t["ts"].to_numpy() - s.t_open, resolution_ns, rng)
    return {"times": times, "side": t["side"].to_numpy()[o], "size": t["size"].to_numpy()[o]}


def mid_change_events(quotes: pd.DataFrame, s: Session, resolution_ns: int, rng) -> dict:
    """E2: times the mid-price changes. Marks: direction (+1 up / -1 down) and the best-queue sizes
    right AFTER the change (these are the initial conditions of the Cont-de Larrard first passage)."""
    q = clean_quotes(clip_session(quotes, s))
    mid = 0.5 * (q["bid_px"].to_numpy() + q["ask_px"].to_numpy())
    idx = np.flatnonzero(np.diff(mid) != 0) + 1
    times, o = jitter_ties(q["ts"].to_numpy()[idx] - s.t_open, resolution_ns, rng)
    return {
        "times": times,
        "direction": np.sign(np.diff(mid)[idx - 1]).astype(int)[o],
        "bid_sz": q["bid_sz"].to_numpy()[idx][o],
        "ask_sz": q["ask_sz"].to_numpy()[idx][o],
        "spread": (q["ask_px"] - q["bid_px"]).to_numpy()[idx][o],
    }


def l1_update_events(quotes: pd.DataFrame, s: Session, resolution_ns: int, rng) -> dict:
    """E3: any change of the top of book (price or size on either side)."""
    q = clean_quotes(clip_session(quotes, s))
    return {"times": jitter_ties(q["ts"].to_numpy() - s.t_open, resolution_ns, rng)[0]}


def l1_queue_flows(quotes: pd.DataFrame, trades: pd.DataFrame, s: Session) -> dict:
    """Approximate add / remove flows at the best queues from L1 snapshots (used when no L2 table).

    Between two snapshots with the SAME best price on a side, a size increase is counted as added
    volume and a decrease as removed volume. Removed volume is split into "trade" (matched against
    the trades table at that price in that interval) and "cancel" (the rest). This is an
    approximation: two events inside one snapshot interval net out. Use LEVEL2 when you have it.

    Returns per side ('bid', 'ask'): total added, cancelled, traded volume, and the number of adds,
    plus the session length, so rates are volume / seconds.
    """
    q = clean_quotes(clip_session(quotes, s))
    tr = aggregate_trades(clip_session(trades, s))
    out = {"seconds": s.seconds}
    for side, px, sz, tside in (("bid", "bid_px", "bid_sz", -1), ("ask", "ask_px", "ask_sz", +1)):
        p = q[px].to_numpy()
        v = q[sz].to_numpy()
        same = p[1:] == p[:-1]
        dv = np.diff(v)
        added = dv[same & (dv > 0)]
        removed_total = -dv[same & (dv < 0)].sum()
        # Aggressive buys (side=+1) hit the ask queue, aggressive sells (-1) hit the bid queue.
        traded = tr.loc[tr["side"] == tside, "size"].sum()
        traded = min(traded, removed_total)
        out[side] = {
            "added": float(added.sum()),
            "n_adds": int(len(added)),
            "traded": float(traded),
            "cancelled": float(removed_total - traded),
        }
    return out
