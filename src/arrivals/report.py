"""Run hypotheses H1-H9 on every symbol-day and write one tidy results table.

    PYTHONPATH=src python -m arrivals.report --synthetic --days 3 --out results.csv
    PYTHONPATH=src python -m arrivals.report --trades trades.csv --quotes quotes.csv \
        --tz America/New_York --open 09:30 --close 16:00 --resolution-ns 1000 --out results.csv

Input files may hold many symbols and days; they need the schema.py columns plus `symbol`.
`ts` is int64 ns since the Unix epoch (UTC); the session is built from --open/--close in --tz.

Output columns (one row per test):
    symbol, date, hypothesis, series, test, stat, se, p_value, effect, effect_name, n, verdict, note
`verdict` is "reject" only when p < alpha AND the effect size passes the practical threshold in
Config; "reject (small effect)" flags statistically significant but immaterial deviations, which
is what you get with 10^5 observations. A BH-adjusted column `reject_fdr` is added at the end.

H1  E1/E2/E3 inter-event times are iid exponential (homogeneous Poisson)
H2  same after removing the intraday rate (non-homogeneous Poisson, NHPP)
H3  marginal law of seasonally adjusted durations (descriptive: exp / Weibull / gamma / GG / ...)
H4  Weibull-ACD(1,1) captures the dependence: residuals iid Exp(1); EACD vs WACD
H5  exponential Hawkes beats Poisson and its compensator increments are iid Exp(1)
H6  best-queue add/removal rates do not depend on queue size (Cont-de Larrard) vs queue-reactive
H7  time to the next mid change follows the Cont-de Larrard first-passage law (PIT test, tail index)
H8  direction of the next mid move is calibrated by phi(q_bid, q_ask)
H9  parameters are stable across days (Cochran Q, I^2)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import distributions, models, queue, schema
from . import stats as st


@dataclass
class Config:
    resolution_ns: int = 0          # feed timestamp resolution; 0 = true ns stamps (exact ties dropped)
    n_bins: int = 26                # seasonal bins per session (6.5 h -> 15 min)
    lags: int = 20                  # Ljung-Box lags
    windows: tuple = (1.0, 10.0, 60.0)   # count windows (s) for the index of dispersion / Fano factor
    n_boot: int = 199               # bootstrap draws for the KS test
    alpha: float = 0.01             # per-test level before the FDR step
    fdr_q: float = 0.05             # Benjamini-Hochberg level over the whole table
    # practical thresholds: a rejection only counts if the effect is at least this large
    cv_tol: float = 0.10            # |CV - 1|
    ks_tol: float = 0.02            # KS distance D
    rho_tol: float = 0.05           # |first autocorrelation|
    fano_tol: float = 0.20          # Fano - 1
    rate_ratio_tol: float = 1.5     # H6 max/min rate across queue-size bins
    slope_tol: float = 0.20         # H8 |calibration slope - 1|
    fit_hawkes: bool = True
    max_hawkes_events: int = 100_000   # cap per day (truncate the day) to bound fit time
    q_cap: int = 30                 # H8 queue cap for phi
    seed: int = 0


def _verdict(p, big, alpha):
    if p is None or not np.isfinite(p):
        return "info"
    if p < alpha:
        return "reject" if big else "reject (small effect)"
    return "not rejected"


def _row(hyp, series, r, verdict="info", **extra):
    keys = ("test", "stat", "se", "p_value", "effect", "effect_name", "n", "note")
    out = {"hypothesis": hyp, "series": series, **{k: r.get(k) for k in keys}, "verdict": verdict}
    out.update(extra)
    return out


def exp_battery(e, hyp, series, cfg: Config, rng) -> list[dict]:
    """Four checks that e is iid exponential: dispersion, KS (bootstrap), Anderson-Darling, Ljung-Box."""
    rows = []
    r = st.dispersion_test(e); r["test"] = "Engle-Russell dispersion"
    rows.append(_row(hyp, series, r, _verdict(r["p_value"], abs(r["effect"] - 1) > cfg.cv_tol, cfg.alpha)))
    r = st.ks_exponential(e, cfg.n_boot, rng); r["test"] = "KS vs exponential (bootstrap)"
    rows.append(_row(hyp, series, r, _verdict(r["p_value"], r["effect"] > cfg.ks_tol, cfg.alpha)))
    r = st.anderson_exponential(e); r["test"] = "Anderson-Darling vs exponential"
    p_ad = 0.0 if r["p_censored"] else r["p_value"]           # censored at the 1% table floor
    rows.append(_row(hyp, series, r, _verdict(p_ad, True, cfg.alpha)))
    r = st.ljung_box(e, cfg.lags); r["test"] = f"Ljung-Box({cfg.lags})"
    rows.append(_row(hyp, series, r, _verdict(r["p_value"], abs(r["effect"]) > cfg.rho_tol, cfg.alpha)))
    return rows


def fano_rows(times, t_end, hyp, series, cfg) -> list[dict]:
    rows = []
    for w in cfg.windows:
        r = st.count_dispersion(times, t_end, w); r["test"] = f"index of dispersion, window {w:g}s"
        rows.append(_row(hyp, series, r, _verdict(r["p_value"], r["effect"] - 1 > cfg.fano_tol, cfg.alpha)))
    return rows


def run_day(trades, quotes, s: schema.Session, cfg: Config, other_days_trade_times=None) -> list[dict]:
    """All hypotheses on one symbol-day. other_days_trade_times: list of E1 time arrays from OTHER
    days, used for the out-of-sample seasonal curve (H2); falls back to in-sample if None/empty."""
    rng = np.random.default_rng(cfg.seed)
    T = s.seconds
    rows: list[dict] = []
    e1 = schema.trade_events(trades, s, cfg.resolution_ns, rng)
    e2 = schema.mid_change_events(quotes, s, cfg.resolution_ns, rng)
    e3 = schema.l1_update_events(quotes, s, cfg.resolution_ns, rng)

    # H1: homogeneous Poisson for each series
    for name, ev in (("E1 trades", e1), ("E2 mid changes", e2), ("E3 L1 updates", e3)):
        if len(ev["times"]) < 200:
            continue
        rows += exp_battery(schema.durations(ev["times"]), "H1", name, cfg, rng)
        rows += fano_rows(ev["times"], T, "H1", name, cfg)

    # H2: NHPP with an intraday rate curve (out-of-sample when other days are given)
    t1 = e1["times"]
    days = other_days_trade_times if other_days_trade_times else [t1]
    edges, rates = models.diurnal_rate(days, T, cfg.n_bins)
    oos = "out-of-sample curve" if other_days_trade_times else "IN-SAMPLE curve (only one day)"
    rows += exp_battery(models.nhpp_residuals(t1, edges, rates), "H2", "E1 trades", cfg, rng)
    rows[-1]["note"] = oos
    u1 = models.business_time(t1, edges, rates)
    rows += fano_rows(u1, T, "H2", "E1 trades (business time)", cfg)

    # H3 + H4: seasonally adjusted durations, marginal fits and Weibull-ACD
    x = np.diff(t1)
    keep = x > 0
    x_adj, _ = models.diurnal_adjust(x[keep], t1[:-1][keep], T, cfg.n_bins)
    x_adj = x_adj / x_adj.mean()
    for r in distributions.compare(distributions.fit_all(x_adj, rng=rng)):
        rows.append(_row("H3", "E1 adjusted durations (iid fit, descriptive)", r))
    eacd = models.fit_wacd(x_adj, fix_k=1.0)
    wacd = models.fit_wacd(x_adj)
    r = st.lr_test(eacd["ll"], wacd["ll"], 1); r["test"] = "LR EACD vs WACD (k = 1?)"
    rows.append(_row("H4", "E1 adjusted durations", r, _verdict(r["p_value"], True, cfg.alpha)))
    names = ["omega", "a", "b", "k"]
    for i, nm in enumerate(names):
        rows.append(_row("H4", "E1 adjusted durations", {"test": f"WACD {nm}", "stat": wacd["params"][i],
                                                         "se": wacd["se"][i], "n": len(x_adj)}))
    se_pers = float(np.hypot(wacd["se"][1], wacd["se"][2]))   # ignores cov(a, b): rough
    rows.append(_row("H4", "E1 adjusted durations", {"test": "WACD persistence a+b",
                                                     "stat": wacd["persistence"], "se": se_pers,
                                                     "n": len(x_adj), "note": "se ignores cov(a,b)"}))
    rows += exp_battery(models.wacd_residuals(wacd["params"], x_adj), "H4", "WACD residuals", cfg, rng)
    for r in distributions.compare(distributions.fit_all(models.acd_standardised(wacd["params"], x_adj), rng=rng)):
        rows.append(_row("H4", "ACD standardised eps (choose error law)", r))

    # H5: exponential Hawkes in business time vs Poisson
    if cfg.fit_hawkes and len(u1) > 200:
        u = u1[: cfg.max_hawkes_events]
        Tu = float(u[-1]) if len(u1) > cfg.max_hawkes_events else T
        hk = models.fit_hawkes_exp(u, Tu)
        ll0 = models.poisson_loglik(u, Tu)
        r = st.lr_test(ll0, hk["ll"], 2, boundary=True); r["test"] = "LR Poisson vs Hawkes (boundary)"
        r["note"] = f"AIC Poisson {st.aic(ll0, 1):.0f}, Hawkes {st.aic(hk['ll'], 3):.0f}"
        rows.append(_row("H5", "E1 trades (business time)", r, _verdict(r["p_value"], True, cfg.alpha)))
        se_n = float(np.hypot(hk["se"][1] / hk["params"][2],
                              hk["params"][1] * hk["se"][2] / hk["params"][2] ** 2))
        rows.append(_row("H5", "E1 trades (business time)", {"test": "Hawkes branching ratio alpha/beta",
                         "stat": hk["branching"], "se": se_n, "n": len(u), "note": "se ignores cov"}))
        rows.append(_row("H5", "E1 trades (business time)", {"test": "Hawkes half-life ln2/beta (s)",
                         "stat": hk["half_life"], "n": len(u)}))
        rows += exp_battery(models.hawkes_residuals(hk["params"], u), "H5", "Hawkes compensator", cfg, rng)

    # H6: queue-size dependence of best-queue flows (from L1)
    for r in queue.queue_reactive_lr(quotes, s, to_business=lambda t: models.business_time(t, edges, rates)):
        rows.append(_row("H6", "best queues (L1, both sides)", r,
                         _verdict(r["p_value"], r["effect"] > cfg.rate_ratio_tol, cfg.alpha)))

    # H7 + H8: next mid change, Cont-de Larrard
    if len(e2["times"]) > 100:
        rates = queue.estimate_rates(quotes, trades, s)
        for side in ("bid", "ask"):
            d = rates[side]
            rows.append(_row("H7", f"{side} queue rates", {"test": "removal / add rate (balanced = 1)",
                             "stat": d["c"] / d["lam"], "n": None,
                             "note": f"lam={d['lam']:.3g}/s c={d['c']:.3g}/s unit={d['unit']:.4g}"}))
        fp = queue.first_passage_sample(e2, T, rates["bid"]["unit"], rates["ask"]["unit"])
        r = queue.uniform_ks(queue.pit_first_passage(fp, rates)); r["test"] = "PIT KS vs Cont-de Larrard survival"
        rows.append(_row("H7", "E2 time to next mid change", r, _verdict(r["p_value"], r["effect"] > 0.05, cfg.alpha)))
        h = st.hill_estimator(fp["tau"]); h["test"] = "Hill tail index (balanced theory: 1)"
        z = (h["stat"] - 1.0) / h["se"]
        h["p_value"] = float(2 * st.stats.norm.sf(abs(z)))
        rows.append(_row("H7", "E2 time to next mid change", h, _verdict(h["p_value"], abs(h["stat"] - 1) > 0.3, cfg.alpha)))
        phi = queue.phi_for_sample(fp, cfg.q_cap)
        c = st.calibration_test(phi, fp["up"]); c["test"] = "calibration of phi(q_bid, q_ask)"
        bss = 1 - c["brier"] / np.mean((0.5 - fp["up"]) ** 2)
        c["note"] = f"intercept={c['intercept']:.3f} brier={c['brier']:.4f} skill vs 0.5={bss:.3f}"
        rows.append(_row("H8", "E2 next move direction", c,
                         _verdict(c["p_value"], abs(c["slope"] - 1) > cfg.slope_tol, cfg.alpha)))
    return rows


H9_PARAMS = [("H5", "Hawkes branching ratio alpha/beta"), ("H4", "WACD persistence a+b"), ("H4", "WACD k")]


def heterogeneity_rows(df: pd.DataFrame) -> list[dict]:
    """H9: for each symbol, are the daily estimates of the key parameters equal (Cochran Q)?"""
    rows = []
    for sym, g in df.groupby("symbol"):
        for hyp, test in H9_PARAMS:
            sub = g[(g["hypothesis"] == hyp) & (g["test"] == test)].dropna(subset=["stat", "se"])
            sub = sub[sub["se"] > 0]
            if len(sub) < 2:
                continue
            r = st.heterogeneity(sub["stat"], sub["se"])
            r["test"] = f"Cochran Q across days: {test}"
            r["note"] = f"pooled={r['pooled']:.4g}, days={len(sub)}"
            rows.append({"symbol": sym, "date": "all", **_row("H9", "daily estimates", r,
                         _verdict(r["p_value"], r["effect"] > 0.5, 0.01))})
    return rows


def finalize(rows: list[dict], cfg: Config) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = pd.concat([df, pd.DataFrame(heterogeneity_rows(df))], ignore_index=True)
    p = df["p_value"].astype(float)
    ok = p.notna().to_numpy()
    rej = np.zeros(len(df), bool)
    rej[ok] = st.benjamini_hochberg(p[ok].to_numpy(), cfg.fdr_q)
    df["reject_fdr"] = rej
    cols = ["symbol", "date", "hypothesis", "series", "test", "stat", "se", "p_value", "effect",
            "effect_name", "n", "verdict", "reject_fdr", "note"]
    return df[cols]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Share of symbol-days with verdict == 'reject' and the median effect, per hypothesis and test."""
    d = df[df["verdict"] != "info"].copy()
    d["is_reject"] = d["verdict"] == "reject"
    d["effect"] = pd.to_numeric(d["effect"], errors="coerce")
    return (d.groupby(["hypothesis", "series", "test"])
             .agg(days=("is_reject", "size"), reject_share=("is_reject", "mean"), median_effect=("effect", "median"))
             .reset_index())


def run_frames(frames, cfg: Config) -> pd.DataFrame:
    """frames: list of (symbol, date, trades, quotes, session). Leave-one-day-out seasonal curves."""
    rng = np.random.default_rng(cfg.seed)
    e1 = {(sym, d): schema.trade_events(tr, s, cfg.resolution_ns, rng)["times"] for sym, d, tr, _, s in frames}
    rows = []
    for sym, d, tr, qu, s in frames:
        others = [t for (sy, dd), t in e1.items() if sy == sym and dd != d]
        for r in run_day(tr, qu, s, cfg, others):
            rows.append({"symbol": sym, "date": d, **r})
        print(f"done {sym} {d}: {len(rows)} rows so far", flush=True)
    return finalize(rows, cfg)


def _read(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def frames_from_files(trades_path, quotes_path, tz, t_open, t_close):
    """Split long files into (symbol, date) frames and build each day's session in local time."""
    tr, qu = _read(trades_path), _read(quotes_path)
    for df in (tr, qu):
        df["date"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.tz_convert(tz).dt.date.astype(str)
    frames = []
    for (sym, d), g in tr.groupby(["symbol", "date"]):
        q = qu[(qu["symbol"] == sym) & (qu["date"] == d)]
        s = schema.Session(pd.Timestamp(f"{d} {t_open}", tz=tz).value, pd.Timestamp(f"{d} {t_close}", tz=tz).value)
        frames.append((sym, d, schema.validate(g, schema.TRADE_COLS, "trades"),
                       schema.validate(q, schema.QUOTE_COLS, "quotes"), s))
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="dry run on synth.make_days")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--day-len", type=float, default=23400.0, help="synthetic session length (s)")
    ap.add_argument("--trades"); ap.add_argument("--quotes")
    ap.add_argument("--tz", default="UTC"); ap.add_argument("--open", default="09:30")
    ap.add_argument("--close", default="16:00")
    ap.add_argument("--resolution-ns", type=int, default=0)
    ap.add_argument("--no-hawkes", action="store_true")
    ap.add_argument("--out", default="arrival_tests.csv")
    a = ap.parse_args(argv)
    cfg = Config(resolution_ns=a.resolution_ns, fit_hawkes=not a.no_hawkes)
    if a.synthetic:
        from . import synth
        cfg.resolution_ns = cfg.resolution_ns or 1_000
        frames = [("SYN", f"day{d}", tr, qu, s)
                  for d, (tr, qu, s) in enumerate(synth.make_days(a.days, day_len=a.day_len))]
    else:
        frames = frames_from_files(a.trades, a.quotes, a.tz, a.open, a.close)
    df = run_frames(frames, cfg)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 60, "display.max_rows", 200)
    print(summary(df).to_string(index=False))
    print(f"\nwrote {len(df)} rows to {a.out}; config: {asdict(cfg)}")
    return df


if __name__ == "__main__":
    main()
