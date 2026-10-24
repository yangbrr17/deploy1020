"""Chart specs for Report.chart(). Each function returns a plain dict; theme.js draws it in the browser.

Pick the form by the reader's job (docs/report-theme/README.md, "图表怎么选"):
    trend over an ordered x ............ line        (<= 4 series get direct end labels)
    compare magnitudes across items .... bar         (horizontal automatically when labels are long)
    distribution of one quantity ....... hist        (log_x=True for durations)
    matrix (correlation, confusion) .... heatmap     (diverging around 0, or sequential)
    two quantities per sample .......... scatter     (<= 3 series)
    classifier quality ................. roc, calibration (need numpy)
Rules the renderer enforces: one y-axis only, colours follow the series name across the report, thin
marks, hairline grid, a legend for >= 2 series, hover tooltips and a "表格" table view on every chart.
Inputs may be lists, numpy arrays or pandas Series; NaN becomes a gap. Format strings (x_format,
y_format, value_format) use the renderer's mini-language: "auto", ".2f", ",.0f", "+.1f", ".1%", ".2s"
(compact K/M/B), ".2e", "int".
"""
from __future__ import annotations

import bisect
import math

from .report import _py


def _vals(v) -> list:
    v = _py(v)
    return [x if isinstance(x, (int, float)) and not isinstance(x, bool) else (None if x is None else x) for x in v]


def _series(series, key="values", default_name="值") -> list[dict]:
    """dict {name: values} | list of dicts {name, values, color?, dash?, area?} | bare values."""
    if hasattr(series, "to_dict") and hasattr(series, "columns"):        # DataFrame: one series per column
        return [{"name": str(c), key: _vals(series[c])} for c in series.columns]
    if isinstance(series, dict):
        return [{"name": str(k), key: _vals(v)} for k, v in series.items()]
    series = list(series)
    if series and isinstance(series[0], dict):
        return [{**{k: v for k, v in s.items() if v is not None}, key: _vals(s[key])} for s in series]
    return [{"name": default_name, key: _vals(series)}]


_KEEP_FALSE = {"endLabels", "labels", "horizontal"}      # False means "off" for these; elsewhere False = default


def _spec(kind: str, **kw) -> dict:
    return {"type": kind, **{k: v for k, v in kw.items()
                             if v is not None and (v is not False or k in _KEEP_FALSE)}}


def _refs(refs):
    """[{"y": 0.5, "label": "随机"}, {"x": 3, "label": "切换点"}] or None."""
    return [dict(r) for r in refs] if refs else None


def line(x, series, *, x_label=None, y_label=None, x_name=None, x_format=None, y_format=None,
         x_min=None, x_max=None, y_min=None, y_max=None, zero=False, x_log=False, refs=None,
         diagonal=False, area=False, markers=False, emphasis=None, end_labels=True, labels=None,
         height=None) -> dict:
    """Lines over an ordered x (numbers, or strings for categories such as dates).
    series: {name: values} or [{"name", "values", "key", "color": "s1".."s8"|"gray", "dash": True, "area": True}]
    ("key" = identity for the colour when the display name differs, e.g. "模型 A（AUC 0.81）" -> "模型 A").
    emphasis: name or names drawn in colour, the rest grey (the story is one series).
    refs: horizontal/vertical reference lines; diagonal: True or a label (y = x, e.g. "随机").
    labels: point annotations [{"x", "y", "text"}]. zero: include 0 on the y-axis."""
    return _spec("line", x=_vals(x), series=_series(series), xLabel=x_label, yLabel=y_label, xName=x_name,
                 xFormat=x_format, yFormat=y_format, xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max,
                 zero=zero, xLog=x_log, refs=_refs(refs), diagonal=diagonal, area=area, markers=markers,
                 emphasis=emphasis, endLabels=None if end_labels else False, labels=_py(labels), height=height)


def bar(categories, series, *, horizontal="auto", stacked=False, value_format=None, value_label=None,
        labels=None, highlight=None, sort=None, category_name=None, max=None, height=None) -> dict:
    """Bars per category. series: values (one series) or {name: values} (grouped, or stacked=True).
    horizontal: "auto" switches to horizontal when category labels do not fit under the bars.
    highlight: category or categories in colour, others grey (single series only).
    sort: "desc" / "asc" by the first series. labels: value at the bar tip (default: single series)."""
    cats = [str(c) for c in _py(categories)]
    ser = _series(series)
    if sort:
        order = sorted(range(len(cats)), key=lambda i: (ser[0]["values"][i] is None, ser[0]["values"][i] or 0),
                       reverse=(sort == "desc"))
        if sort == "desc":       # keep missing values last
            order = [i for i in order if ser[0]["values"][i] is not None] + [i for i in order if ser[0]["values"][i] is None]
        cats = [cats[i] for i in order]
        for s in ser:
            s["values"] = [s["values"][i] for i in order]
    return _spec("bar", categories=cats, series=ser, horizontal=None if horizontal == "auto" else bool(horizontal),
                 stacked=stacked, valueFormat=value_format, yLabel=value_label, labels=labels,
                 highlight=highlight, categoryName=category_name, max=max, height=height)


def _edges(values, bins, rng, log_x):
    vals = [v for v in values if v is not None and math.isfinite(v) and (v > 0 or not log_x)]
    if not vals:
        raise ValueError("hist: no finite values" + (" > 0 for log_x" if log_x else ""))
    lo, hi = rng if rng else (min(vals), max(vals))
    if hi <= lo:
        hi = lo + 1
    if log_x:
        a, b = math.log10(lo), math.log10(hi)
        e = [10 ** (a + (b - a) * i / bins) for i in range(bins + 1)]
    else:
        e = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    e[0], e[-1] = lo, hi            # exact ends, so the extreme values are counted
    return e


def _counts(values, edges, density):
    c = [0] * (len(edges) - 1)
    n = 0
    for v in values:
        if v is None or not math.isfinite(v) or v < edges[0] or v > edges[-1]:
            continue
        i = min(bisect.bisect_right(edges, v) - 1, len(c) - 1)
        c[i] += 1
        n += 1
    if density:
        return [ci / (n * (edges[i + 1] - edges[i])) if n else 0 for i, ci in enumerate(c)]
    return c


def hist(values=None, *, series=None, bins=30, range=None, log_x=False, density=False, edges=None,
         counts=None, x_label=None, y_label=None, x_format=None, y_format=None, refs=None, height=None) -> dict:
    """Histogram. Either raw `values` (one series) / `series` {name: values} (<= 3, drawn as steps),
    or precomputed `edges` (n+1) and `counts` (n, or {name: counts}). log_x: geometric bins, log axis."""
    if edges is None:
        data = {k: _vals(v) for k, v in series.items()} if series else {"样本": _vals(values)}
        pooled = [v for vs in data.values() for v in vs]
        edges = _edges(pooled, bins, range, log_x)
        ser = [{"name": k, "counts": _counts(v, edges, density)} for k, v in data.items()]
    else:
        edges = _vals(edges)
        ser = _series(counts, key="counts", default_name="样本")
    return _spec("hist", edges=edges, series=ser, xLog=log_x, xLabel=x_label,
                 yLabel=y_label or ("密度" if density else "样本数"), xFormat=x_format, yFormat=y_format,
                 refs=_refs(refs), height=height)


def heatmap(values, rows=None, cols=None, *, scale="diverging", domain=None, value_format=".2f",
            value_name=None, labels=True, scale_label=None) -> dict:
    """Matrix of cells. values: 2-D list/array or a DataFrame (rows = index, cols = columns).
    scale: "diverging" (two hues around a grey midpoint; use for correlations, domain=(-1, 1)) or
    "sequential" (one hue, light -> dark). labels: values inside cells when they fit."""
    if hasattr(values, "columns") and hasattr(values, "index"):
        rows = [str(r) for r in values.index] if rows is None else rows
        cols = [str(c) for c in values.columns] if cols is None else cols
        values = values.values
    m = _py(values)
    rows = [str(r) for r in (rows if rows is not None else range(1, len(m) + 1))]
    cols = [str(c) for c in (cols if cols is not None else range(1, len(m[0]) + 1))]
    return _spec("heatmap", rows=rows, cols=cols, values=m, scale=scale, domain=_py(domain),
                 valueFormat=value_format, valueName=value_name, labels=None if labels else False,
                 scaleLabel=scale_label)


def scatter(series, *, x_label=None, y_label=None, x_format=None, y_format=None, x_min=None, x_max=None,
            y_min=None, y_max=None, diagonal=False, refs=None, max_points=2000, seed=0, height=None) -> dict:
    """Points. series: {name: (x, y)} or [{"name", "x", "y"}], at most 3 series (beyond that colours stop
    being distinguishable for every reader). Large inputs are thinned to max_points per series."""
    import random
    if isinstance(series, dict):
        ser = [{"name": str(k), "x": _vals(v[0]), "y": _vals(v[1])} for k, v in series.items()]
    else:
        ser = [{**s, "x": _vals(s["x"]), "y": _vals(s["y"])} for s in series]
    if len(ser) > 3:
        raise ValueError("scatter: at most 3 series; fold the rest into one or use small multiples")
    rnd = random.Random(seed)
    for s in ser:
        n = len(s["x"])
        if n > max_points:
            keep = sorted(rnd.sample(range(n), max_points))
            s["x"] = [s["x"][i] for i in keep]
            s["y"] = [s["y"][i] for i in keep]
    return _spec("scatter", series=ser, xLabel=x_label, yLabel=y_label, xFormat=x_format, yFormat=y_format,
                 xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max, diagonal=diagonal, refs=_refs(refs), height=height)


# ----------------------------------------------------------------------------- classifier charts (numpy)
def roc_curve(y_true, score, points: int = 101):
    """(fpr grid, tpr on that grid, auc). Ties in score are handled by grouping. Needs numpy."""
    import numpy as np
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    tpr = np.r_[0, tp / max(1, y.sum())]
    fpr = np.r_[0, fp / max(1, (~y).sum())]
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    grid = np.linspace(0, 1, points)
    return grid, np.interp(grid, fpr, tpr), auc


def roc(y_true, scores: dict, *, points: int = 101, show_auc: bool = True, height=None) -> dict:
    """ROC curves for {name: score}. Legend names get the AUC appended when show_auc; the colour still
    follows the bare name (series "key"), so the model keeps its colour in the other charts."""
    series = []
    grid = None
    for name, sc in scores.items():
        grid, tpr, auc = roc_curve(y_true, sc, points)
        series.append({"name": f"{name}（AUC {auc:.3f}）" if show_auc else name, "key": name,
                       "values": [round(float(t), 4) for t in tpr]})
    return line([round(float(g), 4) for g in grid], series, x_label="假阳性率", y_label="真阳性率",
                x_name="假阳性率", x_format=".2f", y_format=".2f", x_min=0, x_max=1, y_min=0, y_max=1,
                diagonal="随机", end_labels=False, height=height or 300)


def calibration(y_true, probs: dict, *, bins: int = 10, height=None) -> dict:
    """Reliability curve: mean predicted probability vs observed frequency per quantile bin."""
    import numpy as np
    y = np.asarray(y_true, dtype=float)
    out, xs = {}, None
    for name, p in probs.items():
        p = np.asarray(p, dtype=float)
        q = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
        mp = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
        fr = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
        out[name] = (mp, fr)
    if len(out) == 1:
        (name, (mp, fr)), = out.items()
        return line([round(float(v), 4) for v in mp], {name: [round(float(v), 4) for v in fr]},
                    x_label="平均预测概率", y_label="实际频率", x_name="预测", x_format=".2f", y_format=".2f",
                    diagonal="完美校准", markers=True, height=height or 300)
    return scatter({k: (v[0], v[1]) for k, v in out.items()}, x_label="平均预测概率", y_label="实际频率",
                   x_format=".2f", y_format=".2f", diagonal="完美校准", height=height or 300)
