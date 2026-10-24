"""Chart specs for Report.chart(). Each function returns a plain dict; theme.js draws it in the browser.

Pick the form by the reader's job (docs/report-theme/README.md, "图表怎么选"):
    a value over time / an ordered x ...... line         last values in tags at the right edge
    compare items, rank them .............. bar          columns; horizontal when labels are long
    signed change per period .............. bar(tone="sign")   green up, red down
    from A to B in steps (attribution) .... waterfall    increments, decrements, totals
    two values per item (before / after) .. dumbbell
    one quantity's distribution ........... hist         log_x=True for durations
    distribution per group ................ box          quartiles + the raw points
    matrix (correlation, time x bucket) ... heatmap      diverging around 0, or sequential
    two quantities per item ............... scatter      fit=True adds OLS line and R²
    the same measure for many items ....... multiples    small multiples on one y-scale
    classifier quality .................... roc, calibration (need numpy)
House style the renderer enforces: value axis on the right, one y-axis only, solid hairline grid, ink zero
line, legend in the panel head, colours follow the series name across the report, tooltips and a "数据"
table view on every chart. Inputs may be lists, numpy arrays or pandas objects; NaN becomes a gap.
Format strings (x_format, y_format, value_format) use the renderer's mini-language: "auto", ".2f", ",.0f",
"+.1f", ".1%", ".2s" (compact K/M/B), ".2e", "int".
"""
from __future__ import annotations

import bisect
import math
import random

from .report import _py

_KEEP_FALSE = {"endLabels", "labels", "horizontal", "tags", "shareY"}   # False means "off" for these


def _spec(kind: str, **kw) -> dict:
    return {"type": kind, **{k: v for k, v in kw.items() if v is not None and (v is not False or k in _KEEP_FALSE)}}


def _vals(v) -> list:
    return [x if isinstance(x, (int, float, str)) and not isinstance(x, bool) else None for x in _py(v)]


def _series(series, key="values", default_name="值") -> list[dict]:
    """dict {name: values} | list of dicts {name, values, key?, color?, dash?, area?} | bare values | DataFrame."""
    if hasattr(series, "to_dict") and hasattr(series, "columns"):
        return [{"name": str(c), key: _vals(series[c])} for c in series.columns]
    if isinstance(series, dict):
        return [{"name": str(k), key: _vals(v)} for k, v in series.items()]
    series = list(series)
    if series and isinstance(series[0], dict):
        return [{**{k: v for k, v in s.items() if v is not None}, key: _vals(s[key])} for s in series]
    return [{"name": default_name, key: _vals(series)}]


def _refs(refs):
    """[{"y": 0.5, "label": "随机"}, {"x": 3, "label": "切换点", "legend": True}] or None."""
    return [dict(r) for r in refs] if refs else None


def _round(v, nd=6):
    return [None if x is None else round(float(x), nd) for x in _py(v)]


# ----------------------------------------------------------------------------- lines and bars
def line(x, series, *, x_label=None, x_name=None, x_format=None, y_format=None, x_min=None, x_max=None,
         y_min=None, y_max=None, zero=False, x_log=False, refs=None, diagonal=False, area=False, markers=False,
         emphasis=None, tags=True, labels=None, height=None) -> dict:
    """Lines over an ordered x (numbers, or strings such as dates).
    series: {name: values} or [{"name", "values", "key", "color": "s1".."s8"|"gray"|"ink", "dash", "area"}].
    tags: last value of each series in a coloured box at the right edge (up to 6 series).
    emphasis: a name or names drawn in colour, the rest grey. refs: [{"y": .., "label": ..}] / [{"x": ..}];
    diagonal: True or a legend label (y = x). labels: point notes [{"x", "y", "text", "dx", "dy", "anchor"}].
    The unit belongs in the panel subtitle, not on the axis."""
    return _spec("line", x=_vals(x), series=_series(series), xLabel=x_label, xName=x_name, xFormat=x_format,
                 yFormat=y_format, xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max, zero=zero, xLog=x_log,
                 refs=_refs(refs), diagonal=diagonal, area=area, markers=markers, emphasis=emphasis,
                 tags=None if tags else False, labels=_py(labels), height=height)


def bar(categories, series, *, horizontal=None, stacked=False, tone=None, value_format=None, labels=None,
        highlight=None, sort=None, category_name=None, axis=False, min=None, max=None, refs=None,
        height=None) -> dict:
    """Bars per category. series: values (one series) or {name: values} (grouped, or stacked=True).
    horizontal: None = automatic (horizontal when labels do not fit under columns), True / False to force.
    tone="sign": one series coloured green when >= 0 and red when < 0 (changes, P&L-like values).
    highlight: category or categories in colour, others grey. sort: "desc" / "asc" by the first series.
    labels: value at the bar tip (default on for one series). axis: show the value axis on horizontal bars."""
    cats = [str(c) for c in _py(categories)]
    ser = _series(series)
    if sort:
        vals = ser[0]["values"]
        order = sorted(range(len(cats)), key=lambda i: (vals[i] is None, (-1 if sort == "desc" else 1) * (vals[i] or 0)))
        cats = [cats[i] for i in order]
        for s in ser:
            s["values"] = [s["values"][i] for i in order]
    return _spec("bar", categories=cats, series=ser, horizontal=horizontal, stacked=stacked, tone=tone,
                 valueFormat=value_format, labels=labels, highlight=highlight, categoryName=category_name,
                 axis=axis, min=min, max=max, refs=_refs(refs), height=height)


def waterfall(steps, *, value_format=None, better="up", y_min=None, y_max=None, height=None) -> dict:
    """Bridge from a starting total to an ending total. steps: [(label, value, "total"|"delta"), ...] or dicts
    {label, value, kind, color}. Totals are drawn from the axis floor (y_min; when > 0 the bars get a break
    mark), deltas float from the running total, green when they move in the `better` direction."""
    out = []
    for s in steps:
        d = dict(s) if isinstance(s, dict) else {"label": s[0], "value": s[1], "kind": s[2] if len(s) > 2 else "delta"}
        d["value"] = float(d["value"])
        out.append(d)
    return _spec("waterfall", steps=out, valueFormat=value_format, better=better, yMin=y_min, yMax=y_max, height=height)


def dumbbell(categories, a: dict, b: dict, *, value_format=None, sort=None, min=None, max=None, height=None) -> dict:
    """Two values per category joined by a bar: a = before / reference (grey), b = after / focus (colour).
    a, b: {"name": ..., "values": [...]} (or a one-item dict {name: values}). sort: "desc"/"asc" by b."""
    def one(d):
        if "values" in d:
            return {"name": str(d.get("name", "")), "values": _vals(d["values"]), **({"color": d["color"]} if d.get("color") else {})}
        (k, v), = d.items()
        return {"name": str(k), "values": _vals(v)}
    cats, A, B = [str(c) for c in _py(categories)], one(a), one(b)
    if sort:
        order = sorted(range(len(cats)), key=lambda i: (B["values"][i] is None, (-1 if sort == "desc" else 1) * (B["values"][i] or 0)))
        cats = [cats[i] for i in order]
        A["values"], B["values"] = [A["values"][i] for i in order], [B["values"][i] for i in order]
    return _spec("dumbbell", categories=cats, a=A, b=B, valueFormat=value_format, min=min, max=max, height=height)


# ----------------------------------------------------------------------------- distributions
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
        c[min(bisect.bisect_right(edges, v) - 1, len(c) - 1)] += 1
        n += 1
    if density:
        return [ci / (n * (edges[i + 1] - edges[i])) if n else 0 for i, ci in enumerate(c)]
    return c


def hist(values=None, *, series=None, bins=30, range=None, log_x=False, density=False, edges=None, counts=None,
         x_label=None, x_format=None, y_format=None, refs=None, height=None) -> dict:
    """Histogram. Raw `values` (one series, filled columns) or `series` {name: values} (<= 3, drawn as steps),
    or precomputed `edges` (n+1) and `counts`. log_x: geometric bins on a log axis. refs: [{"x", "label"}]."""
    if edges is None:
        data = {k: _vals(v) for k, v in series.items()} if series else {"样本": _vals(values)}
        edges = _edges([v for vs in data.values() for v in vs], bins, range, log_x)
        ser = [{"name": k, "counts": _counts(v, edges, density)} for k, v in data.items()]
    else:
        edges = _vals(edges)
        ser = _series(counts, key="counts", default_name="样本")
    return _spec("hist", edges=_round(edges), series=ser, xLog=log_x, xLabel=x_label, xFormat=x_format,
                 yFormat=y_format, refs=_refs(refs), height=height)


def _quantile(s, q):
    i = (len(s) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def box(groups: dict, *, value_format=None, whisker="minmax", points=True, y_min=None, y_max=None, height=None) -> dict:
    """Box plot per group from raw values: {group: values}. Box = quartiles, thick line = median,
    whiskers = min..max (or "1.5iqr"), points = the raw values beside the box (keep under ~200 per group)."""
    out = []
    for name, vals in groups.items():
        v = sorted(x for x in _vals(vals) if isinstance(x, (int, float)))
        if not v:
            continue
        q1, med, q3 = _quantile(v, .25), _quantile(v, .5), _quantile(v, .75)
        lo, hi = v[0], v[-1]
        if whisker == "1.5iqr":
            lo = min(x for x in v if x >= q1 - 1.5 * (q3 - q1))
            hi = max(x for x in v if x <= q3 + 1.5 * (q3 - q1))
        out.append({"name": str(name), "n": len(v), "min": lo, "q1": q1, "median": med, "q3": q3, "max": hi,
                    **({"points": v} if points else {})})
    return _spec("box", groups=out, valueFormat=value_format, yMin=y_min, yMax=y_max, height=height)


# ----------------------------------------------------------------------------- matrices, relations, multiples
def heatmap(values, rows=None, cols=None, *, scale="diverging", domain=None, value_format=".2f", value_name=None,
            labels=True, scale_label=None, cell_height=None) -> dict:
    """Matrix of cells. values: 2-D list/array or a DataFrame (rows = index, cols = columns).
    scale: "diverging" (blue - grey - pink around the middle of domain; correlations: domain=(-1, 1)) or
    "sequential" (one hue, light -> dark). More than 12 columns: labels go below, every k-th shown."""
    if hasattr(values, "columns") and hasattr(values, "index"):
        rows = [str(r) for r in values.index] if rows is None else rows
        cols = [str(c) for c in values.columns] if cols is None else cols
        values = values.values
    m = _py(values)
    rows = [str(r) for r in (rows if rows is not None else range(1, len(m) + 1))]
    cols = [str(c) for c in (cols if cols is not None else range(1, len(m[0]) + 1))]
    return _spec("heatmap", rows=rows, cols=cols, values=m, scale=scale, domain=_py(domain), valueFormat=value_format,
                 valueName=value_name, labels=None if labels else False, scaleLabel=scale_label, cellHeight=cell_height)


def scatter(series, *, x_label=None, y_label=None, x_format=None, y_format=None, x_min=None, x_max=None,
            y_min=None, y_max=None, diagonal=False, fit=False, labels=None, max_points=2000, seed=0,
            height=None) -> dict:
    """Points. series: {name: (x, y)} or {name: (x, y, point_labels)} or [{"name", "x", "y", "labels"}];
    at most 3 series (beyond that colours stop being distinguishable for every reader).
    fit: OLS line with R² and slope. diagonal: y = x (True or a legend label). Large inputs are thinned."""
    if isinstance(series, dict):
        ser = []
        for k, v in series.items():
            d = {"name": str(k), "x": _vals(v[0]), "y": _vals(v[1])}
            if len(v) > 2:
                d["labels"] = [str(t) for t in v[2]]
            ser.append(d)
    else:
        ser = [{**s, "x": _vals(s["x"]), "y": _vals(s["y"])} for s in series]
    if len(ser) > 3:
        raise ValueError("scatter: at most 3 series; fold the rest into one or use small multiples")
    rnd = random.Random(seed)
    for s in ser:
        n = len(s["x"])
        if n > max_points:
            keep = sorted(rnd.sample(range(n), max_points))
            for k in ("x", "y", "labels"):
                if k in s:
                    s[k] = [s[k][i] for i in keep]
    return _spec("scatter", series=ser, xLabel=x_label, yLabel=y_label, xFormat=x_format, yFormat=y_format,
                 xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max, diagonal=diagonal, fit=fit, labels=_py(labels),
                 height=height)


def multiples(x, panels, *, cols=None, y_format=None, share_y=True, y_min=None, y_max=None, refs=None,
              color=None, height=None) -> dict:
    """Small multiples: one mini line per item on a shared y-scale (share_y=False for own scales).
    panels: {name: values} or [{"name", "values", "color"}]; refs: [{"y": ..}] drawn in every panel."""
    return _spec("multiples", x=_vals(x), panels=_series(panels), cols=cols, yFormat=y_format,
                 shareY=None if share_y else False, yMin=y_min, yMax=y_max, refs=_refs(refs), color=color, height=height)


# ----------------------------------------------------------------------------- classifier charts (numpy)
def roc_curve(y_true, score, points: int = 101):
    """(fpr grid, tpr on that grid, auc). Ties in score are grouped. Needs numpy."""
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
    """ROC curves for {name: score}. Legend names get the AUC appended; the colour still follows the bare
    name (series "key"), so each model keeps its colour in the other charts."""
    series, grid = [], None
    for name, sc in scores.items():
        grid, tpr, auc = roc_curve(y_true, sc, points)
        series.append({"name": f"{name}（AUC {auc:.3f}）" if show_auc else name, "key": name, "values": _round(tpr, 4)})
    return line(_round(grid, 4), series, x_label="假阳性率", x_name="假阳性率", x_format=".2f", y_format=".2f",
                x_min=0, x_max=1, y_min=0, y_max=1, diagonal="随机", tags=False, height=height)


def calibration(y_true, probs: dict, *, bins: int = 10, height=None) -> dict:
    """Reliability curve: mean predicted probability vs observed frequency per quantile bin (one model:
    line with markers; several: scatter)."""
    import numpy as np
    y = np.asarray(y_true, dtype=float)
    out = {}
    for name, p in probs.items():
        p = np.asarray(p, dtype=float)
        q = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
        mp = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
        fr = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
        out[name] = (mp, fr)
    if len(out) == 1:
        (name, (mp, fr)), = out.items()
        return line(_round(mp, 4), {name: _round(fr, 4)}, x_label="平均预测概率", x_name="预测", x_format=".2f",
                    y_format=".2f", diagonal="完美校准", markers=True, tags=False, height=height)
    return scatter({k: (v[0], v[1]) for k, v in out.items()}, x_label="平均预测概率", y_label="实际频率",
                   x_format=".2f", y_format=".2f", diagonal="完美校准", height=height)
