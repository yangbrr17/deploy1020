"""Tests for the report theme builder (src/reportkit). Rendering in a browser is checked with Playwright by hand
(screenshots, overlap and alignment checks); these cover escaping, the grid contract, numbering, tables, chart
specs and the single-file output."""
import json
import math
import re

import numpy as np
import pytest

from reportkit import Report, badge, charts, fmt_num, inline_assets
from reportkit.report import ASSETS, inline


def _specs(html):
    return [json.loads(s.replace("<\\/", "</")) for s in
            re.findall(r'<div class="rk-chart"><script type="application/json">(.*?)</script>', html, re.S)]


def test_inline_markup_and_escaping():
    assert inline("**a** `<b>` [x](https://e.org)") == '<b>a</b> <code>&lt;b&gt;</code> <a href="https://e.org">x</a>'
    assert 'href="#"' in inline("[x](javascript:alert(1))")
    assert inline("<script>") == "&lt;script&gt;"
    assert 'data-tex="x^2"' in inline("见 $x^2$")
    assert inline(r"价格 \$5 和 \$6") == "价格 $5 和 $6"


def test_title_and_spec_cannot_break_out():
    r = Report("<script>alert(1)</script>")
    r.section("s")
    r.chart(charts.line([1, 2], {"</script><b>": [1, 2]}), title="t")
    h = r.render()
    assert "<script>alert(1)" not in h and "</script><b>" not in h
    assert _specs(h)[0]["series"][0]["name"] == "</script><b>"


def test_rows_must_fill_twelve_columns():
    r = Report("t")
    r.section("一")
    with r.row(height=200):
        r.chart(charts.bar(["a"], [1]), span=8)
        r.text("x", span=4)
    with r.row():                                   # spans left out are split evenly
        r.text("a")
        r.text("b")
        r.text("c")
    h = r.render()
    assert h.count('style="--span:4"') == 4 and 'style="--span:8"' in h and 'style="--h:200px"' in h
    with pytest.raises(ValueError, match="add up to 10"):
        with r.row():
            r.text("a", span=6)
            r.text("b", span=4)
    with pytest.raises(ValueError):
        with r.row():
            r.text("a", span=5)
            r.text("b")
            r.text("c")                             # 7 left for 2 panels: not even
    with pytest.raises(ValueError):
        r.text("alone", span=6)                     # outside a row a panel is full width
    with pytest.raises(RuntimeError):
        with r.row():
            with r.row():
                pass


def test_numbering_sections_and_panels():
    r = Report("t", kicker=["a", "b"], status="review", date="2026-01-01", meta={"作者": "x"})
    assert r.section("一", lead="结论") == "1"
    assert r.chart(charts.bar(["a", "b"], [1, 2]), title="x") == "图 1"
    assert r.table([{"a": 1}], title="y") == "表 1"
    assert r.section("二") == "2"
    assert r.appendix("附") == "A"
    with pytest.raises(RuntimeError):
        r.section("too late")
    h = r.render()
    for s in ('id="s1"', 'id="s2"', 'id="sA"', "rk-appendix-start", 'class="rk-nav"', "rk-st-review", "<dt>作者</dt>"):
        assert s in h
    assert h.count("<section") == h.count("</section>")
    # every panel has head, body and foot (the subgrid needs all three to align a row)
    assert h.count('class="rk-ph"') == h.count('class="rk-pb"') == h.count('class="rk-pf"') == 2


def test_table_formatting():
    r = Report("t")
    r.table([{"m": "a", "v": 0.5, "d": -0.01, "n": 1200, "h": -2.0, "s": badge("ok", "good"), "t": [1, 2, 3]},
             "## 分组",
             {"m": "b", "v": 0.7, "d": 0.02, "n": None, "h": 4.0, "s": "x", "t": [3, 2, 1]}],
            fmt={"v": ".2f", "d": "+.2f"}, best={"v": "max"}, delta={"d": "up"}, bars=["v"], heat={"h": "diverging"},
            spark=["t"], highlight=[1], total=[1])
    h = r.render()
    assert 'class="rk-r rk-best"' in h and "0.70" in h
    assert "rk-tb" in h and "rk-tg" in h and "−0.01" in h
    assert "1,200" in h and 'class="rk-dim rk-r"' in h
    assert 'class="rk-group"' in h and 'class="rk-hl rk-total"' in h
    assert "rk-badge rk-good" in h and "rk-cellbar" in h and 'data-spark="[1,2,3]"' in h
    assert "var(--rk-down)" in h and "var(--rk-up)" in h          # heat cells
    assert 'class="rk-table rk-wide"' in h                         # >= 5 columns scroll on phones


def test_fmt_num():
    assert fmt_num(1234) == "1,234"
    assert fmt_num(-0.5, ".1%") == "−50.0%"
    assert fmt_num(float("nan")) == "–"
    assert fmt_num(3.14159, "{:.2f}") == "3.14"


def test_chart_specs():
    s = charts.line(np.arange(3), {"a": np.array([1.0, np.nan, 3.0])}, tags=False)
    assert s["series"][0]["values"] == [1.0, None, 3.0] and s["tags"] is False
    json.dumps(s, allow_nan=False)
    h = charts.hist(list(np.random.default_rng(0).exponential(1, 1000)), bins=20, log_x=True)
    assert len(h["edges"]) == 21 and sum(h["series"][0]["counts"]) == 1000
    b = charts.bar(["a", "b", "c"], [2, 3, 1], sort="desc", tone="sign")
    assert b["categories"] == ["b", "a", "c"] and b["tone"] == "sign" and "horizontal" not in b
    assert charts.bar(["a"], [1], horizontal=False)["horizontal"] is False
    w = charts.waterfall([("start", 1.0, "total"), ("up", 0.5), ("end", 1.5, "total")], y_min=0.5)
    assert [x["kind"] for x in w["steps"]] == ["total", "delta", "total"] and w["yMin"] == 0.5
    d = charts.dumbbell(["x", "y"], {"before": [1, 2]}, {"name": "after", "values": [3, 1]}, sort="desc")
    assert d["categories"] == ["x", "y"] and d["a"]["name"] == "before" and d["b"]["values"] == [3, 1]
    bx = charts.box({"g": [1, 2, 3, 4, 5]})["groups"][0]
    assert (bx["min"], bx["q1"], bx["median"], bx["q3"], bx["max"], bx["n"]) == (1, 2, 3, 4, 5, 5)
    m = charts.multiples(["d1", "d2"], {"a": [1, 2], "b": [2, 1]}, share_y=False)
    assert len(m["panels"]) == 2 and m["shareY"] is False
    hm = charts.heatmap(np.eye(2), ["x", "y"], ["x", "y"], labels=False)
    assert hm["values"] == [[1.0, 0.0], [0.0, 1.0]] and hm["labels"] is False
    sc = charts.scatter({"s": ([1, 2, 3], [2, 4, 6], ["a", "b", "c"])}, fit=True)
    assert sc["fit"] is True and sc["series"][0]["labels"] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        charts.scatter({str(i): ([1], [1]) for i in range(4)})


def test_roc_and_calibration():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.5
    score = rng.normal(0, 1, 4000) + 1.0 * y          # binormal, d' = 1: AUC = Phi(1/sqrt 2) = 0.760
    grid, tpr, auc = charts.roc_curve(y, score)
    assert abs(auc - 0.760) < 0.02 and np.all(np.diff(tpr) >= 0) and abs(tpr[-1] - 1) < 1e-12
    s = charts.roc(y, {"m": score})
    assert s["series"][0]["key"] == "m" and s["diagonal"] == "随机" and s["tags"] is False
    c = charts.calibration(y, {"m": 1 / (1 + np.exp(-score))})
    assert c["type"] == "line" and len(c["x"]) == 10


def test_example_note_is_single_file(tmp_path):
    from reportkit.example import build
    out = tmp_path / "r.html"
    build(out, generated="2026-01-01 00:00")
    h = out.read_text(encoding="utf8")
    assert 'href="theme.css"' not in h and "data:font/woff2;base64," in h
    specs = _specs(h)
    assert {s["type"] for s in specs} >= {"line", "bar", "waterfall", "dumbbell", "hist", "box", "heatmap", "scatter", "multiples"}
    for s in specs:
        for ser in s.get("series", []) + s.get("panels", []):
            vals = ser.get("values") or ser.get("counts") or ser.get("y") or []
            assert all(v is None or math.isfinite(v) for v in vals)
    assert "katex" in h                               # the appendix has formulas


def test_template_inlines():
    out = inline_assets((ASSETS / "template.html").read_text(encoding="utf8"))
    assert 'href="theme.css"' not in out and 'src="theme.js"' not in out and "data:font/woff2" in out
    assert ".rk-panel" in out and "window.reportkit" in out
    specs = _specs(out.replace('<div class="rk-chart"><script type="application/json">\n', '<div class="rk-chart"><script type="application/json">'))
    assert specs and all(s["type"] in {"line", "bar", "waterfall", "heatmap"} for s in specs)
    spans = [int(x) for x in re.findall(r"--span:(\d+)", out)]
    assert sum(spans) % 12 == 0                       # hand-written rows also fill 12 columns
