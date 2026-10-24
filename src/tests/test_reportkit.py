"""Tests for the report theme builder (src/reportkit). Rendering in a browser is checked by hand / Playwright;
these cover escaping, numbering, tables, chart specs and the single-file output."""
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
    assert "<script>alert(1)" not in h
    assert "</script><b>" not in h
    assert _specs(h)[0]["series"][0]["name"] == "</script><b>"


def test_numbering_and_labels():
    r = Report("t")
    assert r.section("一") == "1"
    assert r.subsection("a") == "1.1"
    assert r.chart(charts.bar(["a", "b"], [1, 2]), title="x") == "图 1"
    assert r.table([{"a": 1}], title="y") == "表 1"
    assert r.section("二") == "2"
    assert r.subsection("b") == "2.1"
    assert r.appendix("附") == "A"
    assert r.subsection("c") == "A.1"
    with pytest.raises(RuntimeError):
        r.section("too late")
    h = r.render()
    for i in ('id="s1"', 'id="s1-1"', 'id="s2-1"', 'id="sA"', 'id="sA-1"', "rk-appendix-start"):
        assert i in h
    assert h.count("<section") == h.count("</section>")


def test_table_formatting():
    r = Report("t")
    r.table([{"m": "a", "v": 0.5, "d": -0.01, "n": 1200, "s": badge("ok", "good")},
             "## 分组",
             {"m": "b", "v": 0.7, "d": 0.02, "n": None, "s": "x"}],
            fmt={"v": ".2f", "d": "+.2f"}, best={"v": "max"}, delta={"d": "up"}, bars=["v"], highlight=[1])
    h = r.render()
    assert 'class="rk-r rk-best"' in h and "0.70" in h
    assert "rk-tb" in h and "rk-tg" in h                 # -0.01 worse, +0.02 better
    assert "−0.01" in h                             # typographic minus
    assert "1,200" in h and 'class="rk-dim rk-r"' in h  # None in a numeric column
    assert 'class="rk-group"' in h and 'class="rk-hl"' in h
    assert 'rk-badge rk-good' in h and "rk-cellbar" in h


def test_fmt_num():
    assert fmt_num(1234) == "1,234"
    assert fmt_num(-0.5, ".1%") == "−50.0%"
    assert fmt_num(float("nan")) == "–"
    assert fmt_num(3.14159, "{:.2f}") == "3.14"


def test_chart_specs_clean_numpy_and_nan():
    s = charts.line(np.arange(3), {"a": np.array([1.0, np.nan, 3.0])}, end_labels=False)
    assert s["series"][0]["values"] == [1.0, None, 3.0] and s["endLabels"] is False
    json.dumps(s, allow_nan=False)
    h = charts.hist(list(np.random.default_rng(0).exponential(1, 1000)), bins=20, log_x=True)
    assert len(h["edges"]) == 21 and sum(h["series"][0]["counts"]) == 1000
    b = charts.bar(["a", "b", "c"], [2, 3, 1], sort="desc")
    assert b["categories"] == ["b", "a", "c"] and "horizontal" not in b
    assert charts.bar(["a"], [1], horizontal=False)["horizontal"] is False
    hm = charts.heatmap(np.eye(2), ["x", "y"], ["x", "y"], labels=False)
    assert hm["values"] == [[1.0, 0.0], [0.0, 1.0]] and hm["labels"] is False
    with pytest.raises(ValueError):
        charts.scatter({str(i): ([1], [1]) for i in range(4)})


def test_roc_and_calibration():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.5
    score = rng.normal(0, 1, 4000) + 1.0 * y          # binormal, d' = 1: AUC = Phi(1/sqrt 2) = 0.760
    grid, tpr, auc = charts.roc_curve(y, score)
    assert abs(auc - 0.760) < 0.02 and np.all(np.diff(tpr) >= 0) and abs(tpr[-1] - 1) < 1e-12
    s = charts.roc(y, {"m": score})
    assert s["series"][0]["key"] == "m" and s["diagonal"] == "随机"
    c = charts.calibration(y, {"m": 1 / (1 + np.exp(-score))})
    assert c["type"] == "line" and len(c["x"]) == 10


def test_example_report_is_single_file(tmp_path):
    from reportkit.example import build
    out = tmp_path / "r.html"
    build(out, generated="2026-01-01 00:00")
    h = out.read_text(encoding="utf8")
    assert '<link rel="stylesheet" href="theme.css"' not in h and "<style>" in h
    specs = _specs(h)
    assert {s["type"] for s in specs} >= {"line", "bar", "hist", "heatmap"}
    for s in specs:
        for ser in s.get("series", []):
            vals = ser.get("values") or ser.get("counts") or ser.get("y") or []
            assert all(v is None or math.isfinite(v) for v in vals)
    assert "katex" in h                                 # the appendix has formulas


def test_template_inlines():
    t = (ASSETS / "template.html").read_text(encoding="utf8")
    out = inline_assets(t)
    assert 'href="theme.css"' not in out and 'src="theme.js"' not in out
    assert ".rk-paper" in out and "window.reportkit" in out
    for s in _specs(out):                               # hand-written specs must be valid JSON
        assert s["type"] in {"line", "bar", "hist", "heatmap", "scatter"}
