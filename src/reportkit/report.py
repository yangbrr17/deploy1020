"""Build a single-file HTML report in the reportkit theme.

    from reportkit import Report, charts, badge
    r = Report("方向信号模型评估", subtitle="一句话说明这份报告回答什么问题", kind="技术评估报告",
               status="review", meta={"日期": "2026-10-24", "作者": "（作者）", "版本": "v1.0"})
    r.summary(["**结论一。** 细节和条件。", "**结论二。** ……"], decision="需要确认的事")
    r.kpis([{"label": "AUC（测试集）", "value": 0.81, "fmt": ".3f", "delta": 0.02, "better": "up", "ref": "对比基线"}])
    r.section("数据与口径")
    r.p("正文支持 **加粗**、`代码`、[链接](https://example.org) 和 $x^2$ 行内公式。")
    r.table(rows, title="……", fmt={"AUC": ".3f"}, best={"AUC": "max"})
    r.chart(charts.line(x, {"模型": y}), title="图标题写结论", source="数据：……")
    r.save("report.html")

Contract
- Output is ONE html file: theme.css and theme.js are inlined, charts are JSON specs that theme.js draws
  in the browser (so they follow the light/dark theme and the container width). No network is needed,
  except KaTeX from a CDN when the report contains math; offline, the LaTeX source is shown instead.
- Every string argument is HTML-escaped. The only markup is the inline set above (**bold**, `code`,
  [text](url), $tex$; write a literal dollar as \\$). Report.html() inserts raw HTML you trust.
- Sections, figures and tables are numbered here (1, 1.1, A; 图 n; 表 n), so the numbers in the text,
  the captions and the table of contents always agree. r.chart() and r.table() return their label
  ("图 3"), use it in the text: f"见{fig}".
- Order of a report (docs/report-theme/README.md, "报告结构"): header -> summary -> kpis -> numbered
  sections -> next steps -> appendix -> changelog.
"""
from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import html
import io
import json
import math
import mimetypes
import re
from pathlib import Path

from . import __version__

ASSETS = Path(__file__).resolve().parent / "assets"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

STATUS = {"draft": ("草稿", "rk-st-draft"), "review": ("评审中", "rk-st-review"), "final": ("定稿", "rk-st-final"),
          "草稿": ("草稿", "rk-st-draft"), "评审中": ("评审中", "rk-st-review"), "定稿": ("定稿", "rk-st-final")}
CALLOUTS = {  # kind -> (default title, icon glyph)
    "note": ("说明", "i"), "warn": ("注意", "!"), "risk": ("风险", "!"),
    "good": ("结论", "✓"), "decision": ("需要决策", "?"),
}
TONES = ("good", "warn", "serious", "bad", "info", "neutral")
ACTION_TONES = {"完成": "good", "已完成": "good", "进行中": "info", "未开始": "neutral", "待开始": "neutral",
                "阻塞": "bad", "有风险": "warn", "待定": "warn", "取消": "neutral"}

EARLY_JS = ("(function(){try{var t=localStorage.getItem('rk-theme');"
            "if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}})();")


# ----------------------------------------------------------------------------- inline markup
def esc(s) -> str:
    return html.escape(str(s), quote=True)


class Raw:
    """Trusted HTML, inserted as is."""
    def __init__(self, html_: str):
        self.html = html_

    def __str__(self):
        return self.html


class Badge:
    """Status pill: icon + label (never colour alone). tone: good, warn, serious, bad, info, neutral."""
    def __init__(self, text: str, tone: str = "neutral"):
        if tone not in TONES:
            raise ValueError(f"tone must be one of {TONES}")
        self.text, self.tone = str(text), tone

    def html(self) -> str:
        cls = "" if self.tone == "neutral" else f" rk-{self.tone}"
        return f'<span class="rk-badge{cls}">{esc(self.text)}</span>'

    __str__ = html


def badge(text: str, tone: str = "neutral") -> Badge:
    return Badge(text, tone)


_TOKENS = re.compile(r"`([^`]+)`|(?<!\\)\$([^$\n]+?)(?<!\\)\$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _url(u: str) -> str:
    u = html.unescape(u)
    return esc(u) if re.match(r"^(https?:|mailto:|#|\.{0,2}/|[\w./-]+$)", u) else "#"


def _rich(s: str) -> str:
    s = esc(s).replace("\\$", "$")
    s = _BOLD.sub(r"<b>\1</b>", s)
    return _LINK.sub(lambda m: f'<a href="{_url(m.group(2))}">{m.group(1)}</a>', s)


def inline(text) -> str:
    """**bold**, `code`, [text](url), $tex$ -> HTML; everything else escaped."""
    if isinstance(text, Raw):
        return text.html
    if isinstance(text, Badge):
        return text.html()
    s, out, pos = str(text), [], 0
    for m in _TOKENS.finditer(s):
        out.append(_rich(s[pos:m.start()]))
        if m.group(1) is not None:
            out.append(f"<code>{esc(m.group(1))}</code>")
        else:
            tex = m.group(2)
            out.append(f'<span class="rk-im" data-tex="{esc(tex)}">{esc(tex)}</span>')
        pos = m.end()
    out.append(_rich(s[pos:]))
    return "".join(out)


def _paras(text) -> str:
    if isinstance(text, (Raw, Badge)):
        return inline(text)
    if isinstance(text, (list, tuple)):
        return "<ul>" + "".join(f"<li>{inline(t)}</li>" for t in text) + "</ul>"
    return "".join(f"<p>{inline(p.strip())}</p>" for p in re.split(r"\n\s*\n", str(text)) if p.strip())


# ----------------------------------------------------------------------------- values
def _py(o):
    """numpy / pandas values -> plain JSON-able Python (NaN and inf -> None)."""
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        o = o.tolist()
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and not math.isfinite(v))


def fmt_num(v, spec=None) -> str:
    """Format a number for display. spec: Python format spec ('.3f', ',.0f', '+.2%'), '{:...}', or callable."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if callable(spec):
        s = spec(v)
    elif spec:
        s = spec.format(v) if "{" in spec else format(v, spec)
    elif isinstance(v, int):
        s = f"{v:,}"
    else:
        a = abs(v)
        s = ("0" if a == 0 else f"{v:,.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3g}" if a >= 1e-3 else f"{v:.2e}")
    return "−" + s[1:] if s.startswith("-") else s          # typographic minus


def _json_script(spec) -> str:
    s = json.dumps(_py(spec), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return s.replace("</", "<\\/")


# ----------------------------------------------------------------------------- the report
class Report:
    def __init__(self, title: str, *, subtitle: str | None = None, kind: str | None = None,
                 doc_id: str | None = None, status: str | None = None, meta: dict | None = None,
                 footer: str | None = "内部资料，仅供讨论。", lang: str = "zh-CN", generated: str | None = None,
                 katex: str = "auto"):
        """status: draft / review / final (or 草稿 / 评审中 / 定稿). meta: ordered {label: value}, e.g.
        {"日期": ..., "作者": ..., "版本": ..., "数据": ..., "范围": ...}. katex: "auto" (load KaTeX only if the
        report has math), "cdn" (always) or None (never; LaTeX source shown)."""
        self.title, self.subtitle, self.kind, self.doc_id = title, subtitle, kind, doc_id
        self.status, self.meta, self.footer, self.lang, self.katex = status, meta or {}, footer, lang, katex
        self.generated = generated
        self._blocks: list[str] = []
        self._sec = 0
        self._sub = 0
        self._app = 0                   # appendix letter index
        self._open = False
        self._fig = 0
        self._tab = 0
        self._changelog: list[tuple] = []
        self._math = False

    # ---- structure -----------------------------------------------------------------------------
    def _close(self):
        if self._open:
            self._blocks.append("</section>")
            self._open = False

    def section(self, title: str, lead: str | None = None) -> str:
        """Numbered top-level section (1, 2, ...). Returns its number."""
        if self._app:
            raise RuntimeError("sections after the appendix are not allowed; use subsection()")
        self._close()
        self._sec += 1
        self._sub = 0
        n = str(self._sec)
        self._blocks.append(f'<section class="rk-section"><h2 id="s{n}"><span class="rk-n">{n}</span>{inline(title)}</h2>')
        if lead:
            self._blocks.append(f'<p class="rk-lead">{inline(lead)}</p>')
        self._open = True
        return n

    def subsection(self, title: str) -> str:
        self._sub += 1
        top = chr(64 + self._app) if self._app else str(self._sec)
        n = f"{top}.{self._sub}"
        self._blocks.append(f'<h3 id="s{n.replace(".", "-")}"><span class="rk-n">{n}</span>{inline(title)}</h3>')
        return n

    def appendix(self, title: str) -> str:
        """Appendix section A, B, ... (after all numbered sections)."""
        self._close()
        if not self._app:
            self._blocks.append('<div class="rk-appendix-start"><div class="rk-label">附录</div></div>')
        self._app += 1
        self._sub = 0
        n = chr(64 + self._app)
        self._blocks.append(f'<section class="rk-section rk-appendix"><h2 id="s{n}"><span class="rk-n">{n}</span>{inline(title)}</h2>')
        self._open = True
        return n

    # ---- text ----------------------------------------------------------------------------------
    def p(self, text) -> None:
        """Paragraph(s); a blank line starts a new paragraph."""
        self._blocks.append(_paras(text))

    def lead(self, text) -> None:
        self._blocks.append(f'<p class="rk-lead">{inline(text)}</p>')

    def bullets(self, items, ordered: bool = False) -> None:
        tag = "ol" if ordered else "ul"
        self._blocks.append(f"<{tag}>" + "".join(f"<li>{inline(i)}</li>" for i in items) + f"</{tag}>")

    def defs(self, pairs) -> None:
        """Definition grid: {term: definition} or [(term, definition), ...]."""
        items = pairs.items() if isinstance(pairs, dict) else pairs
        self._blocks.append('<dl class="rk-defs">' + "".join(
            f"<dt>{inline(k)}</dt><dd>{inline(v)}</dd>" for k, v in items) + "</dl>")

    def callout(self, body, title: str | None = None, kind: str = "note") -> None:
        """kind: note (说明), warn (注意), risk (风险), good (结论), decision (需要决策). body: str or list."""
        if kind not in CALLOUTS:
            raise ValueError(f"kind must be one of {tuple(CALLOUTS)}")
        t, icon = CALLOUTS[kind]
        cls = "" if kind == "note" else f" rk-{kind}"
        self._blocks.append(
            f'<div class="rk-callout{cls}" role="note"><span class="rk-ico" aria-hidden="true">{icon}</span>'
            f'<div class="rk-ctitle">{inline(title or t)}</div><div class="rk-cbody">{_paras(body)}</div></div>')

    def code(self, text: str, lang: str | None = None) -> None:
        cls = " rk-has-lang" if lang else ""
        lab = f'<span class="rk-lang">{esc(lang)}</span>' if lang else ""
        lc = f' class="language-{esc(lang)}"' if lang else ""
        self._blocks.append(f'<div class="rk-code{cls}">{lab}<pre><code{lc}>{esc(text.strip(chr(10)))}</code></pre></div>')

    def math(self, tex: str) -> None:
        """Display formula (KaTeX). The block has a "复制 LaTeX" button."""
        self._math = True
        tex = tex.strip()
        self._blocks.append(f'<div class="rk-math" data-tex="{esc(tex)}"><span class="rk-tex">{esc(tex)}</span></div>')

    def html(self, raw: str) -> None:
        self._blocks.append(raw)

    # ---- front matter --------------------------------------------------------------------------
    def summary(self, findings, decision: str | None = None, label: str = "结论") -> None:
        """Bottom line up front. Each finding: "**一句话结论。** 依据和条件". decision: what the reader must decide."""
        items = "".join(f"<li>{inline(f)}</li>" for f in findings)
        dec = (f'<div class="rk-decision"><span class="rk-tag">需要决策</span><p>{inline(decision)}</p></div>'
               if decision else "")
        self._blocks.append(f'<section class="rk-summary" aria-label="{esc(label)}"><div class="rk-label">{esc(label)}</div>'
                            f'<ol class="rk-findings">{items}</ol>{dec}</section>')

    def kpis(self, items) -> None:
        """Stat tiles. Each item: dict(label, value, fmt=None, unit=None, delta=None, delta_fmt=None,
        better="up"|"down"|None, ref=None, note=None, spark=None). Colour of the delta = direction x better."""
        tiles = []
        for k in items:
            v = k.get("value")
            val = esc(fmt_num(v, k.get("fmt")) if _is_num(v) else ("–" if v is None else v))
            unit = f"<small>{esc(k['unit'])}</small>" if k.get("unit") else ""
            parts = [f'<div class="rk-kpi-label">{inline(k["label"])}</div>', f'<div class="rk-kpi-value">{val}{unit}</div>']
            d = k.get("delta")
            if _is_num(d):
                better = k.get("better")
                tone = "flat" if d == 0 or not better else ("good" if (d > 0) == (better == "up") else "bad")
                arrow = "▲" if d > 0 else "▼" if d < 0 else "–"
                ds = fmt_num(d, k.get("delta_fmt") or "+.3g")
                ref = f'<span class="rk-ref">{inline(k["ref"])}</span>' if k.get("ref") else ""
                parts.append(f'<div class="rk-kpi-delta rk-{tone}-t"><span>{arrow} {esc(ds)}</span>{ref}</div>')
            elif k.get("ref"):
                parts.append(f'<div class="rk-kpi-delta rk-flat-t"><span class="rk-ref">{inline(k["ref"])}</span></div>')
            if k.get("note"):
                parts.append(f'<div class="rk-kpi-note">{inline(k["note"])}</div>')
            if k.get("spark") is not None:
                parts.append(f'<div class="rk-spark" data-spark="{esc(_json_script(list(k["spark"])))}"></div>')
            tiles.append('<div class="rk-kpi">' + "".join(parts) + "</div>")
        self._blocks.append('<div class="rk-kpis">' + "".join(tiles) + "</div>")

    # ---- tables --------------------------------------------------------------------------------
    def table(self, data, columns=None, *, title: str | None = None, subtitle: str | None = None,
              source: str | None = None, fmt: dict | None = None, best: dict | None = None,
              bars=None, delta: dict | None = None, highlight=None, index: bool = False,
              numbered: bool = True, cls: str = "") -> str:
        """data: list of dicts, list of lists (+ columns), or a pandas DataFrame.
        fmt   {column: ".3f" | ",.0f" | "+.1%" | callable}   number format per column
        best  {column: "max" | "min"}                       mark the best value (bold + dot)
        bars  [column, ...]                                 inline magnitude bar in the cell
        delta {column: "up" | "down"}                       colour signed values: green when better
        highlight [row index, ...]                          tint whole rows
        Cells may be Badge(...), Raw(...), numbers or strings (inline markup). A row given as the string
        "## 标签" becomes a group-header row. Returns the label ("表 n") or "" when numbered=False."""
        cols, rows = _tabular(data, columns, index)
        fmt, best, delta = fmt or {}, best or {}, delta or {}
        bars, hl = set(bars or ()), set(highlight or ())
        body_rows = [r for r in rows if not isinstance(r, str)]
        numeric = {c: all(_is_num(r[j]) or r[j] is None for r in body_rows) and any(_is_num(r[j]) for r in body_rows)
                   for j, c in enumerate(cols)}
        best_val = {}
        for c, how in best.items():
            j = cols.index(c)
            vals = [r[j] for r in body_rows if _is_num(r[j])]
            if vals:
                best_val[c] = max(vals) if how == "max" else min(vals)
        bar_max = {c: max([abs(r[cols.index(c)]) for r in body_rows if _is_num(r[cols.index(c)])] or [1]) or 1 for c in bars}
        head = "".join(f'<th{" class=rk-r" if numeric[c] else ""} scope="col">{inline(c)}</th>' for c in cols)
        out = []
        ri = -1
        for r in rows:
            if isinstance(r, str):
                out.append(f'<tr class="rk-group"><td colspan="{len(cols)}">{inline(r.lstrip("#").strip())}</td></tr>')
                continue
            ri += 1
            tds = []
            for j, c in enumerate(cols):
                v, classes = r[j], []
                if _is_num(v):
                    classes.append("rk-r")
                    s = esc(fmt_num(v, fmt.get(c)))
                    if c in best_val and v == best_val[c]:
                        classes.append("rk-best")
                    if c in delta and v != 0:
                        classes.append("rk-tg" if (v > 0) == (delta[c] == "up") else "rk-tb")
                    if c in bars:
                        w = 100 * abs(v) / bar_max[c]
                        s = f'<span class="rk-cellbar"><span class="rk-track"><i style="width:{w:.1f}%"></i></span>{s}</span>'
                elif v is None or (isinstance(v, float) and not math.isfinite(v)):
                    classes.append("rk-dim" + (" rk-r" if numeric[c] else ""))
                    s = "–"
                else:
                    s = inline(v)
                    if numeric[c]:
                        classes.append("rk-r")
                ca = f' class="{" ".join(classes)}"' if classes else ""
                tds.append(f"<td{ca}>{s}</td>")
            rc = ' class="rk-hl"' if ri in hl else ""
            out.append(f"<tr{rc}>{''.join(tds)}</tr>")
        label = ""
        if numbered:
            self._tab += 1
            label = f"表 {self._tab}"
        cap = self._caption(label, title, subtitle) if (title or label) else ""
        src = f'<div class="rk-source">{inline(source)}</div>' if source else ""
        self._blocks.append(f'<div class="rk-tbl {esc(cls)}">{cap}<div class="rk-tablewrap"><table class="rk-table">'
                            f"<thead><tr>{head}</tr></thead><tbody>{''.join(out)}</tbody></table></div>{src}</div>")
        return label

    def actions(self, items, title: str | None = None) -> str:
        """Next steps: list of dicts with 事项/item, 负责人/owner, 时间/due, 状态/status (any subset).
        Status strings in ACTION_TONES get a badge (完成, 进行中, 未开始, 阻塞, 有风险, 待定)."""
        keys = [("事项", "item"), ("负责人", "owner"), ("时间", "due"), ("状态", "status")]
        present = [(zh, en) for zh, en in keys if any(zh in i or en in i for i in items)]
        rows = []
        for n, i in enumerate(items, 1):
            row = [f"{n}"]
            for zh, en in present:
                v = i.get(zh, i.get(en))
                if zh == "状态" and isinstance(v, str):
                    v = Badge(v, ACTION_TONES.get(v, "neutral"))
                row.append(v)
            rows.append(row)
        return self.table(rows, ["#"] + [zh for zh, _ in present], title=title, numbered=False, cls="rk-actions")

    # ---- figures -------------------------------------------------------------------------------
    def _caption(self, label, title, subtitle):
        return ('<div class="rk-cap">' + (f'<span class="rk-capn">{esc(label)}</span>' if label else "")
                + (f'<span class="rk-capt">{inline(title)}</span>' if title else "")
                + (f'<span class="rk-caps">{inline(subtitle)}</span>' if subtitle else "") + "</div>")

    def _figure(self, inner: str, title, subtitle, source, numbered) -> str:
        label = ""
        if numbered:
            self._fig += 1
            label = f"图 {self._fig}"
        src = f'<div class="rk-source">{inline(source)}</div>' if source else ""
        self._blocks.append(f'<figure class="rk-fig"><div class="rk-fighead">{self._caption(label, title, subtitle)}</div>'
                            f"{inner}{src}</figure>")
        return label

    def chart(self, spec: dict, title: str | None = None, *, subtitle: str | None = None,
              source: str | None = None, numbered: bool = True) -> str:
        """A chart from reportkit.charts (line, bar, hist, heatmap, scatter, roc, calibration).
        Title = the takeaway ("加平方项后 AUC 只多 0.002"), subtitle = what is plotted and units,
        source = data and conditions. Returns "图 n"."""
        inner = f'<div class="rk-chart"><script type="application/json">{_json_script(spec)}</script></div>'
        return self._figure(inner, title, subtitle, source, numbered)

    def figure(self, fig, title: str | None = None, *, subtitle: str | None = None, source: str | None = None,
               numbered: bool = True) -> str:
        """Embed a matplotlib figure (as SVG) or an SVG string. Call reportkit.use_matplotlib_theme() first
        so fonts and colours match. Static: no tooltip, no table view."""
        if hasattr(fig, "savefig"):
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            svg = buf.getvalue()
        else:
            svg = str(fig)
        svg = re.sub(r"^.*?(<svg)", r"\1", svg, flags=re.S)
        return self._figure(f'<div class="rk-plate">{svg}</div>', title, subtitle, source, numbered)

    def image(self, path, title: str | None = None, *, alt: str = "", subtitle: str | None = None,
              source: str | None = None, numbered: bool = True) -> str:
        """Embed an image file as a data URI (keeps the report a single file)."""
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        data = base64.b64encode(p.read_bytes()).decode()
        return self._figure(f'<div class="rk-plate"><img src="data:{mime};base64,{data}" alt="{esc(alt or title or "")}"></div>',
                            title, subtitle, source, numbered)

    @contextlib.contextmanager
    def cols(self, n: int = 2):
        """Side-by-side figures: `with r.cols(2): r.chart(...); r.chart(...)` (one column on phones)."""
        self._blocks.append(f'<div class="rk-cols" style="--n:{int(n)}">')
        try:
            yield
        finally:
            self._blocks.append("</div>")

    # ---- back matter ---------------------------------------------------------------------------
    def changelog(self, entries) -> None:
        """[(版本, 日期, 说明), ...], newest first. Shown in the footer."""
        self._changelog = list(entries)

    # ---- output --------------------------------------------------------------------------------
    def _uses_math(self, body: str) -> bool:
        return self._math or 'class="rk-im"' in body

    def render(self) -> str:
        self._close()
        body = "".join(self._blocks)
        css = (ASSETS / "theme.css").read_text(encoding="utf8")
        js = (ASSETS / "theme.js").read_text(encoding="utf8")
        st = STATUS.get(self.status) if self.status else None
        kind = "".join(f"<span>{esc(x)}</span>" for x in (self.kind, self.doc_id) if x)
        eyebrow = (f'<div class="rk-eyebrow"><span class="rk-kind">{kind}</span>'
                   + (f'<span class="rk-status {st[1]}">{st[0]}</span>' if st else "") + "</div>")
        meta = ('<dl class="rk-meta">' + "".join(f"<div><dt>{esc(k)}</dt><dd>{inline(v)}</dd></div>"
                                                 for k, v in self.meta.items()) + "</dl>") if self.meta else ""
        head = (f'<header class="rk-head">{eyebrow}<h1 class="rk-title">{inline(self.title)}</h1>'
                + (f'<p class="rk-subtitle">{inline(self.subtitle)}</p>' if self.subtitle else "") + meta + "</header>")
        foot = []
        if self._changelog:
            rows = "".join(f"<tr><td>{esc(v)}</td><td class=\"rk-tnum\">{esc(d)}</td><td>{inline(t)}</td></tr>"
                           for v, d, t in self._changelog)
            foot.append('<div class="rk-tbl"><div class="rk-cap"><span class="rk-capt">版本记录</span></div>'
                        '<div class="rk-tablewrap"><table class="rk-table"><thead><tr><th>版本</th><th>日期</th><th>说明</th></tr>'
                        f"</thead><tbody>{rows}</tbody></table></div></div>")
        gen = self.generated or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        if self.footer:
            foot.append(f"<p>{inline(self.footer)}</p>")
        foot.append(f'<p>由 reportkit {__version__} 生成 · {esc(gen)}</p>')
        use_katex = self.katex == "cdn" or (self.katex == "auto" and self._uses_math(body))
        katex = (f'<link rel="stylesheet" href="{KATEX}/katex.min.css">\n<script defer src="{KATEX}/katex.min.js"></script>\n'
                 if use_katex else "")
        return f"""<!doctype html>
<html lang="{esc(self.lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="generator" content="reportkit {__version__}">
<meta name="robots" content="noindex, nofollow">
<title>{esc(re.sub(r"[*`$]", "", self.title))}</title>
<script>{EARLY_JS}</script>
<style>
{css}
</style>
{katex}</head>
<body class="rk">
<div class="rk-tools"><button type="button" data-rk="theme" aria-label="切换深浅色">◐ 深浅色</button><button type="button" data-rk="print">打印 / PDF</button></div>
<div class="rk-page">
<nav class="rk-toc" aria-label="目录"></nav>
<article class="rk-paper">
{head}
{body}
<footer class="rk-foot">{"".join(foot)}</footer>
</article>
</div>
<script>
{js}
</script>
</body>
</html>
"""

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(), encoding="utf8")
        return p


def _tabular(data, columns, index):
    """-> (column names, rows as lists). Accepts DataFrame, list of dicts, list of lists."""
    if hasattr(data, "columns") and hasattr(data, "to_dict"):           # pandas DataFrame
        cols = [str(c) for c in data.columns]
        rows = [[_py(v) for v in r] for r in data.itertuples(index=False, name=None)]
        if index:
            name = str(data.index.name) if data.index.name is not None else ""
            cols = [name] + cols
            rows = [[_py(i)] + r for i, r in zip(data.index, rows)]
        return cols, rows
    data = list(data)
    dicts = [r for r in data if isinstance(r, dict)]
    if dicts:
        cols = list(columns) if columns else list(dict.fromkeys(k for r in dicts for k in r))
        return cols, [r if isinstance(r, str) else [_py(r.get(c)) for c in cols] for r in data]
    if columns is None:
        raise ValueError("list-of-lists data needs columns=[...]")
    return list(columns), [r if isinstance(r, str) else [_py(v) for v in r] for r in data]


def inline_assets(src_html: str) -> str:
    """Replace <link href="...theme.css"> and <script src="...theme.js"></script> with the files' content,
    so a hand-written report (see assets/template.html) becomes a single file."""
    css = (ASSETS / "theme.css").read_text(encoding="utf8")
    js = (ASSETS / "theme.js").read_text(encoding="utf8")
    out = re.sub(r'<link[^>]+href="[^"]*theme\.css"[^>]*>', lambda m: f"<style>\n{css}\n</style>", src_html)
    out = re.sub(r'<script[^>]+src="[^"]*theme\.js"[^>]*>\s*</script>', lambda m: f"<script>\n{js}\n</script>", out)
    return out
