"""reportkit: the standard theme for internal report deliverables (single-file HTML research note, prints to A4).

    from reportkit import Report, charts, badge
    r = Report("结论写进标题", dek="……", kicker=["量化研究", "模型评估"], status="review", date="2026-10-24")
    r.kpis([...])
    r.section("主要结果", lead="……")
    with r.row(height=260):
        r.chart(charts.line(x, {"模型": y}), span=8, title="……")
        r.table(rows, span=4, title="……")
    r.save("report.html")

Files
    report.py        Report builder: masthead, headline, KPI strip, takeaways, numbered sections made of
                     12-column rows of panels (charts, tables, text, callouts), actions, appendix -> one HTML file
    charts.py        chart specs (line, bar, waterfall, dumbbell, hist, box, heatmap, scatter, multiples,
                     roc, calibration)
    mpl.py           matplotlib style matching the theme (for figures Report.figure() embeds as SVG)
    example.py       the example report (synthetic data) shown on the site
    assets/          theme.css, theme.js, template.html (hand-written route, no Python), fonts/ (Inter, OFL)
    __main__.py      CLI: example | template | inline | assets
Only the standard library is required; numpy is needed for charts.roc / charts.calibration and the
example. Design rules and the chart spec format: docs/report-theme/README.md.
"""
__version__ = "2.0"

from .report import Badge, Raw, Report, badge, fmt_num, inline_assets  # noqa: E402
from . import charts  # noqa: E402
from .mpl import use_matplotlib_theme  # noqa: E402

__all__ = ["Report", "Badge", "Raw", "badge", "charts", "fmt_num", "inline_assets", "use_matplotlib_theme", "__version__"]
