"""reportkit: the standard theme for internal report deliverables (single-file HTML, prints to A4).

    from reportkit import Report, charts, badge
    r = Report("标题", subtitle="……", kind="技术评估报告", status="review", meta={"日期": "2026-10-24"})
    ...
    r.save("report.html")

Files
    report.py        Report builder: header, summary, KPI tiles, numbered sections, callouts, tables,
                     figures, actions, appendix, changelog -> one HTML file
    charts.py        chart specs (line, bar, hist, heatmap, scatter, roc, calibration)
    mpl.py           matplotlib style matching the theme (for figures Report.figure() embeds as SVG)
    example.py       the example report (synthetic data) shown on the site
    assets/          theme.css, theme.js, template.html (hand-written route, no Python)
    __main__.py      CLI: example | template | inline | assets
Only the standard library is required; numpy is needed for charts.roc / charts.calibration and the
example. Design rules and the chart spec format: docs/report-theme/README.md.
"""
__version__ = "1.0"

from .report import Badge, Raw, Report, badge, fmt_num, inline_assets  # noqa: E402
from . import charts  # noqa: E402
from .mpl import use_matplotlib_theme  # noqa: E402

__all__ = ["Report", "Badge", "Raw", "badge", "charts", "fmt_num", "inline_assets", "use_matplotlib_theme", "__version__"]
