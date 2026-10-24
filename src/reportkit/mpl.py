"""matplotlib style that matches the reportkit theme, for figures embedded with Report.figure().

    import matplotlib.pyplot as plt
    from reportkit import use_matplotlib_theme
    use_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

Prefer reportkit.charts where it can draw the chart (it adapts to dark mode and has tooltips and a
table view); use matplotlib for what it cannot (custom plots, many panels).
"""
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC",
                        "Source Han Sans SC", "SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": "#6b6a65",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ecebe5", "grid.linewidth": 0.8,
    "xtick.color": "#898781", "ytick.color": "#898781", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "ytick.left": False, "xtick.major.size": 3, "xtick.major.width": 0.8,
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.markersize": 5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    "figure.dpi": 110, "svg.fonttype": "none",
}


def use_matplotlib_theme() -> None:
    import matplotlib as mpl
    from cycler import cycler          # ships with matplotlib
    mpl.rcParams.update(RC)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
