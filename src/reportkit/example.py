"""The example note shown on the site: a direction-signal model evaluation on synthetic data.

    PYTHONPATH=src python -m reportkit example _site/report/example.html

It uses every chart type and panel kind once, laid out on the 12-column grid in the order
docs/report-theme/README.md prescribes, so it doubles as the reference for how a note reads.
Every number in the text is computed below from data generated with a fixed seed. Needs numpy.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from . import Report, badge, charts, fmt_num

FEATURES = ["盘口不平衡", "订单流不平衡 5s", "订单流不平衡 30s", "微价格偏离", "价差", "成交强度"]
HOURS = ["0–1h", "1–2h", "2–3h", "3–4h", "4–5h", "5–6h"]
DAYS, PER_DAY, TRAIN_DAYS = 40, 3000, 25
N = NormalDist()


# ----------------------------------------------------------------------------- data and models
def simulate(seed: int = 7):
    """Correlated features; a label whose log-odds is mostly linear with one saturating term; a day-level and
    an hour-of-day drift in signal strength (stronger after the open and before the close)."""
    rng = np.random.default_rng(seed)
    C = np.array([[1.0, 0.45, 0.30, 0.60, -0.10, 0.15],
                  [0.45, 1.0, 0.70, 0.35, 0.00, 0.30],
                  [0.30, 0.70, 1.0, 0.25, 0.05, 0.35],
                  [0.60, 0.35, 0.25, 1.0, -0.05, 0.10],
                  [-0.10, 0.00, 0.05, -0.05, 1.0, -0.20],
                  [0.15, 0.30, 0.35, 0.10, -0.20, 1.0]])
    n = DAYS * PER_DAY
    X = rng.standard_normal((n, 6)) @ np.linalg.cholesky(C).T
    day = np.repeat(np.arange(DAYS), PER_DAY)
    hour = np.tile(np.repeat(np.arange(6), PER_DAY // 6), DAYS)
    strength = (1 + rng.normal(0, 0.18, DAYS)[day]) * np.array([1.25, 1.08, 0.97, 0.9, 0.98, 1.15])[hour]
    logit = strength * (0.85 * X[:, 0] + 0.45 * np.tanh(1.6 * X[:, 1]) + 0.18 * X[:, 2] + 0.22 * X[:, 3] + 0.05 * X[:, 5])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    burst = rng.random(60_000) < 0.68           # inter-arrival times: short gaps inside bursts, long gaps between
    gaps = np.where(burst, np.exp(rng.normal(np.log(0.025), 0.9, 60_000)), rng.exponential(14.0, 60_000))
    return X, y, day, hour, gaps


def fit_logit(X, y, ridge=1e-6):
    """Logistic regression by Newton-Raphson (IRLS); coefficients with the intercept first."""
    A = np.c_[np.ones(len(X)), X]
    b = np.zeros(A.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(A @ b)))
        step = np.linalg.solve((A * (p * (1 - p))[:, None]).T @ A + ridge * np.eye(len(b)), A.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def predict(b, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ b)))


def auc(y, s):
    r = np.empty(len(s))
    r[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def auc_var(a, n1, n0):
    """Hanley-McNeil variance of an AUC estimate."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)


def balanced_acc(y, p, t=0.5):
    hit = p > t
    return float((hit[y == 1].mean() + (~hit[y == 0]).mean()) / 2)


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def max_ba(a):
    """Best balanced accuracy over thresholds for an equal-variance binormal score with this AUC."""
    return N.cdf(N.inv_cdf(a) / np.sqrt(2))


def auc_for_ba(ba):
    return N.cdf(np.sqrt(2) * N.inv_cdf(ba))


def evaluate(seed: int = 7) -> dict:
    X, y, day, hour, gaps = simulate(seed)
    tr, te = day < TRAIN_DAYS, day >= TRAIN_DAYS
    test_days = range(TRAIN_DAYS, DAYS)
    designs = {"单特征": lambda Z: Z[:, :1], "线性 6 特征": lambda Z: Z, "线性 + 非线性": lambda Z: np.c_[Z, Z ** 2, Z ** 3]}
    yt, dt, ht = y[te], day[te], hour[te]
    E = {"y": yt, "gaps": gaps, "models": {}}
    for name, f in designs.items():
        b = fit_logit(f(X[tr]), y[tr])
        p = predict(b, f(X[te]))
        E["models"][name] = dict(b=b, p=p, f=f, auc=auc(yt, p), ba=balanced_acc(yt, p), ll=log_loss(yt, p),
                                 brier=float(np.mean((p - yt) ** 2)),
                                 daily=np.array([auc(yt[dt == d], p[dt == d]) for d in test_days]))
    m = E["models"]
    rng = np.random.default_rng(seed + 1)

    def boot(d):              # bootstrap over test days (days are the independent units)
        s = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        return float(d.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))
    lin, one, nl = m["线性 6 特征"], m["单特征"], m["线性 + 非线性"]
    E["d_lin"], E["d_nl"] = boot(lin["daily"] - one["daily"]), boot(nl["daily"] - lin["daily"])
    E["days_better"] = int(np.sum(lin["daily"] > one["daily"]))
    cb = fit_logit(np.log(lin["p"] / (1 - lin["p"]))[:, None], yt)
    E["cal_intercept"], E["cal_slope"] = float(cb[0]), float(cb[1])
    var = np.array([auc_var(a, yt[dt == d].sum(), (1 - yt[dt == d]).sum()) for a, d in zip(lin["daily"], test_days)])
    w = 1 / var
    q = float(np.sum(w * (lin["daily"] - np.sum(w * lin["daily"]) / w.sum()) ** 2))
    E["Q"], E["I2"] = q, max(0.0, (q - (len(var) - 1)) / q)
    # hour-of-day x day AUC of the linear model
    E["hourly"] = np.array([[auc(yt[(dt == d) & (ht == k)], lin["p"][(dt == d) & (ht == k)]) for d in test_days] for k in range(6)])
    # permutation importance (AUC drop) for the linear model
    Xt = X[te]
    imp = []
    for j in range(6):
        drops = []
        for _ in range(3):
            Z = Xt.copy()
            Z[:, j] = rng.permutation(Z[:, j])
            drops.append(lin["auc"] - auc(yt, predict(lin["b"], lin["f"](Z))))
        imp.append(float(np.mean(drops)))
    E["importance"] = imp
    E["corr"] = np.corrcoef(Xt, rowvar=False)
    # single-feature AUC: daily on the test period, and train vs test
    E["feat_daily"] = {f: [auc(yt[dt == d], Xt[dt == d, j]) for d in test_days] for j, f in enumerate(FEATURES)}
    E["feat_train"] = [auc(y[tr], X[tr][:, j]) for j in range(6)]
    E["feat_test"] = [auc(yt, Xt[:, j]) for j in range(6)]
    return E


# ----------------------------------------------------------------------------- the note
def build(path, generated: str | None = None) -> Report:
    E = evaluate()
    m = E["models"]
    one, lin, nl = m["单特征"], m["线性 6 特征"], m["线性 + 非线性"]
    d_lin, d_nl = E["d_lin"], E["d_nl"]
    n_te, n_days = len(E["y"]), DAYS - TRAIN_DAYS
    days = [f"D{d + 1}" for d in range(TRAIN_DAYS, DAYS)]
    need = auc_for_ba(0.75)
    sgn = lambda v, f=".4f": fmt_num(v, "+" + f)          # noqa: E731  signed, typographic minus
    hourly_mean = E["hourly"].mean(1)
    best_h, worst_h = int(np.argmax(hourly_mean)), int(np.argmin(hourly_mean))

    r = Report(
        f"线性模型已经到顶：AUC {lin['auc']:.3f}，再加非线性项只多 {nl['auc'] - lin['auc']:.4f}",
        dek="同一组特征下三种模型的样本外表现、balanced accuracy 与 AUC 的换算、日间与日内稳定性，以及下一步该把精力放在哪里。"
            "本文是报告主题的示例，数据为合成数据。",
        kicker=["量化研究", "方向信号", "模型评估"], kind="研究简报", doc_id="示例 · 合成数据",
        status="review", date="2026-10-24", version="v1.0",
        meta={"作者": "（作者）", "数据": f"{DAYS} 个交易日 × {PER_DAY:,} 样本", "切分": f"前 {TRAIN_DAYS} 天训练，后 {n_days} 天测试",
              "标签": "下一次中间价变动的方向"},
        generated=generated,
    )
    r.kpis([
        {"label": "AUC · 线性 6 特征", "value": lin["auc"], "fmt": ".3f", "delta": lin["auc"] - one["auc"], "delta_fmt": "+.3f",
         "better": "up", "ref": "对比单特征", "spark": lin["daily"]},
        {"label": "Balanced accuracy", "value": lin["ba"], "fmt": ".3f", "delta": lin["ba"] - one["ba"], "delta_fmt": "+.3f",
         "better": "up", "ref": "对比单特征", "note": f"二元正态上限 {max_ba(lin['auc']):.3f}"},
        {"label": "对数损失", "value": lin["ll"], "fmt": ".4f", "delta": lin["ll"] - one["ll"], "delta_fmt": "+.4f",
         "better": "down", "ref": "对比单特征"},
        {"label": "非线性项增益 · AUC", "value": d_nl[0], "fmt": "+.4f", "note": f"门槛 +0.005；区间 [{sgn(d_nl[1])}, {sgn(d_nl[2])}]"},
        {"label": "单日 AUC 区间", "value": f"{lin['daily'].min():.3f}–{lin['daily'].max():.3f}",
         "note": f"I² = {E['I2']:.0%}，{n_days} 个测试日"},
    ])
    with r.row():
        r.takeaways([
            f"**多特征线性模型比单特征高 {d_lin[0]:+.3f} AUC，而且每天都更好。** 按天 bootstrap 的 95% 区间 "
            f"[{d_lin[1]:+.3f}, {d_lin[2]:+.3f}]，{n_days} 个测试日里 {E['days_better']} 天更好。",
            f"**非线性项几乎没有增益。** 加平方项和立方项后 AUC 只多 {d_nl[0]:.4f}，远低于 0.005 的门槛；模型结构不是瓶颈。",
            f"**balanced accuracy 停在 {lin['ba']:.2f} 是 AUC 的直接结果。** 二元正态假设下 AUC {lin['auc']:.3f} 对应的上限是 "
            f"{max_ba(lin['auc']):.3f}；要到 0.75，AUC 需要 {need:.3f}。",
            f"**日间和日内差异都大于抽样误差。** 单日 AUC 在 {lin['daily'].min():.3f}–{lin['daily'].max():.3f} 之间（I² = {E['I2']:.0%}）；"
            f"按开盘后小时看，{HOURS[best_h]} 最强（{hourly_mean[best_h]:.3f}），{HOURS[worst_h]} 最弱（{hourly_mean[worst_h]:.3f}）。",
        ], span=8)
        r.info({"数据": f"{DAYS} 个交易日，每天 {PER_DAY:,} 个等间隔时点（合成）", "标签": "时点之后中间价第一次变动的方向",
                "特征": f"{len(FEATURES)} 个盘口与订单流特征，按天标准化", "模型": "单特征、线性 6 特征、线性 + 非线性（逻辑回归）",
                "检验": "按测试日 bootstrap；Cochran Q"},
               decision="下一阶段把工作从模型结构转到标签定义和低相关的新特征（第 6 节）。", span=4)

    # 1 ---------------------------------------------------------------------------------------------
    r.section("主要结果", lead="唯一明显的提升来自从单特征到 6 个特征这一步；之后的非线性项可以忽略。")
    with r.row(height=250):
        t1 = r.table([
            {"模型": k, "特征": n_f, "AUC": v["auc"], "Δ AUC": v["auc"] - one["auc"], "Bal. acc.": v["ba"],
             "对数损失": v["ll"], "单日 AUC": [round(x, 4) for x in v["daily"]]}
            for (k, v), n_f in zip(m.items(), (1, 6, 18))
        ], span=7, title="线性模型明显好于单特征，非线性项没有再带来提升",
            subtitle=f"测试集 {n_te:,} 个样本；Δ 相对单特征", fmt={"AUC": ".4f", "Δ AUC": "+.4f", "Bal. acc.": ".4f", "对数损失": ".4f"},
            best={"AUC": "max", "Bal. acc.": "max", "对数损失": "min"}, delta={"Δ AUC": "up"}, spark=["单日 AUC"], highlight=[1],
            source="方块标出每列最好的值；单日 AUC 是测试期 15 天的走势。")
        r.chart(charts.waterfall([("单特征", round(one["auc"], 4), "total"), ("另外 5 个线性特征", round(lin["auc"] - one["auc"], 4)),
                                  ("平方项和立方项", round(nl["auc"] - lin["auc"], 4)), ("最终模型", round(nl["auc"], 4), "total")],
                                 value_format=".4f", y_min=0.74),
                span=5, title=f"AUC 从 {one['auc']:.3f} 到 {nl['auc']:.3f}，几乎全部来自线性特征",
                subtitle="AUC 分解，测试集；纵轴从 0.74 开始", source="逐步加入特征后重新拟合，每一步的 AUC 差。")
    with r.row(height=230):
        r.chart(charts.roc(E["y"], {k: v["p"] for k, v in m.items()}), span=4,
                title="三条 ROC 曲线几乎重合在两条上", subtitle="真阳性率 vs 假阳性率，测试集")
        r.chart(charts.calibration(E["y"], {"线性 6 特征": lin["p"]}), span=4,
                title="预测概率校准良好", subtitle="实际上涨频率 vs 平均预测概率，10 个分位数分箱",
                source=f"校准斜率 {E['cal_slope']:.3f}，截距 {fmt_num(E['cal_intercept'], '+.3f')}。")
        grid = np.round(np.linspace(0.5, 0.95, 46), 3)
        r.chart(charts.line(grid.tolist(), [{"name": "balanced accuracy 上限", "values": [round(max_ba(a), 4) for a in grid], "color": "s7"}],
                            x_label="AUC", x_name="AUC", x_format=".2f", y_format=".2f", tags=False,
                            refs=[{"y": 0.75, "label": "0.75"}],
                            labels=[{"x": round(one["auc"], 3), "y": round(one["ba"], 3), "text": "单特征", "dx": -9, "dy": -8, "anchor": "end"},
                                    {"x": round(lin["auc"], 3), "y": round(lin["ba"], 3), "text": "线性 6 特征", "dx": 9, "dy": 15}]),
                span=4, title=f"balanced accuracy 到 0.75 需要 AUC {need:.3f}", subtitle="二元正态等方差假设下的上限；圆点为实测",
                source=r"$\mathrm{BA}_{\max} = \Phi(\Phi^{-1}(\mathrm{AUC})/\sqrt{2})$，见附录 A。")
    with r.row():
        r.callout(f"balanced accuracy 依赖阈值，AUC 不依赖。实测 {lin['ba']:.3f} 和 AUC 对应的上限 {max_ba(lin['auc']):.3f} "
                  "几乎相同，调阈值没有提升空间。比较模型以 AUC 和对数损失为准。", title="不要用 balanced accuracy 比较模型", kind="warn", span=6)
        r.callout("测试集只用训练期之后的日子。按行随机切分时，相邻几秒的样本会同时出现在训练集和测试集，分数会明显偏高。",
                  title="为什么按天切分", kind="note", span=6)

    # 2 ---------------------------------------------------------------------------------------------
    r.section("稳定性", lead=f"线性模型在 {E['days_better']} / {n_days} 个测试日都更好，但所有模型一起随日子和时段波动。")
    with r.row(height=250):
        r.chart(charts.line(days, {"线性 6 特征": lin["daily"].round(4), "单特征": one["daily"].round(4)}, x_name="测试日", y_format=".3f"),
                span=8, title="两条线同涨同跌：日间波动来自市场，不来自模型", subtitle="单日 AUC，测试期",
                source="合成数据里信号强度逐日变化；右侧标签为最后一天的值。")
        gain = nl["daily"] - lin["daily"]
        r.chart(charts.bar(days, {"非线性增益": gain.round(5)}, tone="sign", value_format="+.4f", labels=False),
                span=4, title=f"非线性项的逐日增益都在 {np.abs(gain).max():.4f} 以内", subtitle="单日 AUC 差：线性 + 非线性 − 线性 6 特征",
                source=f"{int(np.sum(gain > 0))} 天为正，{int(np.sum(gain <= 0))} 天为负或为零。")
    with r.row(height=240):
        r.chart(charts.box({k: v["daily"] for k, v in m.items()}, value_format=".3f"), span=4,
                title="三个模型的单日 AUC 分布", subtitle="箱体为四分位，粗线为中位数，点为每一天")
        r.chart(charts.scatter({"测试日": (one["daily"], lin["daily"], days)}, x_label="单特征 AUC", y_label="线性 6 特征 AUC",
                               x_format=".2f", y_format=".2f", diagonal="y = x", fit=True), span=4,
                title="每一天都在对角线上方", subtitle="单日 AUC：单特征（横轴）vs 线性 6 特征（纵轴）")
        r.chart(charts.heatmap(E["hourly"].round(3), HOURS, days, scale="sequential", value_format=".3f", value_name="AUC",
                               scale_label="AUC"), span=4,
                title=f"{HOURS[best_h]} 信号最强，{HOURS[worst_h]} 最弱", subtitle="线性 6 特征，按开盘后小时 × 测试日的 AUC",
                source="每格约 500 个样本，单格噪声较大，看行的整体深浅。")

    # 3 ---------------------------------------------------------------------------------------------
    r.section("特征", lead="盘口不平衡贡献了大部分排序能力；6 个特征里有两对高度相关，单独看系数会误判。")
    with r.row(height=260):
        r.chart(charts.heatmap(np.round(E["corr"], 2), FEATURES, FEATURES, domain=(-1, 1), value_name="相关系数"), span=6,
                title=f"两对特征高度相关：订单流的两个窗口 {E['corr'][1, 2]:.2f}，盘口不平衡与微价格偏离 {E['corr'][0, 3]:.2f}",
                subtitle="Pearson 相关系数，测试集")
        r.chart(charts.bar(FEATURES, {"AUC 下降": [round(v, 4) for v in E["importance"]]}, sort="desc", value_format=".4f",
                           highlight=FEATURES[0], horizontal=True), span=6,
                title=f"打乱盘口不平衡，AUC 下降 {E['importance'][0]:.2f}；其余都在 {max(E['importance'][1:]):.2f} 以内", subtitle="置换重要性：打乱一列后 AUC 的下降，重复 3 次取平均",
                source="高相关的特征互相分担重要性，所以「订单流不平衡 30s」偏低。")
    with r.row(height=250):
        r.chart(charts.multiples(days, {f: [round(v, 4) for v in E["feat_daily"][f]] for f in FEATURES}, y_format=".2f",
                                 refs=[{"y": 0.5}], cols=3), span=8,
                title=f"单个特征每天的排序能力都稳定；价差始终在 {min(E['feat_daily']['价差']):.2f}–{max(E['feat_daily']['价差']):.2f}，没有信息",
                subtitle="单特征 AUC，测试期逐日；虚线为 0.5（无信息）")
        r.chart(charts.dumbbell(FEATURES, {"name": "训练期", "values": [round(v, 4) for v in E["feat_train"]]},
                                {"name": "测试期", "values": [round(v, 4) for v in E["feat_test"]]}, value_format=".3f", sort="desc"),
                span=4, title=f"训练期到测试期最多变化 {max(abs(a - b) for a, b in zip(E['feat_train'], E['feat_test'])):.3f}",
                subtitle="单特征 AUC，按测试期排序")

    # 4 ---------------------------------------------------------------------------------------------
    raw = DAYS * PER_DAY
    drop = [int(raw * f) for f in (0.031, 0.004, 0.012)]
    total0 = raw + sum(drop)
    lg = np.log10(E["gaps"])
    hc, he = np.histogram(lg, bins=48)
    mid = len(hc) // 2
    peaks = [10 ** ((he[i] + he[i + 1]) / 2) for i in (int(np.argmax(hc[:mid])), mid + int(np.argmax(hc[mid:])))]
    r.section("数据", lead=f"清洗后保留 {raw / total0:.1%} 的时点；事件间隔是双峰分布，两类的预测概率重叠很多。")
    with r.row(height=230):
        r.table([
            {"步骤": "原始时点", "样本数": total0, "占比": 1.0, "处理": None},
            {"步骤": "剔除：集合竞价时段", "样本数": -drop[0], "占比": -drop[0] / total0, "处理": badge("已剔除", "info")},
            {"步骤": "剔除：交叉盘口（买一 ≥ 卖一）", "样本数": -drop[1], "占比": -drop[1] / total0, "处理": badge("已剔除", "info")},
            {"步骤": "剔除：特征缺失", "样本数": -drop[2], "占比": -drop[2] / total0, "处理": badge("已剔除", "info")},
            {"步骤": "进入建模", "样本数": raw, "占比": raw / total0, "处理": badge("通过", "good")},
        ], span=5, title=f"清洗后保留 {raw / total0:.1%} 的时点", subtitle="样本数，40 个交易日合计",
            fmt={"样本数": ",d", "占比": ".1%"}, bars=["占比"], total=[4], source="合成数据；剔除比例是示例值。")
        r.chart(charts.hist(E["gaps"].tolist(), bins=48, log_x=True, x_label="秒（对数刻度）",
                            refs=[{"x": peaks[0], "label": f"{peaks[0]:.2f} s"}, {"x": peaks[1], "label": f"{peaks[1]:.1f} s"}]),
                span=7, title="事件间隔取对数后是双峰，不是指数分布的单峰", subtitle=f"{len(E['gaps']):,} 个间隔，48 个等比分箱，事件数",
                source="合成数据：68% 来自簇内（对数正态），32% 来自簇间（指数）。")
    with r.row(height=220):
        p, yt = lin["p"], E["y"]
        r.chart(charts.hist(series={"实际上涨": p[yt == 1].tolist(), "实际下跌": p[yt == 0].tolist()}, bins=40, range=(0, 1),
                            x_format=".1f", refs=[{"x": 0.5, "label": "阈值 0.5"}]),
                span=7, title="两类的预测概率大面积重叠，这就是 AUC 的上限", subtitle="线性 6 特征的预测概率，按实际方向分开，样本数")
        r.text([f"标签：时点之后中间价**第一次变动**的方向；之后没有变动的时点不进入样本。",
                "切分：按天，前 25 天训练，后 15 天测试，避免相邻时点泄漏。",
                "指标：AUC 衡量排序，对数损失衡量概率质量；balanced accuracy 只作参考。"],
               span=5, title="口径", label="定义")

    # 5 ---------------------------------------------------------------------------------------------
    r.section("检验与风险", lead="四个假设里，非线性增益不成立，日间稳定性存疑；结论依赖标签定义。")
    with r.row():
        v1, v2, v3, v4 = d_lin[1] > 0, d_nl[2] >= 0.005, 0.9 <= E["cal_slope"] <= 1.1, E["I2"] < 0.5
        yes, no = badge("成立", "good"), badge("不成立", "bad")
        r.table([
            {"编号": "H1", "假设": "线性 6 特征的 AUC 高于单特征", "统计量": f"ΔAUC {sgn(d_lin[0])}",
             "95% 区间": f"[{sgn(d_lin[1])}, {sgn(d_lin[2])}]", "结论": yes if v1 else no},
            {"编号": "H2", "假设": "非线性项让 AUC 至少多 0.005", "统计量": f"ΔAUC {sgn(d_nl[0])}",
             "95% 区间": f"[{sgn(d_nl[1])}, {sgn(d_nl[2])}]", "结论": yes if v2 else no},
            {"编号": "H3", "假设": "校准斜率在 [0.9, 1.1] 之内", "统计量": f"斜率 {E['cal_slope']:.3f}", "95% 区间": "—",
             "结论": yes if v3 else no},
            {"编号": "H4", "假设": "单日 AUC 的差异只来自抽样误差", "统计量": f"Q {E['Q']:.1f}，I² {E['I2']:.0%}", "95% 区间": "—",
             "结论": yes if v4 else badge("待定：需要更多天", "warn")},
        ], span=8, title="非线性增益不成立，日间稳定性存疑", subtitle="H1、H2：按测试日 bootstrap 5,000 次；H4：Cochran Q",
            source="单日 AUC 的方差用 Hanley–McNeil 公式。")
        r.callout(["合成数据的标签机制已知；真实数据要先确认标签的填充方向。",
                   f"测试期只有 {n_days} 天，H4 对天数敏感。",
                   "上限换算假设两类得分近似正态、方差相等。"], title="结论成立的前提", kind="risk", span=4)

    # 6 ---------------------------------------------------------------------------------------------
    r.section("下一步", lead="把精力从模型结构转到标签和特征。")
    r.actions(title="四项后续工作，按顺序做", items=[
        {"事项": "在真实数据上确认标签填充方向：画 10 分钟的买一、卖一、中间价和标签", "负责人": "（待定）", "时间": "第 1 周", "状态": "进行中"},
        {"事项": "换成「下一次变动方向」和「30 秒三分类」两版标签，重跑本报告", "负责人": "（待定）", "时间": "第 1 周", "状态": "未开始"},
        {"事项": "加入与现有特征相关性低的新特征（全市场、跨品种）", "负责人": "（待定）", "时间": "第 2 周", "状态": "未开始"},
        {"事项": "测试期延长到 40 天，重新检验 H4；按开盘后小时分别报告", "负责人": "（待定）", "时间": "第 3 周", "状态": "有风险"},
    ])

    # appendix --------------------------------------------------------------------------------------
    r.appendix("指标定义", lead="$s$ 是模型得分，$y \\in \\{0, 1\\}$ 是标签，$\\Phi$ 是标准正态分布函数。")
    r.math(r"\mathrm{AUC} = \Pr\left(s_i > s_j \mid y_i = 1,\ y_j = 0\right)")
    r.math(r"\mathrm{BA}(t) = \tfrac{1}{2}\left[\Pr(s > t \mid y = 1) + \Pr(s \le t \mid y = 0)\right]")
    r.math(r"\mathrm{BA}_{\max} = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{AUC})}{\sqrt{2}}\right), \qquad \mathrm{AUC} = \Phi\!\left(\sqrt{2}\,\Phi^{-1}(\mathrm{BA}_{\max})\right)")
    r.math(r"\mathrm{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\right]")
    r.appendix("复现", lead="所有数字都在 `src/reportkit/example.py` 里从固定种子的合成数据算出。")
    r.code("PYTHONPATH=src python -m reportkit example report.html", lang="bash")
    r.code('''from reportkit import Report, charts

r = Report("结论写进标题", dek="回答什么问题、用什么数据", kicker=["量化研究"], status="review", date="2026-10-24")
r.kpis([{"label": "AUC", "value": 0.784, "fmt": ".3f", "delta": 0.024, "better": "up", "ref": "对比基线"}])
r.section("主要结果", lead="这一节的结论")
with r.row(height=250):
    r.chart(charts.line(days, {"模型 A": a, "模型 B": b}, y_format=".3f"), span=8, title="图标题写结论")
    r.chart(charts.bar(names, values, sort="desc"), span=4, title="……")
r.save("report.html")''', lang="python")
    r.changelog([("v1.0", "2026-10-24", "首次发布")])
    r.save(path)
    return r
