"""The example report shown on the site: a direction-signal model evaluation on synthetic data.

    PYTHONPATH=src python -m reportkit example _site/report/example.html

It uses every component once, in the order docs/report-theme/README.md prescribes, so it doubles as the
reference for how a report reads. Every number in the text is computed below from data generated with a
fixed seed; nothing is typed in by hand. Needs numpy.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from . import Report, badge, charts, fmt_num

FEATURES = ["盘口不平衡", "订单流不平衡 5s", "订单流不平衡 30s", "微价格偏离", "价差", "成交强度"]
DAYS, PER_DAY, TRAIN_DAYS = 40, 3000, 25
N = NormalDist()


# ----------------------------------------------------------------------------- data and models
def simulate(seed: int = 7):
    """Correlated features, a label whose log-odds is mostly linear with one saturating term, and a
    day-level drift in signal strength (the reason daily AUCs differ by more than sampling error)."""
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
    strength = 1 + rng.normal(0, 0.18, DAYS)[day]
    logit = strength * (0.85 * X[:, 0] + 0.45 * np.tanh(1.6 * X[:, 1]) + 0.18 * X[:, 2] + 0.22 * X[:, 3] + 0.05 * X[:, 5])
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    # inter-arrival times of a clustered event stream: short gaps inside bursts, long gaps between them
    burst = rng.random(60_000) < 0.68
    gaps = np.where(burst, np.exp(rng.normal(np.log(0.025), 0.9, 60_000)), rng.exponential(14.0, 60_000))
    return X, y.astype(float), day, gaps


def fit_logit(X, y, ridge=1e-6):
    """Logistic regression by Newton-Raphson (IRLS); returns coefficients with the intercept first."""
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
    """Best balanced accuracy over thresholds for a binormal equal-variance score with this AUC."""
    return N.cdf(N.inv_cdf(a) / np.sqrt(2))


def auc_for_ba(ba):
    return N.cdf(np.sqrt(2) * N.inv_cdf(ba))


def evaluate(seed: int = 7) -> dict:
    X, y, day, gaps = simulate(seed)
    tr, te = day < TRAIN_DAYS, day >= TRAIN_DAYS
    designs = {
        "单特征": lambda Z: Z[:, :1],
        "线性 6 特征": lambda Z: Z,
        "线性 + 非线性": lambda Z: np.c_[Z, Z ** 2, Z ** 3],
    }
    out = {"y": y[te], "day": day[te], "models": {}, "gaps": gaps, "X_te": X[te]}
    for name, f in designs.items():
        b = fit_logit(f(X[tr]), y[tr])
        p = predict(b, f(X[te]))
        daily = [auc(y[te][day[te] == d], p[day[te] == d]) for d in range(TRAIN_DAYS, DAYS)]
        out["models"][name] = dict(b=b, p=p, auc=auc(y[te], p), ba=balanced_acc(y[te], p), ll=log_loss(y[te], p),
                                   brier=float(np.mean((p - y[te]) ** 2)), daily=np.array(daily), f=f)
    m = out["models"]
    rng = np.random.default_rng(seed + 1)

    def boot(d):          # bootstrap over test days (days are the independent units)
        idx = rng.integers(0, len(d), (5000, len(d)))
        s = d[idx].mean(1)
        return float(d.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))

    out["d_lin"] = boot(m["线性 6 特征"]["daily"] - m["单特征"]["daily"])
    out["d_nl"] = boot(m["线性 + 非线性"]["daily"] - m["线性 6 特征"]["daily"])
    out["days_better"] = int(np.sum(m["线性 6 特征"]["daily"] > m["单特征"]["daily"]))
    # calibration slope: regress y on logit(p) of the linear model
    p = m["线性 6 特征"]["p"]
    cb = fit_logit(np.log(p / (1 - p))[:, None], y[te])
    out["cal_intercept"], out["cal_slope"] = float(cb[0]), float(cb[1])
    # heterogeneity of daily AUC (Cochran Q, I^2)
    a = m["线性 6 特征"]["daily"]
    var = np.array([auc_var(ai, y[te][day[te] == d].sum(), (1 - y[te][day[te] == d]).sum())
                    for ai, d in zip(a, range(TRAIN_DAYS, DAYS))])
    w = 1 / var
    q = float(np.sum(w * (a - np.sum(w * a) / w.sum()) ** 2))
    out["Q"], out["I2"] = q, max(0.0, (q - (len(a) - 1)) / q)
    # permutation importance (AUC drop) for the linear model
    lin = m["线性 6 特征"]
    base = lin["auc"]
    imp = []
    Xt = X[te]
    for j in range(6):
        drops = []
        for _ in range(3):
            Z = Xt.copy()
            Z[:, j] = rng.permutation(Z[:, j])
            drops.append(base - auc(y[te], predict(lin["b"], lin["f"](Z))))
        imp.append(float(np.mean(drops)))
    out["importance"] = imp
    out["corr"] = np.corrcoef(Xt, rowvar=False)
    return out


# ----------------------------------------------------------------------------- the report
def build(path, generated: str | None = None) -> Report:
    E = evaluate()
    m = E["models"]
    one, lin, nl = m["单特征"], m["线性 6 特征"], m["线性 + 非线性"]
    d_lin, d_nl = E["d_lin"], E["d_nl"]
    n_te = len(E["y"])
    need = auc_for_ba(0.75)

    r = Report(
        "方向信号模型评估：线性模型够用，下一步该换标签和特征",
        subtitle="同一组特征下三种模型的样本外表现、balanced accuracy 与 AUC 的换算，以及日间稳定性。"
                 "本报告是主题示例，数据为合成数据。",
        kind="技术评估报告", doc_id="示例 · 合成数据", status="review",
        meta={"日期": "2026-10-24", "作者": "（作者）", "版本": "v1.0",
              "数据": f"{DAYS} 个交易日 × {PER_DAY:,} 样本（合成）", "切分": f"前 {TRAIN_DAYS} 天训练，后 {DAYS - TRAIN_DAYS} 天测试"},
        generated=generated,
    )
    r.summary([
        f"**6 个特征的线性模型比单特征高 {d_lin[0]:+.3f} AUC。** 按天 bootstrap 的 95% 区间是 "
        f"[{d_lin[1]:+.3f}, {d_lin[2]:+.3f}]，{DAYS - TRAIN_DAYS} 个测试日里有 {E['days_better']} 天更好。",
        f"**非线性项几乎没有增益。** 加平方项和立方项后 AUC 只变化 {d_nl[0]:+.4f}（区间 [{d_nl[1]:+.4f}, {d_nl[2]:+.4f}]），"
        "低于 0.005 的门槛。",
        f"**balanced accuracy 停在 {lin['ba']:.2f} 是 AUC 的直接结果，不是阈值没调好。** 在二元正态假设下，AUC "
        f"{lin['auc']:.3f} 对应的 balanced accuracy 上限是 {max_ba(lin['auc']):.3f}；要到 0.75，AUC 需要 {need:.3f}。",
        f"**日间差异大于抽样误差。** 单日 AUC 在 {lin['daily'].min():.3f} 到 {lin['daily'].max():.3f} 之间，"
        f"I² = {E['I2']:.0%}，报告单一的平均 AUC 会掩盖这部分波动。",
    ], decision="是否把下一阶段的工作从模型结构转到标签定义和新特征上（第 7 节）。")
    r.kpis([
        {"label": "AUC · 测试集", "value": lin["auc"], "fmt": ".3f", "delta": lin["auc"] - one["auc"], "delta_fmt": "+.3f",
         "better": "up", "ref": "对比单特征", "spark": lin["daily"]},
        {"label": "Balanced accuracy", "value": lin["ba"], "fmt": ".3f", "delta": lin["ba"] - one["ba"], "delta_fmt": "+.3f",
         "better": "up", "ref": "对比单特征", "note": f"上限 {max_ba(lin['auc']):.3f}（二元正态）"},
        {"label": "对数损失", "value": lin["ll"], "fmt": ".4f", "delta": lin["ll"] - one["ll"], "delta_fmt": "+.4f",
         "better": "down", "ref": "对比单特征"},
        {"label": "测试样本", "value": n_te, "ref": f"{DAYS - TRAIN_DAYS} 天 × {PER_DAY:,}", "note": "按天切分，无重叠"},
    ])

    # 1 ------------------------------------------------------------------------------------------
    r.section("背景与问题")
    r.p("方向信号的分类准确率长期停在 70% 出头，换模型只多一两个百分点。这份报告回答三个问题：")
    r.bullets([
        "多特征的线性模型比单一特征好多少，这个提升在不同的日子里是否稳定；",
        "加入非线性项能否带来有意义的提升（门槛：AUC 至少多 0.005）；",
        "balanced accuracy 卡在某个数附近，是阈值问题还是排序能力本身的上限。",
    ])

    # 2 ------------------------------------------------------------------------------------------
    r.section("数据与口径")
    r.defs({
        "样本": f"每个交易日 {PER_DAY:,} 个等间隔时点，共 {DAYS} 天（合成数据）。",
        "标签": "时点之后中间价第一次变动的方向（上 = 1，下 = 0）；之后没有变动的时点不进入样本。",
        "特征": "、".join(FEATURES) + "，均已按天标准化。",
        "切分": f"前 {TRAIN_DAYS} 天训练，后 {DAYS - TRAIN_DAYS} 天测试。按天切分，避免相邻时点同时出现在训练集和测试集。",
        "指标": "AUC（排序能力，不依赖阈值）、balanced accuracy（阈值 0.5）、对数损失、Brier 分数。定义见附录 A。",
    })
    raw = DAYS * PER_DAY
    drop = [int(raw * f) for f in (0.031, 0.004, 0.012)]
    r.table([
        {"步骤": "原始时点", "样本数": raw + sum(drop), "占比": 1.0, "处理": None},
        {"步骤": "剔除：集合竞价时段", "样本数": -drop[0], "占比": -drop[0] / (raw + sum(drop)), "处理": badge("已剔除", "info")},
        {"步骤": "剔除：交叉盘口（买一 ≥ 卖一）", "样本数": -drop[1], "占比": -drop[1] / (raw + sum(drop)), "处理": badge("已剔除", "info")},
        {"步骤": "剔除：特征缺失", "样本数": -drop[2], "占比": -drop[2] / (raw + sum(drop)), "处理": badge("已剔除", "info")},
        {"步骤": "进入建模", "样本数": raw, "占比": raw / (raw + sum(drop)), "处理": badge("通过", "good")},
    ], title=f"清洗后保留 {raw / (raw + sum(drop)):.1%} 的时点", fmt={"样本数": ",d", "占比": ".1%"}, highlight=[4],
        source="合成数据；剔除比例是示例值。")
    gaps = E["gaps"]
    lg = np.log10(gaps)
    hcounts, hedges = np.histogram(lg, bins=48)
    mid = len(hcounts) // 2
    peaks = [10 ** ((hedges[i] + hedges[i + 1]) / 2) for i in (int(np.argmax(hcounts[:mid])), mid + int(np.argmax(hcounts[mid:])))]
    r.p(f"事件之间的时间间隔不是指数分布。取对数后有两个峰（{peaks[0]:.2f} 秒和 {peaks[1]:.1f} 秒），"
        "分别对应同一簇内部的短间隔和簇与簇之间的长间隔。这影响「下一笔什么时候来」的建模，但不影响本报告的方向标签。")
    r.chart(charts.hist(gaps.tolist(), bins=48, log_x=True, x_label="间隔（秒，对数刻度）", y_label="事件数",
                        refs=[{"x": peaks[0], "label": f"{peaks[0]:.2f} s"}, {"x": peaks[1], "label": f"{peaks[1]:.1f} s"}]),
            title="事件间隔取对数后是双峰，不是指数分布的单峰",
            subtitle=f"{len(gaps):,} 个间隔，48 个等比分箱", source="合成数据：68% 来自簇内（对数正态），32% 来自簇间（指数）。")
    r.callout("测试集只用训练期之后的日子。如果按行随机切分，相邻几秒的样本会同时出现在训练集和测试集里，"
              "测试分数会明显偏高。", kind="note", title="为什么按天切分")

    # 3 ------------------------------------------------------------------------------------------
    r.section("主要结果")
    t1 = r.table([
        {"模型": k, "特征数": n_f, "AUC": v["auc"], "Balanced acc.": v["ba"], "对数损失": v["ll"], "Brier": v["brier"],
         "Δ AUC": v["auc"] - one["auc"]}
        for (k, v), n_f in zip(m.items(), (1, 6, 18))
    ], title="线性模型明显好于单特征，非线性项没有再带来提升",
        subtitle=f"测试集，{n_te:,} 个样本", fmt={"AUC": ".4f", "Balanced acc.": ".4f", "对数损失": ".4f", "Brier": ".4f", "Δ AUC": "+.4f"},
        best={"AUC": "max", "Balanced acc.": "max", "对数损失": "min", "Brier": "min"}, delta={"Δ AUC": "up"},
        source="Δ AUC 相对单特征模型。圆点标出每列最好的值。")
    r.p(f"{t1} 里三个模型的差别集中在第一步：从单特征到 6 个特征。之后再加 12 个非线性项，"
        f"AUC 只变化 {fmt_num(nl['auc'] - lin['auc'], '+.4f')}，对数损失变化 {fmt_num(nl['ll'] - lin['ll'], '+.4f')}。")
    with r.cols(2):
        r.chart(charts.roc(E["y"], {k: v["p"] for k, v in m.items()}),
                title="三条 ROC 曲线几乎重合在两条上", subtitle="测试集")
        r.chart(charts.calibration(E["y"], {"线性 6 特征": lin["p"]}),
                title="线性模型的预测概率校准良好", subtitle="10 个分位数分箱",
                source=f"校准斜率 {E['cal_slope']:.3f}，截距 {fmt_num(E['cal_intercept'], '+.3f')}。")
    days = [f"D{d + 1}" for d in range(TRAIN_DAYS, DAYS)]
    r.chart(charts.line(days, {"单特征": one["daily"].round(4), "线性 6 特征": lin["daily"].round(4)},
                        y_label="AUC", x_name="测试日", y_format=".3f", markers=True),
            title=f"线性模型在 {E['days_better']} / {DAYS - TRAIN_DAYS} 个测试日更好，但两者一起随日子波动",
            subtitle="单日 AUC，测试期", source="日间波动来自合成数据里逐日变化的信号强度。")
    grid = np.round(np.linspace(0.5, 0.95, 46), 3)
    r.chart(charts.line(grid.tolist(), [{"name": "balanced accuracy 上限", "values": [round(max_ba(a), 4) for a in grid], "color": "s7"}],
                        x_label="AUC", y_label="balanced accuracy", x_name="AUC", x_format=".2f", y_format=".3f",
                        refs=[{"y": 0.75, "label": "0.75"}],
                        labels=[{"x": round(one["auc"], 3), "y": round(one["ba"], 3), "text": "单特征（实测）", "dx": -10, "dy": -8, "anchor": "end"},
                                {"x": round(lin["auc"], 3), "y": round(lin["ba"], 3), "text": "线性 6 特征（实测）", "dx": 10, "dy": 16}]),
            title=f"要让 balanced accuracy 到 0.75，AUC 需要 {need:.3f}",
            subtitle="二元正态等方差假设下，balanced accuracy 的最大可能值随 AUC 的变化；圆点是两个模型的实测值",
            source=r"公式见附录 A：$\mathrm{BA}_{\max} = \Phi\left(\Phi^{-1}(\mathrm{AUC})/\sqrt{2}\right)$。")
    gap = max_ba(lin["auc"]) - lin["ba"]
    r.callout(f"balanced accuracy 依赖阈值，AUC 不依赖。实测的 {lin['ba']:.3f} 和 AUC 对应的上限 "
              f"{max_ba(lin['auc']):.3f} 相差 {abs(gap):.3f}，调阈值没有提升空间。"
              "比较模型时以 AUC 和对数损失为主。", kind="warn", title="不要用 balanced accuracy 比较模型")

    # 4 ------------------------------------------------------------------------------------------
    r.section("特征诊断")
    r.subsection("相关性")
    r.p("「订单流不平衡 5s」和「30s」高度相关，「盘口不平衡」和「微价格偏离」也是。两组各自更像一个信号的两种写法，"
        "单独看系数会误判谁重要。")
    r.chart(charts.heatmap(np.round(E["corr"], 2), FEATURES, FEATURES, domain=(-1, 1), value_name="相关系数"),
            title="6 个特征分成两组高相关的对", subtitle="测试集上的 Pearson 相关系数")
    r.subsection("重要性")
    r.chart(charts.bar(FEATURES, {"AUC 下降": [round(v, 4) for v in E["importance"]]}, sort="desc", value_format=".4f",
                       value_label="打乱该特征后 AUC 的下降"),
            title="盘口不平衡贡献了大部分排序能力", subtitle="置换重要性：在测试集上打乱一列，重复 3 次取平均",
            source="高相关的特征会互相分担重要性，所以「订单流不平衡 30s」的数值偏低。")

    # 5 ------------------------------------------------------------------------------------------
    r.section("假设检验")
    v1 = d_lin[1] > 0
    v2 = d_nl[2] >= 0.005
    v3 = 0.9 <= E["cal_slope"] <= 1.1
    v4 = E["I2"] < 0.5
    r.table([
        {"编号": "H1", "假设": "线性 6 特征的 AUC 高于单特征", "统计量": f"ΔAUC = {d_lin[0]:+.4f}",
         "95% 区间": f"[{d_lin[1]:+.4f}, {d_lin[2]:+.4f}]", "结论": badge("成立", "good") if v1 else badge("不成立", "bad")},
        {"编号": "H2", "假设": "非线性项让 AUC 至少多 0.005", "统计量": f"ΔAUC = {d_nl[0]:+.4f}",
         "95% 区间": f"[{d_nl[1]:+.4f}, {d_nl[2]:+.4f}]", "结论": badge("成立", "good") if v2 else badge("不成立", "bad")},
        {"编号": "H3", "假设": "校准斜率在 [0.9, 1.1] 之内", "统计量": f"斜率 = {E['cal_slope']:.3f}",
         "95% 区间": "—", "结论": badge("成立", "good") if v3 else badge("不成立", "bad")},
        {"编号": "H4", "假设": "单日 AUC 的差异只来自抽样误差", "统计量": f"Q = {E['Q']:.1f}，I² = {E['I2']:.0%}",
         "95% 区间": "—", "结论": badge("成立", "good") if v4 else badge("待定：需要更多天", "warn")},
    ], title="四个假设里，非线性增益不成立，日间稳定性存疑",
        source="H1、H2：按测试日做 5,000 次 bootstrap。H4：Cochran Q，单日 AUC 方差用 Hanley–McNeil 公式。")

    # 6 ------------------------------------------------------------------------------------------
    r.section("风险与局限")
    r.callout([
        "合成数据的标签机制是已知的，真实数据里还要先确认标签的填充方向（向前还是向后）。",
        f"测试期只有 {DAYS - TRAIN_DAYS} 天，H4 的结论对天数敏感。",
        "balanced accuracy 上限的换算假设两类得分近似正态、方差相等；偏离较大时只作参考。",
    ], kind="risk", title="结论成立的前提")
    r.p("这些局限不改变主要结论的方向：多特征线性模型有稳定的增益，非线性项没有。")

    # 7 ------------------------------------------------------------------------------------------
    r.section("下一步")
    r.actions([
        {"事项": "在真实数据上确认标签填充方向，画 10 分钟的买一、卖一、中间价和标签", "负责人": "（待定）", "时间": "第 1 周", "状态": "进行中"},
        {"事项": "换成「下一次变动方向」和「30 秒三分类」两版标签，重跑本报告", "负责人": "（待定）", "时间": "第 1 周", "状态": "未开始"},
        {"事项": "加入与现有特征相关性低的新特征（全市场、跨品种）", "负责人": "（待定）", "时间": "第 2 周", "状态": "未开始"},
        {"事项": "把测试期延长到 40 天，重新检验 H4", "负责人": "（待定）", "时间": "第 3 周", "状态": "有风险"},
    ])

    # appendix -----------------------------------------------------------------------------------
    r.appendix("指标定义")
    r.p("设 $s$ 是模型得分，$y \\in \\{0, 1\\}$ 是标签，$n_1$、$n_0$ 是正负样本数。")
    r.math(r"\mathrm{AUC} = \Pr\left(s_i > s_j \mid y_i = 1,\ y_j = 0\right)")
    r.math(r"\mathrm{BA}(t) = \tfrac{1}{2}\left[\Pr(s > t \mid y = 1) + \Pr(s \le t \mid y = 0)\right]")
    r.p("两类得分是方差相等的正态分布时，balanced accuracy 在最优阈值处的值只由 AUC 决定：")
    r.math(r"\mathrm{BA}_{\max} = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{AUC})}{\sqrt{2}}\right), \qquad \mathrm{AUC} = \Phi\!\left(\sqrt{2}\,\Phi^{-1}(\mathrm{BA}_{\max})\right)")
    r.math(r"\mathrm{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\right]")
    r.appendix("复现")
    r.p("报告由下面的命令生成，所有数字都在 `src/reportkit/example.py` 里从固定种子的合成数据算出：")
    r.code("PYTHONPATH=src python -m reportkit example report.html", lang="bash")
    r.code('''from reportkit import Report, charts, badge

r = Report("标题写结论", subtitle="回答什么问题", kind="技术评估报告", status="review",
           meta={"日期": "2026-10-24", "作者": "（作者）", "版本": "v1.0"})
r.summary(["**一句话结论。** 依据和条件。"], decision="需要读者决定的事")
r.section("主要结果")
r.chart(charts.line(x, {"模型 A": a, "模型 B": b}, y_label="AUC"), title="图标题写结论")
r.save("report.html")''', lang="python")
    r.changelog([("v1.0", "2026-10-24", "首次发布")])
    r.save(path)
    return r
