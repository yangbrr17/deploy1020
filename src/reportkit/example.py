"""The example note shown on the site: a direction-signal model evaluation on synthetic data.

    PYTHONPATH=src python -m reportkit example _site/report/example.html
    PYTHONPATH=src python -m reportkit example _site/report/example-en.html --lang en

The Chinese and English notes share one layout; only TEXT differs. It uses every chart type and panel kind once, laid out on the 12-column grid in the order
docs/report-theme/README.md prescribes, so it doubles as the reference for how a note reads.
Every number in the text is computed below from data generated with a fixed seed. Needs numpy.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from . import Report, badge, charts, fmt_num

FEATURES = ["盘口不平衡", "订单流不平衡 5s", "订单流不平衡 30s", "微价格偏离", "价差", "成交强度"]
HOURS = [f"{a}\u2060–\u2060{a + 1}h" for a in range(6)]       # word joiners: "0–1h" never breaks across lines
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


# ----------------------------------------------------------------------------- the note, in two languages
# One layout, two texts: every string the note shows lives in TEXT[lang]; values are str.format templates
# filled from the computed context `c` in build(). Keys ending in "_raw" contain LaTeX and are not formatted.
FEATURES_EN = ["Book imbalance", "Order-flow imbalance 5s", "Order-flow imbalance 30s", "Micro-price deviation", "Spread",
               "Trade intensity"]
TEXT = {
    "zh": {
        "features": FEATURES, "models": ["单特征", "线性 6 特征", "线性 + 非线性"],
        "title": "线性模型已经到顶：AUC {auc:.3f}，再加非线性项只多 {gain:.4f}",
        "dek": "同一组特征下三种模型的样本外表现、balanced accuracy 与 AUC 的换算、日间与日内稳定性，以及下一步该把精力放在哪里。"
               "本文是报告主题的示例，数据为合成数据。",
        "kicker": ["量化研究", "方向信号", "模型评估"], "doc_id": "示例 · 合成数据", "date": "2026-10-24",
        "meta": {"作者": "（作者）", "数据": "{days} 个交易日 × {per:,} 样本", "切分": "前 {train} 天训练，后 {n} 天测试",
                 "标签": "下一次中间价变动的方向"},
        "k_auc": "AUC · 线性 6 特征", "k_ba": "Balanced accuracy", "k_ll": "对数损失", "k_gain": "非线性项增益 · AUC",
        "k_range": "单日 AUC 区间", "vs": "对比单特征", "k_cap": "二元正态上限 {cap:.3f}", "k_gain_note": "门槛 +0.005；区间 [{glo}, {ghi}]",
        "k_range_note": "I² = {i2:.0%}，{n} 个测试日",
        "points": [
            "**多特征线性模型比单特征高 {d:+.3f} AUC，而且每天都更好。** 按天 bootstrap 的 95% 区间 [{dlo:+.3f}, {dhi:+.3f}]，"
            "{n} 个测试日里 {k} 天更好。",
            "**非线性项几乎没有增益。** 加平方项和立方项后 AUC 只多 {g:.4f}，远低于 0.005 的门槛；模型结构不是瓶颈。",
            "**balanced accuracy 停在 {ba:.2f} 是 AUC 的直接结果。** 二元正态假设下 AUC {auc:.3f} 对应的上限是 {cap:.3f}；"
            "要到 0.75，AUC 需要 {need:.3f}。",
            "**日间和日内差异都大于抽样误差。** 单日 AUC 在 {dmin:.3f}–{dmax:.3f} 之间（I² = {i2:.0%}）；按开盘后小时看，"
            "{best} 最强（{bv:.3f}），{worst} 最弱（{wv:.3f}）。"],
        "info": {"数据": "{days} 个交易日，每天 {per:,} 个等间隔时点（合成）", "标签": "时点之后中间价第一次变动的方向",
                 "特征": "{nf} 个盘口与订单流特征，按天标准化", "模型": "单特征、线性 6 特征、线性 + 非线性（逻辑回归）",
                 "检验": "按测试日 bootstrap；Cochran Q"},
        "decision": "下一阶段把工作从模型结构转到标签定义和低相关的新特征（第 6 节）。",
        "s1": "主要结果", "s1_lead": "唯一明显的提升来自从单特征到 6 个特征这一步；之后的非线性项可以忽略。",
        "t1_cols": ["模型", "特征", "AUC", "Δ AUC", "Bal. acc.", "对数损失", "单日 AUC"],
        "t1_title": "线性模型明显好于单特征，非线性项没有再带来提升", "t1_sub": "测试集 {N:,} 个样本；Δ 相对单特征",
        "t1_src": "方块标出每列最好的值；单日 AUC 是测试期 {n} 天的走势。",
        "wf_steps": ["单特征", "另外 5 个线性特征", "平方项和立方项", "最终模型"],
        "wf_title": "AUC 从 {auc1:.3f} 到 {aucnl:.3f}，几乎全部来自线性特征", "wf_sub": "AUC 分解，测试集；纵轴从 0.74 开始",
        "wf_src": "逐步加入特征后重新拟合，每一步的 AUC 差。",
        "roc_title": "三条 ROC 曲线几乎重合在两条上", "roc_sub": "真阳性率 vs 假阳性率，测试集",
        "cal_title": "预测概率校准良好", "cal_sub": "实际上涨频率 vs 平均预测概率，10 个分位数分箱",
        "cal_src": "校准斜率 {slope:.3f}，截距 {icpt}。",
        "ba_series": "balanced accuracy 上限", "ba_title": "balanced accuracy 到 0.75 需要 AUC {need:.3f}",
        "ba_sub": "二元正态等方差假设下的上限；圆点为实测",
        "ba_src_raw": r"$\mathrm{BA}_{\max} = \Phi(\Phi^{-1}(\mathrm{AUC})/\sqrt{2})$，见附录 A。",
        "warn_title": "不要用 balanced accuracy 比较模型",
        "warn": "balanced accuracy 依赖阈值，AUC 不依赖。实测 {ba:.3f} 和 AUC 对应的上限 {cap:.3f} 几乎相同，调阈值没有提升空间。"
                "比较模型以 AUC 和对数损失为准。",
        "note_title": "为什么按天切分",
        "note": "测试集只用训练期之后的日子。按行随机切分时，相邻几秒的样本会同时出现在训练集和测试集，分数会明显偏高。",
        "s2": "稳定性", "s2_lead": "线性模型在 {k} / {n} 个测试日都更好，但所有模型一起随日子和时段波动。",
        "day_name": "测试日", "line_title": "两条线同涨同跌：日间波动来自市场，不来自模型", "line_sub": "单日 AUC，测试期",
        "line_src": "合成数据里信号强度逐日变化；右侧标签为最后一天的值。",
        "gain_series": "非线性增益", "gain_title": "非线性项的逐日增益都在 {gmax:.4f} 以内",
        "gain_sub": "单日 AUC 差：线性 + 非线性 − 线性 6 特征", "gain_src": "{gpos} 天为正，{gneg} 天为负或为零。",
        "box_title": "三个模型的单日 AUC 分布", "box_sub": "箱体为四分位，粗线为中位数，点为每一天",
        "sc_series": "测试日", "sc_x": "单特征 AUC", "sc_y": "线性 6 特征 AUC", "sc_title": "每一天都在对角线上方",
        "sc_sub": "单日 AUC：单特征（横轴）vs 线性 6 特征（纵轴）",
        "hm_title": "{best} 信号最强，{worst} 最弱", "hm_sub": "线性 6 特征，按开盘后小时 × 测试日的 AUC",
        "hm_src": "每格约 500 个样本，单格噪声较大，看行的整体深浅。",
        "s3": "特征", "s3_lead": "盘口不平衡贡献了大部分排序能力；6 个特征里有两对高度相关，单独看系数会误判。",
        "corr_title": "两对特征高度相关：订单流的两个窗口 {c1:.2f}，盘口不平衡与微价格偏离 {c2:.2f}", "corr_sub": "Pearson 相关系数，测试集",
        "corr_name": "相关系数",
        "imp_series": "AUC 下降", "imp_title": "打乱盘口不平衡，AUC 下降 {imp0:.2f}；其余都在 {imp1:.2f} 以内",
        "imp_sub": "置换重要性：打乱一列后 AUC 的下降，重复 3 次取平均",
        "imp_src": "高相关的特征互相分担重要性，所以「订单流不平衡 30s」偏低。",
        "mult_title": "单个特征每天的排序能力都稳定；价差始终在 {smin:.2f}–{smax:.2f}，没有信息",
        "mult_sub": "单特征 AUC，测试期逐日；虚线为 0.5（无信息）",
        "db_a": "训练期", "db_b": "测试期", "db_title": "训练期到测试期最多变化 {fmax:.3f}", "db_sub": "单特征 AUC，按测试期排序",
        "s4": "数据", "s4_lead": "清洗后保留 {keep:.1%} 的时点；事件间隔是双峰分布，两类的预测概率重叠很多。",
        "cl_cols": ["步骤", "样本数", "占比", "处理"],
        "cl_rows": ["原始时点", "剔除：集合竞价时段", "剔除：交叉盘口（买一 ≥ 卖一）", "剔除：特征缺失", "进入建模"],
        "cl_removed": "已剔除", "cl_pass": "通过", "cl_title": "清洗后保留 {keep:.1%} 的时点", "cl_sub": "样本数，{days} 个交易日合计",
        "cl_src": "合成数据；剔除比例是示例值。",
        "gap_x": "秒（对数刻度）", "gap_title": "事件间隔取对数后是双峰，不是指数分布的单峰",
        "gap_sub": "{ngaps:,} 个间隔，48 个等比分箱，事件数", "gap_src": "合成数据：68% 来自簇内（对数正态），32% 来自簇间（指数）。",
        "ph_up": "实际上涨", "ph_down": "实际下跌", "ph_ref": "阈值 0.5", "ph_title": "两类的预测概率大面积重叠，这就是 AUC 的上限",
        "ph_sub": "线性 6 特征的预测概率，按实际方向分开，样本数",
        "def_label": "定义", "def_title": "口径",
        "defs": ["标签：时点之后中间价**第一次变动**的方向；之后没有变动的时点不进入样本。",
                 "切分：按天，前 {train} 天训练，后 {n} 天测试，避免相邻时点泄漏。",
                 "指标：AUC 衡量排序，对数损失衡量概率质量；balanced accuracy 只作参考。"],
        "s5": "检验与风险", "s5_lead": "四个假设里，非线性增益不成立，日间稳定性存疑；结论依赖标签定义。",
        "h_cols": ["编号", "假设", "统计量", "95% 区间", "结论"],
        "h_rows": ["线性 6 特征的 AUC 高于单特征", "非线性项让 AUC 至少多 0.005", "校准斜率在 [0.9, 1.1] 之内",
                   "单日 AUC 的差异只来自抽样误差"],
        "h_stat": ["ΔAUC {sd}", "ΔAUC {sg}", "斜率 {slope:.3f}", "Q {Q:.1f}，I² {i2:.0%}"],
        "yes": "成立", "no": "不成立", "open": "待定：需要更多天",
        "h_title": "非线性增益不成立，日间稳定性存疑", "h_sub": "H1、H2：按测试日 bootstrap 5,000 次；H4：Cochran Q",
        "h_src": "单日 AUC 的方差用 Hanley–McNeil 公式。",
        "risk_title": "结论成立的前提",
        "risks": ["合成数据的标签机制已知；真实数据要先确认标签的填充方向。", "测试期只有 {n} 天，H4 对天数敏感。",
                  "上限换算假设两类得分近似正态、方差相等。"],
        "s6": "下一步", "s6_lead": "把精力从模型结构转到标签和特征。", "act_title": "四项后续工作，按顺序做",
        "acts": [("在真实数据上确认标签填充方向：画 10 分钟的买一、卖一、中间价和标签", "（待定）", "第 1 周", "进行中"),
                 ("换成「下一次变动方向」和「30 秒三分类」两版标签，重跑本报告", "（待定）", "第 1 周", "未开始"),
                 ("加入与现有特征相关性低的新特征（全市场、跨品种）", "（待定）", "第 2 周", "未开始"),
                 ("测试期延长到 40 天，重新检验 H4；按开盘后小时分别报告", "（待定）", "第 3 周", "有风险")],
        "a1": "指标定义", "a1_lead_raw": "$s$ 是模型得分，$y \\in \\{0, 1\\}$ 是标签，$\\Phi$ 是标准正态分布函数。",
        "a2": "复现", "a2_lead": "所有数字都在 `src/reportkit/example.py` 里从固定种子的合成数据算出。",
        "cmd": "PYTHONPATH=src python -m reportkit example report.html",
        "demo": '''from reportkit import Report, charts

r = Report("结论写进标题", dek="回答什么问题、用什么数据", kicker=["量化研究"], status="review", date="2026-10-24")
r.kpis([{"label": "AUC", "value": 0.784, "fmt": ".3f", "delta": 0.024, "better": "up", "ref": "对比基线"}])
r.section("主要结果", lead="这一节的结论")
with r.row(height=250):
    r.chart(charts.line(days, {"模型 A": a, "模型 B": b}, y_format=".3f"), span=8, title="图标题写结论")
    r.chart(charts.bar(names, values, sort="desc"), span=4, title="……")
r.save("report.html")''',
        "log": ("v1.0", "2026-10-24", "首次发布"),
    },
    "en": {
        "features": FEATURES_EN, "models": ["Single feature", "Linear, 6 features", "Linear + nonlinear"],
        "title": "The linear model has peaked at {auc:.3f} AUC; nonlinear terms add only {gain:.4f}",
        "dek": "How three models built on the same six features perform out of sample, what balanced accuracy can and cannot "
               "tell us, and how stable the signal is across days and hours of the session. An example note for the report "
               "theme; all data are synthetic.",
        "kicker": ["Quant research", "Direction signal", "Model evaluation"], "doc_id": "Example · synthetic data",
        "date": "24 Oct 2026",
        "meta": {"Author": "(author)", "Data": "{days} trading days × {per:,} samples",
                 "Split": "First {train} days train, last {n} test", "Label": "Direction of the next mid-price move"},
        "k_auc": "AUC · linear, 6 features", "k_ba": "Balanced accuracy", "k_ll": "Log loss", "k_gain": "Nonlinear gain · AUC",
        "k_range": "Daily AUC range", "vs": "vs single feature", "k_cap": "Binormal ceiling {cap:.3f}",
        "k_gain_note": "Threshold +0.005; 95% CI [{glo}, {ghi}]", "k_range_note": "I² = {i2:.0%} over {n} test days",
        "points": [
            "**Six features beat one by {d:+.3f} AUC — and on every single day.** Day-clustered bootstrap 95% CI "
            "[{dlo:+.3f}, {dhi:+.3f}]; better on {k} of {n} test days.",
            "**Nonlinear terms add almost nothing.** Squares and cubes lift AUC by just {g:.4f}, well short of the 0.005 bar. "
            "Model structure is not the bottleneck.",
            "**Balanced accuracy is stuck near {ba:.2f} because of the AUC, not the threshold.** Under a binormal model an AUC of "
            "{auc:.3f} caps balanced accuracy at {cap:.3f}; reaching 0.75 would take an AUC of {need:.3f}.",
            "**Day-to-day and intraday swings both exceed sampling error.** Daily AUC ranges from {dmin:.3f} to {dmax:.3f} "
            "(I² = {i2:.0%}); by hour since the open, {best} is strongest ({bv:.3f}) and {worst} weakest ({wv:.3f})."],
        "info": {"Data": "{days} trading days, {per:,} evenly spaced timestamps a day (synthetic)",
                 "Label": "Direction of the first mid-price move after each timestamp",
                 "Features": "{nf} order-book and order-flow features, standardised by day",
                 "Models": "Single feature; linear on six features; linear plus nonlinear terms (logistic regression)",
                 "Tests": "Bootstrap over test days; Cochran's Q"},
        "decision": "Shift the next phase from model structure to the label definition and new, low-correlation features "
                    "(Section 6).",
        "s1": "Main results", "s1_lead": "The only material gain comes from going from one feature to six; the nonlinear terms "
                                          "added after that are negligible.",
        "t1_cols": ["Model", "Features", "AUC", "Δ AUC", "Bal. acc.", "Log loss", "Daily AUC"],
        "t1_title": "The linear model clearly beats the single feature; nonlinear terms add nothing further",
        "t1_sub": "Test set, {N:,} samples; Δ vs single feature",
        "t1_src": "Square marks the best value in each column; daily AUC shows the {n}-day test period.",
        "wf_steps": ["Single feature", "5 more linear features", "Squares and cubes", "Final model"],
        "wf_title": "AUC rises from {auc1:.3f} to {aucnl:.3f}, almost entirely through linear features",
        "wf_sub": "AUC attribution, test set; axis starts at 0.74",
        "wf_src": "Refit after adding each block of features; bars show the change in AUC.",
        "roc_title": "Three ROC curves collapse onto two", "roc_sub": "True vs false positive rate, test set",
        "cal_title": "Predicted probabilities are well calibrated",
        "cal_sub": "Observed up-move frequency vs mean predicted probability, 10 quantile bins",
        "cal_src": "Calibration slope {slope:.3f}, intercept {icpt}.",
        "ba_series": "Balanced accuracy ceiling", "ba_title": "A balanced accuracy of 0.75 would need an AUC of {need:.3f}",
        "ba_sub": "Ceiling under an equal-variance binormal model; dots are measured",
        "ba_src_raw": r"$\mathrm{BA}_{\max} = \Phi(\Phi^{-1}(\mathrm{AUC})/\sqrt{2})$; see Appendix A.",
        "warn_title": "Do not rank models on balanced accuracy",
        "warn": "Balanced accuracy depends on the threshold; AUC does not. The measured {ba:.3f} already sits at the {cap:.3f} "
                "ceiling implied by the AUC, so there is nothing left to gain from tuning the threshold. Compare models on AUC "
                "and log loss.",
        "note_title": "Why the split is by day",
        "note": "The test set uses only days after the training period. A random row split would put neighbouring seconds in "
                "both sets and inflate every score.",
        "s2": "Stability", "s2_lead": "The linear model wins on {k} of {n} test days, but all models rise and fall together "
                                      "with the day and the hour.",
        "day_name": "Test day", "line_title": "The two lines move together: daily swings come from the market, not the model",
        "line_sub": "Daily AUC, test period", "line_src": "Signal strength varies by day in the synthetic data; tags show the last day.",
        "gain_series": "Nonlinear gain", "gain_title": "The daily gain from nonlinear terms never exceeds {gmax:.4f}",
        "gain_sub": "Daily AUC difference: linear + nonlinear minus linear, 6 features",
        "gain_src": "Positive on {gpos} days, zero or negative on {gneg}.",
        "box_title": "Daily AUC by model", "box_sub": "Box spans the quartiles, thick line is the median, dots are days",
        "sc_series": "Test day", "sc_x": "Single-feature AUC", "sc_y": "Linear-model AUC",
        "sc_title": "Every day sits above the diagonal", "sc_sub": "Daily AUC: single feature (x) vs linear, 6 features (y)",
        "hm_title": "The signal is strongest at {best} and weakest at {worst}",
        "hm_sub": "Linear-model AUC by hour since the open × test day",
        "hm_src": "About 500 samples per cell; read the shade of each row, not single cells.",
        "s3": "Features", "s3_lead": "Book imbalance carries most of the ranking power; two pairs of features are highly "
                                     "correlated, so single coefficients mislead.",
        "corr_title": "Two pairs move together: the order-flow windows ({c1:.2f}) and book imbalance with micro-price ({c2:.2f})",
        "corr_sub": "Pearson correlation, test set", "corr_name": "Correlation",
        "imp_series": "AUC drop", "imp_title": "Shuffling book imbalance costs {imp0:.2f} AUC; no other feature costs more than {imp1:.2f}",
        "imp_sub": "Permutation importance: AUC drop after shuffling one column, mean of 3 repeats",
        "imp_src": "Correlated features share importance, which is why order-flow imbalance 30s looks small.",
        "mult_title": "Each feature ranks consistently day to day; spread stays at {smin:.2f}–{smax:.2f}, i.e. no information",
        "mult_sub": "Single-feature AUC by test day; dashed line at 0.5 (no information)",
        "db_a": "Training", "db_b": "Test", "db_title": "No feature moves by more than {fmax:.3f} from training to test",
        "db_sub": "Single-feature AUC, sorted by test period",
        "s4": "Data", "s4_lead": "Cleaning keeps {keep:.1%} of timestamps; gaps between events are bimodal, and the two classes' "
                                 "predicted probabilities overlap heavily.",
        "cl_cols": ["Step", "Samples", "Share", "Treatment"],
        "cl_rows": ["Raw timestamps", "Removed: call auctions", "Removed: crossed book (bid ≥ ask)", "Removed: missing features",
                    "Modelled"],
        "cl_removed": "Removed", "cl_pass": "Passed", "cl_title": "Cleaning keeps {keep:.1%} of timestamps",
        "cl_sub": "Sample counts, {days} trading days", "cl_src": "Synthetic data; removal shares are illustrative.",
        "gap_x": "Seconds (log scale)", "gap_title": "Log gaps between events are bimodal, not the single hump of an exponential",
        "gap_sub": "{ngaps:,} gaps, 48 log-spaced bins, event counts",
        "gap_src": "Synthetic: 68% within bursts (log-normal), 32% between bursts (exponential).",
        "ph_up": "Actual up", "ph_down": "Actual down", "ph_ref": "Threshold 0.5",
        "ph_title": "The two classes overlap heavily, and that is what caps the AUC",
        "ph_sub": "Predicted probability of the linear model by actual direction, counts",
        "def_label": "Definitions", "def_title": "How the numbers are defined",
        "defs": ["**Label**: direction of the **first** mid-price move after the timestamp; timestamps with no later move are dropped.",
                 "**Split**: by day — first {train} days train, last {n} test — so neighbouring timestamps cannot leak.",
                 "**Metrics**: AUC measures ranking and log loss measures probability quality; balanced accuracy is for "
                 "reference only."],
        "s5": "Tests and risks", "s5_lead": "Of four hypotheses, the nonlinear gain is rejected and day-to-day stability is "
                                            "unresolved; the conclusions depend on the label definition.",
        "h_cols": ["ID", "Hypothesis", "Statistic", "95% CI", "Result"],
        "h_rows": ["Linear (6 features) AUC exceeds single feature", "Nonlinear terms add at least 0.005 AUC",
                   "Calibration slope within [0.9, 1.1]", "Daily AUC differences are sampling error only"],
        "h_stat": ["ΔAUC {sd}", "ΔAUC {sg}", "Slope {slope:.3f}", "Q {Q:.1f}, I² {i2:.0%}"],
        "yes": "Holds", "no": "Rejected", "open": "Open: needs more days",
        "h_title": "Nonlinear gain rejected; day-to-day stability unresolved",
        "h_sub": "H1, H2: 5,000 bootstrap draws over test days; H4: Cochran's Q",
        "h_src": "Daily AUC variance from the Hanley–McNeil formula.",
        "risk_title": "What the conclusions rest on",
        "risks": ["The label mechanism is known in synthetic data; on real data, confirm the label fill direction first.",
                  "The test period is only {n} days long, and H4 is sensitive to that.",
                  "The ceiling assumes roughly normal, equal-variance scores in both classes."],
        "s6": "Next steps", "s6_lead": "Move the effort from model structure to labels and features.",
        "act_title": "Four follow-ups, in order",
        "acts": [("Confirm the label fill direction on real data: plot 10 minutes of bid, ask, mid and label", "(TBD)", "Week 1",
                  "In progress"),
                 ("Rebuild with two new labels — next-move direction and a 30-second three-class label — and rerun this note",
                  "(TBD)", "Week 1", "Not started"),
                 ("Add features with low correlation to the current set (market-wide, cross-asset)", "(TBD)", "Week 2",
                  "Not started"),
                 ("Extend the test period to 40 days, retest H4 and report by hour since the open", "(TBD)", "Week 3", "At risk")],
        "a1": "Metric definitions",
        "a1_lead_raw": "$s$ is the model score, $y \\in \\{0, 1\\}$ the label and $\\Phi$ the standard normal CDF.",
        "a2": "Reproduction", "a2_lead": "Every number is computed in `src/reportkit/example.py` from synthetic data with a fixed seed.",
        "cmd": "PYTHONPATH=src python -m reportkit example report-en.html --lang en",
        "demo": '''from reportkit import Report, charts

r = Report("Put the conclusion in the headline", dek="What question, which data", kicker=["Quant research"],
           status="review", date="24 Oct 2026", lang="en")
r.kpis([{"label": "AUC", "value": 0.784, "fmt": ".3f", "delta": 0.024, "better": "up", "ref": "vs baseline"}])
r.section("Main results", lead="The section's conclusion in one sentence.")
with r.row(height=250):
    r.chart(charts.line(days, {"Model A": a, "Model B": b}, y_format=".3f"), span=8, title="Chart title states the finding")
    r.chart(charts.bar(names, values, sort="desc", lang="en"), span=4, title="...")
r.save("report.html")''',
        "log": ("v1.0", "24 Oct 2026", "First release"),
    },
}


def build(path, generated: str | None = None, lang: str = "zh", translations: dict | None = None) -> Report:
    """Write the example note. lang: "zh" or "en"; translations: {lang: href} for the masthead switch."""
    T = TEXT["en" if str(lang).startswith("en") else "zh"]
    en = T is TEXT["en"]
    E = evaluate()
    m = E["models"]
    one, lin, nl = m["单特征"], m["线性 6 特征"], m["线性 + 非线性"]
    MOD = dict(zip(m, T["models"]))                      # internal key -> display name
    FT = T["features"]
    d_lin, d_nl = E["d_lin"], E["d_nl"]
    n_days = DAYS - TRAIN_DAYS
    days = [f"D{d + 1}" for d in range(TRAIN_DAYS, DAYS)]
    need = auc_for_ba(0.75)
    sgn = lambda v, f=".4f": fmt_num(v, "+" + f)          # noqa: E731  signed, typographic minus
    hourly_mean = E["hourly"].mean(1)
    best_h, worst_h = int(np.argmax(hourly_mean)), int(np.argmin(hourly_mean))
    gain = nl["daily"] - lin["daily"]
    raw = DAYS * PER_DAY
    drop = [int(raw * f) for f in (0.031, 0.004, 0.012)]
    total0 = raw + sum(drop)
    hc, he = np.histogram(np.log10(E["gaps"]), bins=48)
    mid = len(hc) // 2
    peaks = [10 ** ((he[i] + he[i + 1]) / 2) for i in (int(np.argmax(hc[:mid])), mid + int(np.argmax(hc[mid:])))]
    spread = E["feat_daily"][FEATURES[4]]
    c = dict(auc=lin["auc"], auc1=one["auc"], aucnl=nl["auc"], gain=nl["auc"] - lin["auc"], ba=lin["ba"],
             cap=max_ba(lin["auc"]), need=need, d=d_lin[0], dlo=d_lin[1], dhi=d_lin[2], g=d_nl[0], glo=sgn(d_nl[1]),
             ghi=sgn(d_nl[2]), sd=sgn(d_lin[0]), sg=sgn(d_nl[0]), k=E["days_better"], n=n_days, dmin=lin["daily"].min(),
             dmax=lin["daily"].max(), i2=E["I2"], Q=E["Q"], best=HOURS[best_h], worst=HOURS[worst_h], bv=hourly_mean[best_h],
             wv=hourly_mean[worst_h], N=len(E["y"]), days=DAYS, per=PER_DAY, train=TRAIN_DAYS, nf=len(FEATURES),
             slope=E["cal_slope"], icpt=fmt_num(E["cal_intercept"], "+.3f"), gmax=float(np.abs(gain).max()),
             gpos=int(np.sum(gain > 0)), gneg=int(np.sum(gain <= 0)), c1=E["corr"][1, 2], c2=E["corr"][0, 3],
             imp0=E["importance"][0], imp1=max(E["importance"][1:]), smin=min(spread), smax=max(spread),
             fmax=max(abs(a - b) for a, b in zip(E["feat_train"], E["feat_test"])), keep=raw / total0, ngaps=len(E["gaps"]))

    def t(key, i=None):
        v = T[key] if i is None else T[key][i]
        return v if key.endswith("_raw") else v.format(**c)

    r = Report(t("title"), dek=t("dek"), kicker=T["kicker"], doc_id=T["doc_id"], status="review", date=T["date"],
               version="v1.0", meta={k: v.format(**c) for k, v in T["meta"].items()}, generated=generated,
               lang="en" if en else "zh-CN", translations=translations)
    r.kpis([
        {"label": t("k_auc"), "value": lin["auc"], "fmt": ".3f", "delta": lin["auc"] - one["auc"], "delta_fmt": "+.3f",
         "better": "up", "ref": t("vs"), "spark": lin["daily"]},
        {"label": t("k_ba"), "value": lin["ba"], "fmt": ".3f", "delta": lin["ba"] - one["ba"], "delta_fmt": "+.3f",
         "better": "up", "ref": t("vs"), "note": t("k_cap")},
        {"label": t("k_ll"), "value": lin["ll"], "fmt": ".4f", "delta": lin["ll"] - one["ll"], "delta_fmt": "+.4f",
         "better": "down", "ref": t("vs")},
        {"label": t("k_gain"), "value": d_nl[0], "fmt": "+.4f", "note": t("k_gain_note")},
        {"label": t("k_range"), "value": f"{c['dmin']:.3f}–{c['dmax']:.3f}", "note": t("k_range_note")},
    ])
    with r.row():
        r.takeaways([p.format(**c) for p in T["points"]], span=8)
        r.info({k: v.format(**c) for k, v in T["info"].items()}, decision=t("decision"), span=4)

    # 1 ---------------------------------------------------------------------------------------------
    r.section(t("s1"), lead=t("s1_lead"))
    cols = T["t1_cols"]
    with r.row(height=250):
        r.table([dict(zip(cols, (MOD[k], n_f, v["auc"], v["auc"] - one["auc"], v["ba"], v["ll"], [round(x, 4) for x in v["daily"]])))
                 for (k, v), n_f in zip(m.items(), (1, 6, 18))],
                span=7, title=t("t1_title"), subtitle=t("t1_sub"), source=t("t1_src"),
                fmt={cols[2]: ".4f", cols[3]: "+.4f", cols[4]: ".4f", cols[5]: ".4f"},
                best={cols[2]: "max", cols[4]: "max", cols[5]: "min"}, delta={cols[3]: "up"}, spark=[cols[6]], highlight=[1])
        w = T["wf_steps"]
        r.chart(charts.waterfall([(w[0], round(one["auc"], 4), "total"), (w[1], round(lin["auc"] - one["auc"], 4)),
                                  (w[2], round(nl["auc"] - lin["auc"], 4)), (w[3], round(nl["auc"], 4), "total")],
                                 value_format=".4f", y_min=0.74),
                span=5, title=t("wf_title"), subtitle=t("wf_sub"), source=t("wf_src"))
    with r.row(height=230):
        r.chart(charts.roc(E["y"], {MOD[k]: v["p"] for k, v in m.items()}, lang=lang), span=4, title=t("roc_title"), subtitle=t("roc_sub"))
        r.chart(charts.calibration(E["y"], {MOD["线性 6 特征"]: lin["p"]}, lang=lang), span=4,
                title=t("cal_title"), subtitle=t("cal_sub"), source=t("cal_src"))
        grid = np.round(np.linspace(0.5, 0.95, 46), 3)
        r.chart(charts.line(grid.tolist(), [{"name": t("ba_series"), "values": [round(max_ba(a), 4) for a in grid], "color": "s7"}],
                            x_label="AUC", x_name="AUC", x_format=".2f", y_format=".2f", tags=False, refs=[{"y": 0.75, "label": "0.75"}],
                            labels=[{"x": round(one["auc"], 3), "y": round(one["ba"], 3), "text": MOD["单特征"], "dx": -9, "dy": -8, "anchor": "end"},
                                    {"x": round(lin["auc"], 3), "y": round(lin["ba"], 3), "text": MOD["线性 6 特征"], "dx": 9, "dy": 15}]),
                span=4, title=t("ba_title"), subtitle=t("ba_sub"), source=t("ba_src_raw"))
    with r.row():
        r.callout(t("warn"), title=t("warn_title"), kind="warn", span=6)
        r.callout(t("note"), title=t("note_title"), kind="note", span=6)

    # 2 ---------------------------------------------------------------------------------------------
    r.section(t("s2"), lead=t("s2_lead"))
    with r.row(height=250):
        r.chart(charts.line(days, {MOD["线性 6 特征"]: lin["daily"].round(4), MOD["单特征"]: one["daily"].round(4)},
                            x_name=t("day_name"), y_format=".3f"),
                span=8, title=t("line_title"), subtitle=t("line_sub"), source=t("line_src"))
        r.chart(charts.bar(days, {t("gain_series"): gain.round(5)}, tone="sign", value_format="+.4f", labels=False),
                span=4, title=t("gain_title"), subtitle=t("gain_sub"), source=t("gain_src"))
    with r.row(height=240):
        r.chart(charts.box({MOD[k]: v["daily"] for k, v in m.items()}, value_format=".3f"), span=4,
                title=t("box_title"), subtitle=t("box_sub"))
        r.chart(charts.scatter({t("sc_series"): (one["daily"], lin["daily"], days)}, x_label=t("sc_x"), y_label=t("sc_y"),
                               x_format=".2f", y_format=".2f", diagonal="y = x", fit=True), span=4,
                title=t("sc_title"), subtitle=t("sc_sub"))
        r.chart(charts.heatmap(E["hourly"].round(3), HOURS, days, scale="sequential", value_format=".3f", value_name="AUC",
                               scale_label="AUC"), span=4, title=t("hm_title"), subtitle=t("hm_sub"), source=t("hm_src"))

    # 3 ---------------------------------------------------------------------------------------------
    r.section(t("s3"), lead=t("s3_lead"))
    with r.row(height=260):
        r.chart(charts.heatmap(np.round(E["corr"], 2), FT, FT, domain=(-1, 1), value_name=t("corr_name")), span=6,
                title=t("corr_title"), subtitle=t("corr_sub"))
        r.chart(charts.bar(FT, {t("imp_series"): [round(v, 4) for v in E["importance"]]}, sort="desc", value_format=".4f",
                           highlight=FT[0], horizontal=True), span=6,
                title=t("imp_title"), subtitle=t("imp_sub"), source=t("imp_src"))
    with r.row(height=250):
        r.chart(charts.multiples(days, {FT[j]: [round(v, 4) for v in E["feat_daily"][f]] for j, f in enumerate(FEATURES)},
                                 y_format=".2f", refs=[{"y": 0.5}], cols=3), span=8, title=t("mult_title"), subtitle=t("mult_sub"))
        r.chart(charts.dumbbell(FT, {"name": t("db_a"), "values": [round(v, 4) for v in E["feat_train"]]},
                                {"name": t("db_b"), "values": [round(v, 4) for v in E["feat_test"]]}, value_format=".3f", sort="desc"),
                span=4, title=t("db_title"), subtitle=t("db_sub"))

    # 4 ---------------------------------------------------------------------------------------------
    r.section(t("s4"), lead=t("s4_lead"))
    cc, cr = T["cl_cols"], T["cl_rows"]
    removed, passed = badge(t("cl_removed"), "info"), badge(t("cl_pass"), "good")
    with r.row(height=230):
        r.table([dict(zip(cc, row)) for row in [
            (cr[0], total0, 1.0, None),
            (cr[1], -drop[0], -drop[0] / total0, removed), (cr[2], -drop[1], -drop[1] / total0, removed),
            (cr[3], -drop[2], -drop[2] / total0, removed), (cr[4], raw, raw / total0, passed)]],
                span=5, title=t("cl_title"), subtitle=t("cl_sub"), source=t("cl_src"),
                fmt={cc[1]: ",d", cc[2]: ".1%"}, bars=[cc[2]], total=[4])
        r.chart(charts.hist(E["gaps"].tolist(), bins=48, log_x=True, x_label=t("gap_x"), lang=lang,
                            refs=[{"x": peaks[0], "label": f"{peaks[0]:.2f} s"}, {"x": peaks[1], "label": f"{peaks[1]:.1f} s"}]),
                span=7, title=t("gap_title"), subtitle=t("gap_sub"), source=t("gap_src"))
    with r.row(height=220):
        p, yt = lin["p"], E["y"]
        r.chart(charts.hist(series={t("ph_up"): p[yt == 1].tolist(), t("ph_down"): p[yt == 0].tolist()}, bins=40, range=(0, 1),
                            x_format=".1f", refs=[{"x": 0.5, "label": t("ph_ref")}]),
                span=7, title=t("ph_title"), subtitle=t("ph_sub"))
        r.text([d.format(**c) for d in T["defs"]], span=5, title=t("def_title"), label=t("def_label"))

    # 5 ---------------------------------------------------------------------------------------------
    r.section(t("s5"), lead=t("s5_lead"))
    hc_ = T["h_cols"]
    held = [d_lin[1] > 0, d_nl[2] >= 0.005, 0.9 <= E["cal_slope"] <= 1.1, E["I2"] < 0.5]
    ci = [f"[{sgn(d_lin[1])}, {sgn(d_lin[2])}]", f"[{sgn(d_nl[1])}, {sgn(d_nl[2])}]", "—", "—"]
    verdict = [badge(t("yes"), "good") if h else badge(t("no"), "bad") for h in held[:3]]
    verdict.append(badge(t("yes"), "good") if held[3] else badge(t("open"), "warn"))
    with r.row():
        r.table([dict(zip(hc_, (f"H{i + 1}", T["h_rows"][i], t("h_stat", i), ci[i], verdict[i]))) for i in range(4)],
                span=8, title=t("h_title"), subtitle=t("h_sub"), source=t("h_src"))
        r.callout([x.format(**c) for x in T["risks"]], title=t("risk_title"), kind="risk", span=4)

    # 6 ---------------------------------------------------------------------------------------------
    r.section(t("s6"), lead=t("s6_lead"))
    r.actions([dict(zip(("item", "owner", "due", "status"), a)) for a in T["acts"]], title=t("act_title"))

    # appendix --------------------------------------------------------------------------------------
    r.appendix(t("a1"), lead=t("a1_lead_raw"))
    r.math(r"\mathrm{AUC} = \Pr\left(s_i > s_j \mid y_i = 1,\ y_j = 0\right)")
    r.math(r"\mathrm{BA}(t) = \tfrac{1}{2}\left[\Pr(s > t \mid y = 1) + \Pr(s \le t \mid y = 0)\right]")
    r.math(r"\mathrm{BA}_{\max} = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{AUC})}{\sqrt{2}}\right), \qquad \mathrm{AUC} = \Phi\!\left(\sqrt{2}\,\Phi^{-1}(\mathrm{BA}_{\max})\right)")
    r.math(r"\mathrm{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\right]")
    r.appendix(t("a2"), lead=t("a2_lead"))
    r.code(T["cmd"], lang="bash")
    r.code(T["demo"], lang="python")
    r.changelog([T["log"]])
    r.save(path)
    return r
