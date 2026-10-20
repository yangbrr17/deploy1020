# 到达分布检验：要什么数据、检验什么、怎么判

2026-10 整理。[`event-arrivals.md`](event-arrivals.md) 讲了「下一笔什么时候来」有哪些模型；本页把它变成**可以直接执行的检验方案**：每个假设写清楚原假设是哪个分布、需要哪些数据字段、用什么统计量、怎么判定、文献里预期是什么结果。代码在 [`src/arrivals/`](../../src/arrivals/__init__.py)，已在合成数据上跑通（第 13 节有输出），拿到真实数据后一条命令即可出结果表。

## 结论

1. **要检验的对象有三条事件流**：E1 成交（同一时间戳、同一方向的多笔成交合并为一次主动订单）、E2 中间价变动、E3 最优档任何变化。有 L2 逐笔数据时再加 E4（各档的新增 / 撤单 / 成交）。
2. **假设按从简单到复杂排列，前一个被拒绝，才有理由用后一个**：H1 齐次泊松 → H2 去掉日内季节后的非齐次泊松 → H3 间隔的边际分布（指数 / Weibull / 广义伽马……）→ H4 ACD → H5 Hawkes → H6 队列状态依赖 → H7 中间价变动时间服从 Cont–de Larrard 首达分布 → H8 方向概率 $\phi(q^b,q^a)$ 校准 → H9 参数跨日稳定。
3. **一天有 $10^4$–$10^6$ 个事件，任何真实偏离的 p 值都接近 0。** 所以判定用「p 值显著 **且** 效应量超过实务阈值」（第 2 节），全表再用 Benjamini–Hochberg 控制错误发现率，汇总时看「被拒绝的股票日占比」和效应量中位数，而不是单个 p 值。
4. **按文献，预期是**：H1、H2 被拒绝（过度离散、聚集）；H3 里 Weibull 形状参数小于 1 或广义伽马胜出；H4 的 ACD 去掉大部分自相关，但残差仍略过度离散；H5 的 Hawkes 显著好于泊松，分支比在 0.6–1 之间且有争议；H6 被拒绝（强度依赖队列长度）；H8 的方向概率有区分度。H7 在真实数据上大概率被拒绝，因为常数强度的排队模型不含季节和聚集，这时它只作为基线。
5. **明天的执行顺序**（第 14 节）：按第 1 节导出数据 → 先跑合成数据 dry run → 1 只股票 1 天 → 全部样本 → 看汇总表，按第 13.3 节的决策表决定 hazard 模型里加哪些特征。

## 1. 数据：需要什么

### 1.1 表结构

每张表按「一只股票、一个交易日」切分，时间戳统一为 **int64 纳秒**（交易所时间优先；同一张表内必须一致）。多只股票、多天可以放在一个文件里，加 `symbol` 列，日期由时间戳换算。

| 表 | 字段 | 类型 | 含义 | 用于 |
|---|---|---|---|---|
| trades（必需） | `ts` | int64 | 成交时间（纳秒） | 全部 |
| | `price`, `size` | float | 成交价、成交量 | H6–H7 流量估计 |
| | `side` | int8 | +1 买方主动，−1 卖方主动，0 未知（无主动方标记时用 Lee–Ready 推断） | H7 流量、E1 合并 |
| quotes（必需） | `ts` | int64 | 最优档变化时间，**每次变化一行** | E2、E3、H6–H8 |
| | `bid_px`, `bid_sz`, `ask_px`, `ask_sz` | float | 买一卖一的价和量 | 同上 |
| level2（可选） | `ts`, `book`（+1 买 / −1 卖）, `event`（add / cancel / trade）, `size`, `q_after` | | 逐笔事件及事件后的队列长度 | H6 的精确版、E4 |
| 交易时段 | 开盘、收盘时间，时区 | | 只保留连续竞价 | 全部 |

**最低要求只有 trades 和 quotes（L1）。** H1–H5 只需要 trades；H6–H8 需要 quotes 里的最优档量。没有 L2 时，H6–H7 用 L1 相邻快照的差分近似新增和撤单（[`schema.l1_queue_flows`](../../src/arrivals/schema.py)），近似误差在第 9 节说明。

### 1.2 样本范围

| 维度 | 建议 | 原因 |
|---|---|---|
| 股票 | 6–10 只：大 tick（价差几乎总是 1 档）和小 tick 各半；流动性高、中各有 | 排队模型（H6–H8）只对大 tick 股票有意义；结论要看是否随流动性变化 |
| 天数 | 20 个交易日，剔除半天交易日和重大事件日 | H9 的跨日稳定性至少要 10 天；季节曲线要用「其他天」估计 |
| 时段 | 只用连续竞价；开盘后、收盘前各 5 分钟可以先保留，单独看一次 | 集合竞价不是连续到达过程 |

### 1.3 清洗规则（顺序执行）

1. 只保留连续竞价时段内的行。
2. 删掉交叉或锁定的盘口（$a\le b$）、量为非正的行，以及不改变任何字段的重复快照。
3. **合并成交**：同一时间戳、同一方向的多笔成交合并为一次主动订单（一张大单扫多档会打印多笔成交），记录总量、VWAP 和笔数。
4. **处理并列时间戳**：时间戳精度为 $r$（如 1 毫秒）时，把每个事件时间加上均匀抖动 $t_i \leftarrow t_i + U_i,\ U_i\sim\mathrm{U}[0,r)$ 后重新排序。0 间隔会让所有连续时间检验失效（指数分布在 0 处没有点质量）。纳秒精度的数据令 $r=0$，残留的完全并列直接丢掉。
5. 间隔只在同一天内计算，不跨隔夜；每天开盘后的第一个间隔不用。

### 1.4 四条事件流

| 名称 | 定义 | 典型一天的事件数（活跃股票） |
|---|---|---|
| E1 成交 | 清洗规则 3 之后的主动订单时间 $t^{(1)}_i$ | $10^4$–$10^5$ |
| E2 中间价变动 | $m_t=(b_t+a_t)/2$ 发生变化的时刻；同时记录方向 $\mathrm{sign}(\Delta m)$ 和变动后的 $(q^b,q^a)$ | 大 tick：$10^2$–$10^3$；小 tick：$10^4$ 以上 |
| E3 最优档更新 | 买一卖一的价或量任何一项变化 | $10^5$–$10^6$ |
| E4 逐档事件（L2） | 各档 add / cancel / trade | $10^6$ 以上 |

## 2. 判定规则：p 值加效应量

**为什么不能只看 p 值**：样本量 $n=10^5$ 时，变异系数 $\mathrm{CV}=1.02$ 也会被色散检验以 $p<10^{-6}$ 拒绝，但这个偏离对建模没有意义。所以每个检验同时输出 p 值和一个效应量，只有两者都超过阈值才算「拒绝」：

$$
\text{verdict}=\begin{cases}
\text{reject} & p<\alpha \ \text{且}\ \lvert\text{effect}-\text{null}\rvert>\delta\\
\text{reject (small effect)} & p<\alpha\ \text{但效应量小}\\
\text{not rejected} & p\ge\alpha
\end{cases}
\qquad \alpha=0.01
$$

| 效应量 | 零假设下的值 | 默认阈值 $\delta$ | 对应配置项 |
|---|---|---|---|
| 间隔的变异系数 CV | 1 | 0.10 | `cv_tol` |
| KS 距离 $D$ | 0 | 0.02 | `ks_tol` |
| 一阶自相关 $\rho_1$ | 0 | 0.05 | `rho_tol` |
| Fano 因子 $F$ | 1 | 0.20 | `fano_tol` |
| 各队列长度分箱的强度最大 / 最小比 | 1 | 1.5 | `rate_ratio_tol` |
| 校准斜率 | 1 | 0.20 | `slope_tol` |

阈值是起点，放在 [`report.Config`](../../src/arrivals/report.py) 里，可以按业务需要改。

**多重检验**：全表（股票 × 天 × 检验）用 Benjamini–Hochberg。把 $m$ 个 p 值从小到大排成 $p_{(1)}\le\dots\le p_{(m)}$，取

$$
k=\max\Big\{i:\ p_{(i)}\le \frac{i}{m}\,q\Big\},\qquad \text{拒绝 } p_{(1)},\dots,p_{(k)},\qquad q=0.05 .
$$

结果表里的 `reject_fdr` 列就是它。

**参数是估计出来的**：检验「是不是指数分布」时，速率 $\lambda$ 是从同一份数据估计的，教科书 KS 的 p 值会偏保守（Lilliefors 问题）。代码用参数 bootstrap：KS 统计量在零假设下的分布与 $\lambda$ 无关，所以模拟同样长度的 Exp(1) 样本、重新估计速率、重新算 $D$，取经验分位数。

**逐股逐日检验，再汇总**：不要把多天拼成一个序列（会混入隔夜间隔和日间水平差异）。汇总表报告每个检验「被拒绝的股票日占比」和效应量中位数。

## 3. 检验清单

| 编号 | 对象 | 原假设 $H_0$（具体分布） | 备择 | 检验与统计量 | 需要的数据 |
|---|---|---|---|---|---|
| H1 | E1、E2、E3 的间隔 | iid $\mathrm{Exp}(\lambda)$，即齐次泊松 | 过度离散、聚集 | 色散检验、KS（bootstrap）、Anderson–Darling、Ljung–Box、计数的离散指数 | trades；quotes |
| H2 | E1 | 强度为 $\lambda(t)=\lambda_0\,s(\text{时刻})$ 的非齐次泊松；重标度后 iid Exp(1) | 同上 | 同 H1，作用在重标度残差上；业务时间里的 Fano 因子 | trades，多天 |
| H3 | E1 季节调整后的间隔 | 指数 ⊂ Weibull ⊂ 广义伽马；伽马、对数正态、Burr XII | — | 嵌套用似然比，非嵌套用 Vuong；BIC 排序 | trades |
| H4 | E1 季节调整后的间隔 | WACD(1,1) 正确：广义残差 iid Exp(1) | 残差仍有结构 | 似然比（EACD vs WACD），残差上的 H1 四件套 | trades |
| H5 | E1（业务时间） | 指数核 Hawkes 正确：补偿量增量 iid Exp(1) | 核形状不对、多尺度 | 似然比（泊松 vs Hawkes，边界修正）、残差四件套、分支比 | trades，多天 |
| H6 | 最优档新增 / 撤单流 | 强度与队列长度无关（Cont–de Larrard 的假设） | queue-reactive：强度依赖队列长度 | 泊松过程似然比，$\chi^2_{B-1}$ | quotes（L1）或 level2 |
| H7 | E2 的等待时间 | Cont–de Larrard 首达分布 $S_{q^b}(t)\,S_{q^a}(t)$ | 季节、聚集使尾部更重 | 概率积分变换（PIT）的 KS；Hill 尾指数（平衡时理论值为 1） | quotes + trades |
| H8 | E2 下一次方向 | $\Pr(\text{涨}\mid q^b,q^a)=\phi(q^b,q^a)$ | 校准偏差 | 逻辑回归校准：截距 0、斜率 1 的 Wald 检验；Brier 技能分 | quotes |
| H9 | H4、H5 的关键参数 | 各天参数相等 | 跨日异质 | Cochran $Q$、$I^2$ | 多天结果 |

## 4. H1：齐次泊松

**原假设**：间隔 $X_i=t_i-t_{i-1}$ 独立同分布于 $\mathrm{Exp}(\lambda)$，即 $\Pr(X>x)=e^{-\lambda x}$。等价地，任一长度 $\Delta$ 的窗口内计数 $N\sim\mathrm{Poisson}(\lambda\Delta)$，不重叠窗口相互独立。

**四个检验**（先把间隔归一化为单位均值 $e_i=X_i/\bar X$）：

1. **色散检验**（Engle、Russell 1998）。Exp(1) 下 $\mathrm{Var}(e)=1$，四阶中心矩为 9，所以 $\mathrm{Var}\big[(e-1)^2\big]=9-1=8$：

$$
z=\sqrt{\frac{n}{8}}\,\big(\hat\sigma_e^2-1\big)\ \overset{a}{\sim}\ N(0,1),\qquad \text{效应量 } \mathrm{CV}=\hat\sigma_X/\bar X .
$$

   $z>0$ 表示过度离散：短间隔和长间隔都比指数分布多。

2. **KS 检验**（速率估计后用 bootstrap 求 p 值）：

$$
D=\sup_x\big\lvert \hat F_n(x)-\big(1-e^{-x/\bar X}\big)\big\rvert .
$$

3. **Anderson–Darling**：对尾部更敏感，$A^2=-n-\frac{1}{n}\sum_{i=1}^n(2i-1)\big[\ln F(x_{(i)})+\ln\big(1-F(x_{(n+1-i)})\big)\big]$，临界值查估计了尺度参数的指数分布表。
4. **独立性（Ljung–Box）**：

$$
Q=n(n+2)\sum_{h=1}^{H}\frac{\hat\rho_h^2}{n-h}\ \sim\ \chi^2_H,\qquad H=20,\qquad \text{效应量 } \hat\rho_1 .
$$

5. **计数的离散指数**：把 $[0,T]$ 切成 $k$ 个长度 $\Delta$ 的窗口，计数 $N_1,\dots,N_k$：

$$
I=\frac{(k-1)\,s_N^2}{\bar N}\ \sim\ \chi^2_{k-1},\qquad F(\Delta)=\frac{s_N^2}{\bar N}\ \ (\text{Fano 因子}),
$$

   取 $\Delta=1,10,60$ 秒。泊松下 $F(\Delta)\equiv1$；聚集或长记忆使 $F$ 随 $\Delta$ 上升。日内季节本身也会让长窗口的 $F$ 上升，所以 H2 在业务时间里再算一次。

**判定**：任一检验「reject」即拒绝 H1。**预期**：四个检验在 E1、E3 上都拒绝，CV 明显大于 1，$\hat\rho_1$ 明显为正（Engle、Russell 1998 及之后的大量实证）。

## 5. H2：去掉日内季节后的非齐次泊松

**原假设**：E1 是强度 $\lambda(t)$ 只依赖于时刻的非齐次泊松过程。估计日内强度曲线：把交易时段分成 $B=26$ 个分箱（6.5 小时时段即每 15 分钟一箱），**用其他交易日**的平均计数：

$$
\hat\lambda(b)=\frac{1}{D-1}\sum_{d'\ne d}\frac{N_{d'}(b)}{w},\qquad \hat\Lambda(t)=\int_0^t\hat\lambda(u)\,du\ \ (\text{分段线性}),
$$

$w$ 是分箱宽度。由时间重标度定理（Brown 等 2002），零假设下

$$
e_i=\hat\Lambda(t_i)-\hat\Lambda(t_{i-1})\ \overset{\text{iid}}{\sim}\ \mathrm{Exp}(1).
$$

各天活跃度水平不同，只假设曲线**形状**稳定，所以把当天的 $e_i$ 再除以均值。对 $e_i$ 做 H1 的四件套；再把事件时间映射到业务时间 $u_i=\hat\Lambda(t_i)\,T/\hat\Lambda(T)$，计算业务时间里的 Fano 因子。

**判定和预期**：如果 H2 不被拒绝，日内季节就是全部故事，hazard 模型只加季节因子即可。**预期是 H2 仍被拒绝**：季节调整后 CV 下降但仍明显大于 1，业务时间里的 Fano 因子仍随窗口变长而上升，说明有季节之外的聚集，需要 H4 或 H5。只有一天数据时代码会用当天自己估计季节曲线，并在结果里标注 `IN-SAMPLE`；这种情况下的检验偏乐观。

## 6. H3：间隔的边际分布属于哪一族

把季节调整后的间隔归一化为单位均值，用最大似然拟合下面几族（尺度参数都自由）：

| 分布 | 密度 $f(x)$，$x>0$ | 参数个数 | hazard 形状 |
|---|---|---|---|
| 指数 | $\frac{1}{s}e^{-x/s}$ | 1 | 常数 |
| Weibull | $\frac{k}{s}\big(\frac{x}{s}\big)^{k-1}e^{-(x/s)^k}$ | 2 | $k<1$ 递减，$k>1$ 递增 |
| 伽马 | $\frac{x^{a-1}e^{-x/s}}{\Gamma(a)s^a}$ | 2 | 单调 |
| 对数正态 | $\frac{1}{x\sigma\sqrt{2\pi}}\exp\!\big(-\frac{(\ln x-m)^2}{2\sigma^2}\big)$ | 2 | 先升后降 |
| 广义伽马（Stacy 1962） | $\frac{\lvert c\rvert\,x^{ca-1}e^{-(x/s)^c}}{\Gamma(a)\,s^{ca}}$ | 3 | 单调或驼峰 |
| Burr XII | $F(x)=1-\big(1+(x/s)^c\big)^{-d}$ | 3 | 可驼峰，尾部幂律 |

**嵌套关系**：指数 = Weibull$(k=1)$ = 伽马$(a=1)$；Weibull = 广义伽马$(a=1)$；伽马 = 广义伽马$(c=1)$。嵌套对用似然比：

$$
\mathrm{LR}=2\big(\ell_{\text{大}}-\ell_{\text{小}}\big)\ \sim\ \chi^2_{\Delta k}.
$$

**非嵌套对**（Weibull vs 对数正态、广义伽马 vs Burr）用 Vuong (1989) 检验，$m_i=\ell_{1,i}-\ell_{2,i}$ 是两模型的逐点对数似然之差：

$$
z_V=\frac{\sum_i m_i-\frac{k_1-k_2}{2}\ln n}{\sqrt n\,\hat\sigma_m}\ \overset{a}{\sim}\ N(0,1),\qquad z_V>0\ \text{支持模型 1}.
$$

最后按 $\mathrm{BIC}=k\ln n-2\ell$ 排序。

**注意**：这一节假设间隔独立，但 H1 已经说明它们相关，所以这里的标准误偏小，结论只作描述。**真正用来选分布的是 H4 的标准化残差** $\varepsilon_i=x_i/\psi_i$：报告里同时给出两者。**预期**：成交间隔的 Weibull 形状参数 $k<1$（hazard 递减，Engle、Russell 1998）；广义伽马或 Burr 更好，意味着 hazard 可能先升后降（Grammig、Maurer 2000）。

## 7. H4：ACD 是否刻画了间隔的相关性

**模型**（季节调整后的间隔 $x_i$）：

$$
x_i=\psi_i\,\varepsilon_i,\qquad \psi_i=\omega+\alpha\,x_{i-1}+\beta\,\psi_{i-1},\qquad \varepsilon_i\overset{\text{iid}}{\sim}\frac{\mathrm{Weibull}(k)}{\Gamma(1+1/k)}\ \ (\text{单位均值}).
$$

$k=1$ 就是指数 ACD（EACD）。对数似然：

$$
\ell(\omega,\alpha,\beta,k)=\sum_i\Big[\ln k-\ln x_i+k\ln\frac{\Gamma(1+1/k)\,x_i}{\psi_i}-\Big(\frac{\Gamma(1+1/k)\,x_i}{\psi_i}\Big)^k\Big].
$$

**原假设**：模型正确，广义残差（积分 hazard）

$$
e_i=\Big(\frac{\Gamma(1+1/k)\,x_i}{\psi_i}\Big)^k\ \overset{\text{iid}}{\sim}\ \mathrm{Exp}(1).
$$

**检验**：

1. EACD vs WACD：$\mathrm{LR}=2(\ell_{\text{WACD}}-\ell_{\text{EACD}})\sim\chi^2_1$（$H_0: k=1$）。
2. 残差上的 H1 四件套：Ljung–Box 不拒绝，说明相关性已被 $\psi_i$ 吸收；色散和 KS 不拒绝，说明误差分布选对了。
3. 对 $\varepsilon_i=x_i/\psi_i$ 跑一遍 H3，选误差分布。
4. 报告持久性 $\hat\alpha+\hat\beta$ 及其标准误（代码里忽略了 $\alpha,\beta$ 的协方差，只是粗略值）。

**判定与预期**：Ljung–Box 不再拒绝、色散仍略大于 1 是常见结果，这时把误差分布换成广义伽马或 Burr（Grammig、Maurer 2000）。持久性 $\hat\alpha+\hat\beta$ 一般在 0.9 以上。**不做季节调整，持久性会被高估**（演示中真值 0.9 被估成 0.98，见 [`event-arrivals.md`](event-arrivals.md) 第 11 节）。

## 8. H5：Hawkes 是否刻画了聚集

在业务时间 $u_i$ 上拟合指数核 Hawkes：

$$
\lambda^*(u)=\mu+\alpha\sum_{u_j<u}e^{-\beta(u-u_j)},\qquad
\ell=\sum_i\ln\lambda^*(u_i)-\mu U-\frac{\alpha}{\beta}\sum_i\big(1-e^{-\beta(U-u_i)}\big),
$$

分支比 $n=\alpha/\beta<1$ 是平稳条件，也是「被其他事件引发的事件占比」；半衰期 $\ln 2/\beta$。

**检验**：

1. 泊松 vs Hawkes：$\mathrm{LR}=2(\ell_{\text{Hawkes}}-\ell_{\text{Poisson}})$。原假设 $\alpha=0$ 在参数空间边界上，只有一个参数在边界时，渐近分布是 $\tfrac12\chi^2_0+\tfrac12\chi^2_1$（Self、Liang 1987），p 值减半。另外 $\alpha=0$ 时 $\beta$ 不可识别（Davies 问题），所以这个 p 值只作参考，同时看 AIC。
2. 补偿量增量

$$
e_i=\int_{u_{i-1}}^{u_i}\lambda^*(u)\,du=\mu\,\Delta u_i+\frac{\alpha}{\beta}\big(1+A_{i-1}\big)\big(1-e^{-\beta\Delta u_i}\big),\qquad A_i=e^{-\beta\Delta u_i}\big(1+A_{i-1}\big),
$$

   在原假设下 iid Exp(1)；对它跑 H1 四件套。
3. 报告 $\hat n$ 及其标准误（delta 方法：$\mathrm{se}(\hat n)^2\approx(\mathrm{se}_\alpha/\beta)^2+(\alpha\,\mathrm{se}_\beta/\beta^2)^2$，忽略协方差）。

**判定与预期**：Hawkes 的 AIC 会远好于泊松。如果补偿量残差的 Ljung–Box 或色散仍拒绝，说明单指数核不够，下一步换双指数核或幂律核（Bacry 等 2016；代码中标为 TODO）。$\hat n$ 的文献值在 0.6–1 之间，是否接近 1 有争议（Filimonov、Sornette 2012；Hardiman 等 2013；Lallouache、Challet 2016）；它对季节调整和窗口长度很敏感，所以 H9 要看它跨日是否稳定。

## 9. H6：最优档的新增和撤单强度是否依赖队列长度

Cont–de Larrard 假设各类事件强度是常数；queue-reactive 模型（Huang、Lehalle、Rosenbaum 2015）认为强度是队列长度的函数。用 L1 快照检验：对每一侧，相邻两个快照之间价格不变的区间 $k$ 有队列长度 $q_k$、暴露时长 $\Delta_k$，区间以「量增加」（新增）或「量减少」（撤单或成交）结束；以价格变动结束的区间只计暴露、不计事件。把 $q$ 按时间加权分位数分成 $B=5$ 箱，第 $b$ 箱的事件数 $N_b$、总暴露 $T_b$：

$$
\ell_1=\sum_{b}\big(N_b\ln\hat\lambda_b-\hat\lambda_b T_b\big),\quad \hat\lambda_b=\frac{N_b}{T_b};\qquad
\ell_0=N\ln\hat\lambda-\hat\lambda T,\quad \hat\lambda=\frac{N}{T};\qquad
\mathrm{LR}=2(\ell_1-\ell_0)\sim\chi^2_{B-1}.
$$

效应量是 $\max_b\hat\lambda_b/\min_b\hat\lambda_b$。

**一定要在业务时间里算暴露。** 繁忙时段事件强度高，队列长度分布也不同，两者混在一起会制造虚假的依赖。合成数据里的强度本来与队列长度无关，用物理时间算暴露时，某一天的最大 / 最小比是 1.68，检验错误地拒绝了原假设；改用业务时间后比值降到 1.04，不再拒绝（第 13 节）。代码默认用 H2 的季节曲线换算业务时间。

**L1 近似的误差**：两次快照之间的多个事件会互相抵消，强度被低估；成交和撤单按「同价位成交量」拆分，可能分错。有 L2 时直接用 `level2` 表的逐笔事件计数。**预期**：拒绝。Huang 等（2015）的估计里，撤单强度随队列长度上升，新增强度在长队列时下降。

## 10. H7：下一次中间价变动的等待时间

**原假设**（Cont–de Larrard 2013）：以平均订单量为单位，每侧最优档是生灭过程，新增速率 $\lambda$，移除速率 $c=\mu+\theta$（市价单加撤单）。从 $x$ 单开始的耗尽时间密度为

$$
f_x(t)=\frac{x}{t}\Big(\frac{c}{\lambda}\Big)^{x/2}I_x\big(2\sqrt{\lambda c}\,t\big)\,e^{-(\lambda+c)t},\qquad
S_x(t)=1-\int_0^t f_x(u)\,du ,
$$

两侧独立时，下一次中间价变动的等待时间 $\tau$ 满足 $\Pr(\tau>t\mid q^b,q^a)=S_{q^b}(t)\,S_{q^a}(t)$。

**参数估计**：$\hat\lambda=\dfrac{\text{新增量}/\bar v}{T}$，$\hat c=\dfrac{(\text{成交量}+\text{撤单量})/\bar v}{T}$，$\bar v$ 是平均新增订单量；队列长度换算成 $\mathrm{round}(q/\bar v)$，下限 1。

**检验**：

1. **概率积分变换（PIT）**：对每次变动 $k$，算

$$
U_k=S_{q^b_k}(\tau_k)\,S_{q^a_k}(\tau_k),
$$

   原假设下 $U_k\sim\mathrm{U}(0,1)$，用 KS 检验，效应量是 $D$（阈值 0.05）。
2. **尾指数**：平衡时（$\lambda=c$）$S_x(t)\sim x/\sqrt{\pi\lambda t}$，两侧相乘得 $\Pr(\tau>t)\sim t^{-1}$。用 Hill 估计量（取最大的 $k$ 个观测，默认 $k=2\%\,n$）：

$$
\hat\alpha_{\text{Hill}}=\Big[\frac1k\sum_{i=1}^{k}\ln X_{(n-i+1)}-\ln X_{(n-k)}\Big]^{-1},\qquad \mathrm{se}\approx\frac{\hat\alpha}{\sqrt k},\qquad z=\frac{\hat\alpha-1}{\mathrm{se}} .
$$

   不平衡时（$\lambda<c$）尾部按 $e^{-(\sqrt c-\sqrt\lambda)^2t}$ 截断，Hill 估计会偏大。
3. **平衡度**：报告每侧的 $\hat c/\hat\lambda$。

**判定与预期**：在**完全符合模型**的合成数据（泊松流、无季节）上，代码给出 PIT 的 $D=0.020$（$p=0.73$）、Hill 估计 0.99，说明实现正确。加入季节和 Hawkes 聚集后，PIT 的 $D$ 升到 0.26，被拒绝。真实数据预计也会被拒绝；这时它的用途是 [`event-arrivals.md`](event-arrivals.md) 第 10.3 节的基线 B1，被拒绝的方向（尾部过重还是过轻）告诉我们 hazard 模型里要加什么。

## 11. H8：下一次中间价变动的方向

**原假设**：平衡流下，下一次变动向上的概率只取决于两侧队列长度：

$$
\phi(n,p)=\frac{1}{\pi}\int_0^{\pi}\Big(2-\cos t-\sqrt{(2-\cos t)^2-1}\Big)^{p}\,\frac{\sin(nt)\cos(t/2)}{\sin(t/2)}\,dt ,
$$

$n=q^b$，$p=q^a$（以平均订单量为单位）。

**检验**：用逻辑回归做校准（Cox 1958）。$y_k=1$ 表示第 $k$ 次之后的下一次变动向上：

$$
\Pr(y_k=1)=\sigma\big(a+b\,\mathrm{logit}\,\phi_k\big),\qquad H_0:\ a=0,\ b=1,\qquad
W=(\hat\theta-\theta_0)^\top\widehat{\mathrm{Cov}}(\hat\theta)^{-1}(\hat\theta-\theta_0)\sim\chi^2_2 .
$$

斜率 $b<1$ 说明 $\phi$ 太极端，$b>1$ 说明太保守。同时报告 Brier 分数 $\frac1K\sum_k(\phi_k-y_k)^2$，以及相对「总猜 0.5」的技能分 $1-\mathrm{Brier}/\mathrm{Brier}_{0.5}$。

**预期**：队列不平衡对下一次方向有明显的预测力，大 tick 股票尤其强（Gould、Bonart 2016）。斜率可能偏离 1，这时用 $\mathrm{logit}\,\phi$ 当 hazard 模型的一个特征（让模型自己学斜率），而不是直接当概率用。合成数据（完全符合模型）上斜率为 0.93，$p=0.61$。

## 12. H9：参数跨日是否稳定

对 H4 的持久性 $\hat\alpha+\hat\beta$、形状 $\hat k$，以及 H5 的分支比 $\hat n$，每只股票用 $D$ 天的估计 $\hat\theta_d$ 和标准误 $s_d$：

$$
w_d=\frac{1}{s_d^2},\qquad \bar\theta=\frac{\sum_d w_d\hat\theta_d}{\sum_d w_d},\qquad
Q=\sum_d w_d\big(\hat\theta_d-\bar\theta\big)^2\sim\chi^2_{D-1},\qquad
I^2=\max\Big(0,\frac{Q-(D-1)}{Q}\Big).
$$

$I^2$ 是跨日差异中「真实异质」所占的比例。$I^2>0.5$ 时，模型参数要滚动估计（例如每天用前 5 天），不能用一个固定值。

## 13. 结果表和解读

### 13.1 输出格式

[`report.py`](../../src/arrivals/report.py) 输出一张长表，每个检验一行：

| 列 | 含义 |
|---|---|
| `symbol`, `date` | 股票、日期（H9 的 `date` 为 `all`） |
| `hypothesis`, `series`, `test` | 如 `H4`、`WACD residuals`、`Ljung-Box(20)` |
| `stat`, `se`, `p_value` | 统计量（参数行是估计值）、标准误、p 值 |
| `effect`, `effect_name` | 效应量及其名称（CV、KS D、rho_1、Fano……） |
| `n` | 样本量 |
| `verdict` | reject / reject (small effect) / not rejected / info |
| `reject_fdr` | BH 控制 FDR 后是否拒绝 |
| `note` | 补充信息（AIC、各分箱强度、季节曲线是否样本外等） |

终端同时打印汇总：每个 (假设, 事件流, 检验) 的天数、被拒绝的占比、效应量中位数。

### 13.2 合成数据上的参考输出

`PYTHONPATH=src python -m arrivals.report --synthetic --days 3` 的汇总（3 天，每天 6.5 小时；合成数据含 U 形日内季节、分支比 0.6 的 Hawkes 市价单、平衡的排队流；单核约 30 秒）。H3 和参数估计行的 verdict 是 info，不进汇总，在 CSV 里看：

| 假设 | 事件流 | 检验 | 拒绝占比 | 效应量中位数 |
|---|---|---|---|---|
| H1 | E1 成交 | 色散 | 3/3 | CV = 1.97 |
| H1 | E1 成交 | KS | 3/3 | D = 0.25 |
| H1 | E1 成交 | 离散指数，60 秒窗口 | 3/3 | Fano = 14.5 |
| H2 | E1 成交 | 色散（季节调整后） | 3/3 | CV = 1.79 |
| H2 | E1（业务时间） | 离散指数，60 秒窗口 | 3/3 | Fano = 6.7 |
| H4 | WACD 残差 | Ljung–Box | 0/3 | $\rho_1=-0.004$ |
| H4 | WACD 残差 | 色散 | 3/3 | CV = 1.14 |
| H5 | 泊松 vs Hawkes | 似然比 | 3/3 | Δ对数似然 ≈ 5700 |
| H5 | Hawkes 补偿量 | 色散 / KS / Ljung–Box | 0/3 | CV = 1.003，D = 0.005 |
| H6 | 最优档新增 / 移除 | 似然比（业务时间） | 0/3 | 最大 / 最小比 = 1.04 / 1.06 |
| H7 | E2 等待时间 | PIT KS | 3/3 | D = 0.26 |

读法：数据本身是 Hawkes 生成的，所以 H1、H2 被拒绝，ACD 吸收了相关性但误差分布不对（色散仍拒绝），Hawkes 的残差三项全部通过；H6 正确地不拒绝；H7 因为聚集和季节被拒绝。这和数据生成机制一致，说明整套检验能分辨这些情形。

### 13.3 结果怎么用：决策表

| 结果 | 结论 | hazard 模型里怎么做 |
|---|---|---|
| H1 拒绝，H2 不拒绝 | 只有日内季节 | 只加季节因子 $\hat s(t)$ |
| H2 拒绝，H4 残差通过 | ACD 足够 | 加 $\ln\hat\psi_i$（期望间隔） |
| H2 拒绝，H5 残差通过 | 自激聚集是主因 | 加多尺度衰减计数（[`event-arrivals.md`](event-arrivals.md) 第 10.2 节），不必单独拟合 Hawkes |
| H5 残差仍拒绝 | 单指数核不够 | 衰减计数用 3–4 个尺度（10、1、0.1、0.01 /秒） |
| H6 拒绝 | 强度依赖队列 | 加 $q^b,q^a$、不平衡度及其与流量的交互；模拟器用 queue-reactive |
| H7 拒绝、H8 有区分度 | 方向好预测，时间不好预测 | 方向用 $\mathrm{logit}\,\phi$ 当特征；时间交给 hazard 的时变部分 |
| H9 的 $I^2>0.5$ | 参数不稳定 | 滚动估计，监控漂移 |

## 14. 明天的执行步骤

1. **导出数据**：按第 1.1 节的字段导出 trades 和 quotes（CSV 或 Parquet；读 Parquet 需要 `pyarrow`），第 1.2 节的股票和日期范围。
2. **装依赖**：`pip install numpy scipy pandas pytest`（pyarrow 可选）。
3. **自检**：在仓库根目录运行单元测试和合成数据 dry run。

```bash
PYTHONPATH=src python -m pytest -q src/tests/test_arrivals.py
PYTHONPATH=src python -m arrivals.report --synthetic --days 3 --out synthetic.csv
```

4. **单只股票单日**：先跑一个小样本看耗时和输出。`--resolution-ns` 按数据的时间戳精度填（毫秒精度填 1000000）。

```bash
PYTHONPATH=src python -m arrivals.report \
    --trades trades_one.csv --quotes quotes_one.csv \
    --tz America/New_York --open 09:30 --close 16:00 \
    --resolution-ns 1000000 --out one_day.csv
```

5. **全部样本**：同一条命令换成全量文件。每个股票日的耗时主要在 Hawkes 拟合（合成数据 2 万事件约几秒，$10^5$ 事件估计 1 分钟左右，未实测）；先加 `--no-hawkes` 跑一遍其他检验。多只股票可以按股票拆文件，在 i9 上开 8–16 个进程并行（每个进程设 `OMP_NUM_THREADS=1`）。
6. **看结果**：先看汇总表，再按第 13.3 节填决策表；把每个假设的结论和效应量中位数记到 [`event-arrivals.md`](event-arrivals.md) 第 10.2 节对应的特征行。

## 15. 代码结构

| 文件 | 作用 |
|---|---|
| [`schema.py`](../../src/arrivals/schema.py) | 表结构、校验、清洗规则、四条事件流、间隔、L1 流量近似 |
| [`stats.py`](../../src/arrivals/stats.py) | 色散、KS bootstrap、AD、Ljung–Box、离散指数、LR、Vuong、Hill、Cochran Q、校准、BH |
| [`distributions.py`](../../src/arrivals/distributions.py) | 6 族分布的 MLE 和比较（H3） |
| [`models.py`](../../src/arrivals/models.py) | 季节曲线、业务时间、WACD、指数核 Hawkes、残差 |
| [`queue.py`](../../src/arrivals/queue.py) | Cont–de Larrard 密度、生存函数、$\phi$、PIT、H6 似然比 |
| [`synth.py`](../../src/arrivals/synth.py) | 合成数据（同一表结构），`season=False` 且市价单为泊松时就是精确的 Cont–de Larrard 模型 |
| [`report.py`](../../src/arrivals/report.py) | H1–H9 的执行顺序、判定规则、命令行 |
| [`test_arrivals.py`](../../src/tests/test_arrivals.py) | 7 个单元测试：零假设下不拒绝、明显违背时拒绝、精确 Cont–de Larrard 情形 |

**给后续修改的说明**（也写在各文件的 docstring 里）：

- 新增一个检验：在 `stats.py` 写一个返回 dict（`stat`、`p_value`、`effect`、`effect_name`、`n`）的函数，在 `report.run_day` 里调用 `_row(...)` 加一行；效应量阈值加到 `Config`。
- 有 L2 数据时：在 `schema.py` 加一个从 `level2` 表计数的函数，替换 `queue.estimate_rates` 和 `queue.queue_reactive_lr` 里的 L1 近似。
- Hawkes 太慢：给 `models._hawkes_A` 加 `numba.njit`（循环已经写成 numba 友好的形式）。
- 双指数核或幂律核 Hawkes：在 `models.py` 仿照 `fit_hawkes_exp` 增加，`report.run_day` 的 H5 段同时报告两种核的 AIC。

## 16. 公式的 LaTeX 源码

每个公式块右上角都有「复制 LaTeX」按钮。下面是本页全部检验统计量的完整源码，可以整段粘贴到 Overleaf（需要 `amsmath`、`amssymb`）：

```latex
\documentclass{article}
\usepackage{amsmath,amssymb}
\begin{document}
\section*{Arrival-time hypothesis tests}

\paragraph{Decision rule.}
\[
\text{verdict}=\begin{cases}
\text{reject} & p<\alpha \text{ and } |\text{effect}-\text{null}|>\delta\\
\text{reject (small effect)} & p<\alpha \text{ only}\\
\text{not rejected} & p\ge\alpha
\end{cases}\qquad
k=\max\Big\{i:\ p_{(i)}\le \tfrac{i}{m}\,q\Big\}\ \text{(Benjamini--Hochberg)}
\]

\paragraph{H1: homogeneous Poisson.} $X_i \overset{\text{iid}}{\sim}\mathrm{Exp}(\lambda)$, $e_i=X_i/\bar X$.
\[
z=\sqrt{\tfrac{n}{8}}\,(\hat\sigma_e^2-1)\overset{a}{\sim}N(0,1),\qquad
D=\sup_x\big|\hat F_n(x)-(1-e^{-x/\bar X})\big|,
\]
\[
A^2=-n-\frac1n\sum_{i=1}^n(2i-1)\big[\ln F(x_{(i)})+\ln(1-F(x_{(n+1-i)}))\big],\qquad
Q=n(n+2)\sum_{h=1}^{H}\frac{\hat\rho_h^2}{n-h}\sim\chi^2_H,
\]
\[
I=\frac{(k-1)s_N^2}{\bar N}\sim\chi^2_{k-1},\qquad F(\Delta)=\frac{s_N^2}{\bar N}.
\]

\paragraph{H2: NHPP after intraday seasonality.}
\[
\hat\lambda(b)=\frac{1}{D-1}\sum_{d'\ne d}\frac{N_{d'}(b)}{w},\qquad
\hat\Lambda(t)=\int_0^t\hat\lambda(u)\,du,\qquad
e_i=\hat\Lambda(t_i)-\hat\Lambda(t_{i-1})\overset{\text{iid}}{\sim}\mathrm{Exp}(1).
\]

\paragraph{H3: marginal law.}
\[
f_{\mathrm{GG}}(x)=\frac{|c|\,x^{ca-1}e^{-(x/s)^c}}{\Gamma(a)\,s^{ca}},\qquad
\mathrm{LR}=2(\ell_1-\ell_0)\sim\chi^2_{\Delta k},\qquad
z_V=\frac{\sum_i m_i-\frac{k_1-k_2}{2}\ln n}{\sqrt n\,\hat\sigma_m}.
\]

\paragraph{H4: Weibull ACD.}
\[
x_i=\psi_i\varepsilon_i,\quad \psi_i=\omega+\alpha x_{i-1}+\beta\psi_{i-1},\quad
e_i=\Big(\frac{\Gamma(1+1/k)\,x_i}{\psi_i}\Big)^k\overset{\text{iid}}{\sim}\mathrm{Exp}(1),
\]
\[
\ell=\sum_i\Big[\ln k-\ln x_i+k\ln\frac{\Gamma(1+1/k)x_i}{\psi_i}-\Big(\frac{\Gamma(1+1/k)x_i}{\psi_i}\Big)^k\Big].
\]

\paragraph{H5: exponential Hawkes.}
\[
\lambda^*(u)=\mu+\alpha\sum_{u_j<u}e^{-\beta(u-u_j)},\qquad
\ell=\sum_i\ln\lambda^*(u_i)-\mu U-\frac{\alpha}{\beta}\sum_i\big(1-e^{-\beta(U-u_i)}\big),
\]
\[
e_i=\mu\Delta u_i+\frac{\alpha}{\beta}(1+A_{i-1})(1-e^{-\beta\Delta u_i}),\quad
A_i=e^{-\beta\Delta u_i}(1+A_{i-1}),\quad
\mathrm{LR}\sim\tfrac12\chi^2_0+\tfrac12\chi^2_1 .
\]

\paragraph{H6: queue-size dependence.}
\[
\mathrm{LR}=2\Big[\sum_b\big(N_b\ln\hat\lambda_b-\hat\lambda_bT_b\big)-\big(N\ln\hat\lambda-\hat\lambda T\big)\Big]\sim\chi^2_{B-1}.
\]

\paragraph{H7: Cont--de Larrard first passage.}
\[
f_x(t)=\frac{x}{t}\Big(\frac{c}{\lambda}\Big)^{x/2}I_x\big(2\sqrt{\lambda c}\,t\big)e^{-(\lambda+c)t},\qquad
U_k=S_{q^b_k}(\tau_k)\,S_{q^a_k}(\tau_k)\sim\mathrm{U}(0,1),
\]
\[
\hat\alpha_{\text{Hill}}=\Big[\frac1k\sum_{i=1}^{k}\ln X_{(n-i+1)}-\ln X_{(n-k)}\Big]^{-1}.
\]

\paragraph{H8: direction.}
\[
\phi(n,p)=\frac1\pi\int_0^\pi\Big(2-\cos t-\sqrt{(2-\cos t)^2-1}\Big)^p\frac{\sin(nt)\cos(t/2)}{\sin(t/2)}\,dt,\qquad
\Pr(y=1)=\sigma(a+b\,\mathrm{logit}\,\phi).
\]

\paragraph{H9: stability across days.}
\[
Q=\sum_d w_d(\hat\theta_d-\bar\theta)^2\sim\chi^2_{D-1},\qquad I^2=\max\Big(0,\frac{Q-(D-1)}{Q}\Big).
\]
\end{document}
```

## 参考文献

- Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3/4):562–565.
- Stacy, E. W. (1962). A generalization of the gamma distribution. *Ann. Math. Statist.* 33(3):1187–1192.
- Lilliefors, H. W. (1969). On the Kolmogorov–Smirnov test for the exponential distribution with mean unknown. *JASA* 64(325):387–389.
- Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Ann. Statist.* 3(5):1163–1174.
- Ljung, G. M., Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika* 65(2):297–303.
- Self, S. G., Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *JASA* 82(398):605–610.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica* 57(2):307–333.
- Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B* 57(1):289–300.
- Engle, R. F., Russell, J. R. (1998). Autoregressive conditional duration. *Econometrica* 66(5):1127–1162.
- Grammig, J., Maurer, K.-O. (2000). Non-monotonic hazard functions and the autoregressive conditional duration model. *Econometrics Journal* 3(1):16–38.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., Frank, L. M. (2002). The time-rescaling theorem and its application to neural spike train data analysis. *Neural Computation* 14(2):325–346.
- Higgins, J. P. T., Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine* 21(11):1539–1558.
- Filimonov, V., Sornette, D. (2012). Quantifying reflexivity in financial markets. *Phys. Rev. E* 85:056108.
- Cont, R., de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Math.* 4(1):1–25.
- Hardiman, S. J., Bercot, N., Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. *Eur. Phys. J. B* 86:442.
- Huang, W., Lehalle, C.-A., Rosenbaum, M. (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110(509):107–122.
- Bacry, E., Jaisson, T., Muzy, J.-F. (2016). Estimation of slowly decreasing Hawkes kernels. *Quantitative Finance* 16(8):1179–1201.
- Gould, M. D., Bonart, J. (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. *Market Microstructure and Liquidity* 2(2):1650006.
- Lallouache, M., Challet, D. (2016). The limits of statistical significance of Hawkes processes fitted to financial data. *Quantitative Finance* 16(1):1–11.
