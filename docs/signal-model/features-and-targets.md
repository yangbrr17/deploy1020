# 变量与预测目标

2026-10 调研整理。tick 级盘口信号的输入变量（定义、直觉、经济含义）、背后的经济学模型、跨股票标准化，以及预测目标怎么选。总览见 [`modeling-guide.md`](modeling-guide.md)。

## 结论

1. **最强的短期输入是订单流（OFI 类）和盘口不平衡（queue imbalance、microprice）。** 约一分钟的窗口内，OFI 和同期中间价变化近似线性，斜率和深度成反比（Cont、Kukanov、Stoikov 2014）。多档 OFI 用第一主成分合成 integrated OFI 后，同期解释力的样本外 $R^2$ 从约 65% 升到约 84%（Cont、Cucuringu、Zhang 2023）。
2. **tick 大小是第一位的分组变量。** 大 tick 股票（价差几乎总是 1 档、队列长）的中间价变动本质上是「哪一侧队列先被打穿」，不平衡类变量很强；小 tick 股票盘口稀疏，可预测性弱得多。
3. **「下一笔成交价涨跌」这个标签被 bid-ask bounce 严重污染。** 模型学到的多半是「下一笔是主动买还是主动卖」，不是价值变化。标签要用中间价或 microprice。
4. **FI-2010 那种平滑标签不适合执行。** 阈值是为了平衡类别选的，没有经济含义；对 horizon 很敏感；平均值也不可交易。
5. **三重障碍是好的起点，升级成「首达时间 + 竞争风险」更好。** 一个模型给出一组 τ 下先涨 / 先跌的概率。
6. **原始的 OFI 回归是「同期」关系，不是预测。** 拿来预测时信号弱得多，有效 horizon 只有约 2 次平均价格变动（Kolm 等 2023）。

## 1. 记号

| 记号 | 含义 |
|---|---|
| $k = 1, \dots, K$ | 档位，$k = 1$ 为最优档 |
| $P^b_k, P^a_k$ | 第 $k$ 档买价、卖价 |
| $q^b_k, q^a_k$ | 第 $k$ 档买量、卖量 |
| $n$ | 事件序号（每次盘口更新或成交是一个事件） |
| $h$ | 回看窗口（时钟时间或事件数） |
| $\delta$ | 最小价位变动（tick） |
| $m = (P^b_1 + P^a_1)/2$ | 中间价 |
| $S = P^a_1 - P^b_1$，$S_{\text{tk}} = S / \delta$ | 价差，以价格计 / 以 tick 计 |
| $\varepsilon_i \in \{+1, -1\}$，$v_i$ | 第 $i$ 笔成交的主动方向（$+1$ = 主动买）和成交量 |

## 2. 变量分四类

| 类型 | 含义 | 典型变量 | 注意 |
|---|---|---|---|
| **状态量** | 某一时刻盘口的形状 | 队列不平衡、多档不平衡、microprice、价差、深度 | 附带「距上次更新的时间」 |
| **流量** | 一个窗口内发生了什么 | OFI、多档 OFI、成交不平衡、撤单率、队列消耗速度、事件强度、波动、间隔 | 多个窗口（多尺度）；只用 $t$ 之前的数据 |
| **慢变属性** | 股票级、日级的结构参数 | tick 相对价格、平均价差、价差 = 1 档的时间占比、平均队列 / 平均成交、Amihud | 用于跨股票标准化、分组、交互 |
| **跨资产** | 其他资产的流量和状态 | 指数期货、ETF、板块的滞后 OFI 和收益 | **必须滞后**；只在极短 horizon 有增量 |

## 3. 盘口状态量

### 3.1 最优档队列不平衡（OBI）

$$
\mathrm{OBI}_1 = \frac{q^b_1 - q^a_1}{q^b_1 + q^a_1} \in [-1, 1],
\qquad
I = \frac{q^b_1}{q^b_1 + q^a_1} = \frac{1 + \mathrm{OBI}_1}{2} \in [0, 1]
$$

- **直觉**：买一厚、卖一薄，卖一更容易先被吃光或撤光，中间价更可能上移。
- **经济含义**：大 tick 股票的价格变化由「队列耗尽」驱动，不平衡衡量两侧队列的「耗尽竞赛」。在排队模型里，两侧队列独立地以随机游走被消耗，「卖一先耗尽」的概率是 $I$ 的增函数（Cont、de Larrard 2013 给出了解析式）。
- **实证**：逻辑回归 $P(\text{下一次中间价上移}) = \sigma(\beta_0 + \beta_1 I)$ 在大 tick 股票上增益很大、在小 tick 股票上中等（Gould 和 Bonart 2016）。量不平衡能预测下一笔市价单的方向，纳入挂单策略能降低逆向选择（Cartea、Donnelly、Jaimungal 2018）。
- **用法**：做非线性变换（分箱、样条或 $\operatorname{atanh}(\mathrm{OBI})$），**和价差交互**。价差大于 1 档时，价差内可能出现新挂单，中间价可以移动半个 tick，OBI 的含义变了。

### 3.2 多档不平衡

按档位加权：

$$
\mathrm{OBI}_K = \frac{\sum_{k=1}^{K} w_k\,(q^b_k - q^a_k)}{\sum_{k=1}^{K} w_k\,(q^b_k + q^a_k)},
\qquad
w_k = e^{-\omega (k-1)} \ \text{或}\ w_k = 1/k
$$

按价格距离加权（小 tick 股票推荐，因为档位之间可能有空价位）：

$$
D^b(j) = \sum_{k:\ P^b_k \ge m - j\delta} q^b_k,
\qquad
D^a(j) = \sum_{k:\ P^a_k \le m + j\delta} q^a_k,
\qquad
\mathrm{OBI}(j) = \frac{D^b(j) - D^a(j)}{D^b(j) + D^a(j)},\quad j \in \{1, 2, 5, 10\}
$$

深档决定「打穿一档之后还会走多远」，也反映深处的供需意图。

### 3.3 Microprice（Stoikov 2018）

**加权中间价**是最简单的版本：

$$
M^w = I\,P^a_1 + (1 - I)\,P^b_1,
\qquad
\frac{M^w - m}{\delta} = \Big(I - \frac{1}{2}\Big) S_{\text{tk}}
$$

它不是鞅，噪声大：卖一只剩 1 股被撤后，新的卖一更深、量更大，$M^w$ 反而下移。

**Stoikov 的 microprice** 定义为未来中间价的极限条件期望：

$$
P^{\text{micro}}_t = \lim_{i \to \infty} \mathbb{E}\big[m_{\vartheta_i} \mid \mathcal{F}_t\big] = m_t + \sum_{i \ge 1} g_i(I_t, S_t)
$$

$\vartheta_i$ 是第 $i$ 次中间价变动的时刻。假设动力学只依赖状态 $X = (I, S)$、和价格水平无关，把 $I$ 分成 $n_I$ 箱、$S$ 取 $n_S$ 个值，估计三个矩阵：

- $Q$：中间价**不变**时的状态转移矩阵；
- $R_1$：从各状态出发、中间价变动 $\Delta M$ 的概率；$\mathbf{k}$ 是 $\Delta M$ 的取值向量（如 $\pm\delta/2, \pm\delta$）；
- $R_2$：中间价**变动后**新状态的转移矩阵。

$$
g_1 = (\mathbf{I} - Q)^{-1} R_1 \mathbf{k},
\qquad
B = (\mathbf{I} - Q)^{-1} R_2,
\qquad
g_{i+1} = B\,g_i
$$

（$\mathbf{I}$ 是单位矩阵。）$\sum_{i \ge 1} g_i = \sum_{i \ge 0} B^i g_1$ 收敛需要两个条件：$B^i \to B^*$（状态链遍历）和 $B^* g_1 = 0$。估计前把样本按 $(I, S, \Delta M) \mapsto (1 - I, S, -\Delta M)$ 镜像扩充，使 $g(I, S) = -g(1 - I, S)$，这保证了第二个条件。实际截断到 $i \approx 6$。特征用 $G^* = (P^{\text{micro}} - m)/\delta$。

- 它把不平衡和价差映射成「以价格计的预期漂移」，**可以直接和吃单成本 $S/2 + f_t$ 比较**。
- $G^*$ 本身是强特征，也是一个基线模型；有研究在部分市场没复现出它的优势，要在自己的数据上验证。

### 3.4 价差

$S_{\text{tk}}$、$\mathbf{1}[S_{\text{tk}} = 1]$、$S_{\text{tk}}$ 相对日内同时段均值、窗口内 $S_{\text{tk}} = 1$ 的时间占比。既是特征，也是**条件变量**：模型按 $S_{\text{tk}} = 1$ 和 $S_{\text{tk}} > 1$ 分段或交互。大 tick 股票的价差被「钉」在 1 档，真实的隐含价差小于 1 档（Dayri 和 Rosenbaum 2015）。

### 3.5 深度

$$
\mathrm{Depth}_k = \sum_{j=1}^{k} (q^b_j + q^a_j),
\qquad
\widetilde{\mathrm{Depth}}_k = \frac{\mathrm{Depth}_k}{\overline{\mathrm{Depth}}_k^{\,\text{同档、同时段、过去 } N \text{ 日}}},
\qquad
\frac{q^b_1}{\operatorname{median}(v^{\text{MO}})}
$$

最后一个是「要几笔典型市价单才能打穿」。价格冲击系数约和深度成反比（见 4.1 节）。

### 3.6 tick 相对价格（大 tick vs 小 tick）

$$
\frac{\delta}{m}\ (\text{bp}),\qquad
\frac{\delta}{\sigma},\qquad
P(S_{\text{tk}} = 1),\qquad
\overline{S}_{\text{tk}} = \frac{\overline{S}}{\delta}
$$

Briola 等（2025）按 $\overline{S}_{\text{tk}}$ 分组：$\lesssim 1.5$ 为大 tick，$\gtrsim 3$ 为小 tick。tick 越大，价格离散化越严重：价差被钉在 1 档，队列很长，价格变动稀少、多是 $\pm 1$ 档。**这是最重要的跨股票条件变量**，用于分组建模、交互项和标签设计（障碍宽度）。

## 4. 订单流量

### 4.1 OFI（Cont、Kukanov、Stoikov 2014）

单个事件 $n$ 对最优档的贡献：

$$
e_n = q^b_1(n)\,\mathbf{1}\big[P^b_1(n) \ge P^b_1(n-1)\big] - q^b_1(n-1)\,\mathbf{1}\big[P^b_1(n) \le P^b_1(n-1)\big]
- q^a_1(n)\,\mathbf{1}\big[P^a_1(n) \le P^a_1(n-1)\big] + q^a_1(n-1)\,\mathbf{1}\big[P^a_1(n) \ge P^a_1(n-1)\big]
$$

窗口内求和：$\mathrm{OFI}(t-h, t] = \sum_{n \in (t-h, t]} e_n$。

- **直觉**：买一增加（加挂、价格上移）是买压，买一减少（撤单、被吃）是卖压，卖一对称。OFI 把限价单、市价单、撤单统一成「最优档的净供需变化」。
- **价格冲击关系**（同期）：

$$
\Delta m_j = \beta^{\text{OFI}}\,\mathrm{OFI}_j + \epsilon_j,
\qquad
\beta^{\text{OFI}} \approx \frac{c}{\mathrm{AD}}
$$

$j$ 是第 $j$ 个时间区间，$\mathrm{AD}$ 是平均深度。斜率和深度成反比，这就是 Kyle $\lambda$（6.1 节）的微观对应。所以 $\mathrm{OFI}/\mathrm{AD}$ 跨股票可比。基于成交量的价量关系噪声更大。

- **注意**：这是**同期**回归，不是预测。

### 4.2 多档 OFI 和 integrated OFI

第 $\ell$ 档的订单流（Xu、Gould、Howison 2019）：

$$
\mathrm{OF}^b_\ell(n) =
\begin{cases}
q^b_\ell(n) & P^b_\ell(n) > P^b_\ell(n-1) \\
q^b_\ell(n) - q^b_\ell(n-1) & P^b_\ell(n) = P^b_\ell(n-1) \\
-q^b_\ell(n-1) & P^b_\ell(n) < P^b_\ell(n-1)
\end{cases}
\qquad
\mathrm{OF}^a_\ell(n) =
\begin{cases}
-q^a_\ell(n-1) & P^a_\ell(n) > P^a_\ell(n-1) \\
q^a_\ell(n) - q^a_\ell(n-1) & P^a_\ell(n) = P^a_\ell(n-1) \\
q^a_\ell(n) & P^a_\ell(n) < P^a_\ell(n-1)
\end{cases}
$$

$$
\mathrm{OFI}_\ell(t-h, t] = \sum_{n \in (t-h, t]} \big(\mathrm{OF}^b_\ell(n) - \mathrm{OF}^a_\ell(n)\big)
$$

（$\ell = 1$ 时和 4.1 节的 $e_n$ 一致。）

深度标准化和主成分合成（Cont、Cucuringu、Zhang 2023）：

$$
\mathrm{ofi}_\ell = \frac{\mathrm{OFI}_\ell}{Q^{(M)}},
\qquad
Q^{(M)} = \frac{1}{M}\sum_{\ell=1}^{M} \frac{1}{2\,\Delta N}\sum_{n}\big(q^b_\ell(n) + q^a_\ell(n)\big),
\qquad
\mathrm{ofi}^{I} = \frac{w_1^\top \mathbf{ofi}}{\lVert w_1 \rVert_1}
$$

$\Delta N$ 是窗口内的事件数，$w_1$ 是历史上多档 $\mathbf{ofi}$ 向量协方差矩阵的第一主成分。

- 多档 OFI 高度相关，第一主成分解释 89% 以上的方差。同期 $R^2$（样本外）：最优档 64.6%，integrated OFI 83.8%。
- **预测场景下，integrated OFI 没有显著好于最优档 OFI**。作者推测主成分丢了「档位距离」的信息。所以预测时把各档 $\mathrm{ofi}_\ell$ 分开输入，让模型自己学权重。
- 小 tick 股票档位会重排，可以改用「距中间价的 tick 距离」定义档位。

### 4.3 跨资产 OFI

$$
r_i(t, t+f] = \alpha_i + \sum_{j=1}^{N}\sum_{k \in \mathcal{L}} \beta_{ijk}\,\mathrm{ofi}_j^{(k)}(t) + \epsilon_i
\qquad \text{（LASSO 估计）}
$$

实用简化：市场 / 板块的因子 OFI $F_t = \sum_j \omega_j\,\mathrm{ofi}_j(t)$（加权和或第一主成分），取滞后；本股的特质 OFI 是对因子回归的残差 $\mathrm{ofi}_i - \hat b_i F$。

- 订单流有共同因子，解释收益共性的约三分之二（Hasbrouck 和 Seppi 2001）。
- **同期**跨资产项在用 integrated OFI 后没有增量；**滞后**的跨资产 OFI 能改善约 1 分钟的预测，衰减很快（Cont、Cucuringu、Zhang 2023；Chinco、Clark-Joseph、Ye 2019）。

### 4.4 成交不平衡

$$
\mathrm{TI}(t-h, t] = \sum_i \varepsilon_i v_i,
\qquad
\mathrm{TI}^{\text{norm}} = \frac{V^{\text{buy}} - V^{\text{sell}}}{V^{\text{buy}} + V^{\text{sell}}},
\qquad
\mathrm{TI}^{(1/2)} = \sum_i \varepsilon_i\, v_i^{1/2}
$$

$$
\text{意外部分：}\quad \tilde\varepsilon_i = \varepsilon_i - \mathbb{E}\big[\varepsilon_i \mid \varepsilon_{i-1}, \varepsilon_{i-2}, \dots\big]
$$

成交符号有长记忆（4.12 节），可预测的那部分已经被流动性的反向调整抵消，**真正携带信息的是意外部分** $\tilde\varepsilon$。条件期望可以用符号序列的 AR 模型估计。

### 4.5 成交方向怎么判断

| 方法 | 规则 | 评价 |
|---|---|---|
| 交易所主动方标志 | 直接用 | 有就用 |
| Lee-Ready（1991） | $p_i > m^-$ 为买，$p_i < m^-$ 为卖（$m^-$ 是成交前最后一个报价的中间价）；$p_i = m^-$ 时用 tick test | 逐笔数据优先 |
| Tick rule | $p_i$ 高于前一个不同的成交价为买，低于为卖 | 和 Lee-Ready 准确率相近 |
| BVC | 按 bar 切分，$V^{\text{buy}}_\tau = V_\tau\,\Phi\big(\Delta P_\tau / \sigma_{\Delta P}\big)$ | 只适合缺逐笔方向、bar 较粗的场景；**秒级以下基本不适用** |

### 4.6 队列消耗速度

$$
\text{depl}^b = \frac{\text{最优买档被成交的量} + \text{被撤的量}}{h},
\qquad
\text{add}^b = \frac{\text{最优买档新增挂单量}}{h},
\qquad
T^b = \frac{q^b_1}{\max(\text{depl}^b - \text{add}^b,\ \epsilon)}
$$

特征 $\log(T^b / T^a)$ 或 $\mathbf{1}[T^a < T^b]$。大 tick 股票的中间价变动 = 某侧队列被耗尽，**哪侧跑得快比哪侧现在短更有信息量**。理论依据：queue-reactive 模型里各类事件的到达强度只依赖当前队列状态（Huang、Lehalle、Rosenbaum 2015）；Cont、Stoikov、Talreja（2010）和 Cont、de Larrard（2013）给出了「给定盘口时中间价上涨的概率」的解析式。这些模型假设事件间隔服从独立指数分布，和实证不符，当特征的启发有用，当精确概率要谨慎。完整推导（生灭过程首达时间、平衡时的重尾、先涨概率公式）见 [`event-arrivals.md`](event-arrivals.md) 第 7 节。

### 4.7 撤单率

$$
\text{cancel\_rate}^b = \frac{\text{最优（或前 } k\text{）买档撤单量}}{\overline{q^b}},
\qquad
\frac{\text{撤单量}}{\text{新增挂单量}},
\qquad
\frac{C^a - C^b}{C^a + C^b}
$$

一侧集中撤单，说明那一侧的流动性提供者觉得被逆向选择了：买方撤单预示下跌。OFI 已经隐含了撤单，单独拆出来可以区分「被吃掉」和「撤走」。

### 4.8 事件强度（Hawkes）

多元 Hawkes 过程，指数核：

$$
\lambda_e(t) = \mu_e + \sum_{e'} \sum_{t^{e'}_i < t} \alpha_{e e'}\,e^{-\beta_{e e'}(t - t^{e'}_i)}
$$

直接拟合成本高、容易误设。**廉价近似**：对每类事件 $e$（主动买、主动卖、买一加挂、卖一加挂、买一撤单、卖一撤单、中间价上、中间价下）维护指数衰减计数，取几个半衰期：

$$
C_e(t; \beta) = \sum_{t_i^e < t} e^{-\beta (t - t_i^e)},
\qquad
C_e(t_{j}) = e^{-\beta (t_j - t_{j-1})}\Big(C_e(t_{j-1}) + \mathbf{1}[\text{事件 } j-1 \text{ 属于 } e]\Big)
$$

递推形式是 $O(1)$ 增量更新。特征：$C_{\text{buyMO}}/C_{\text{sellMO}}$，$C_e$ 相对日内同时段均值。Hawkes 的似然、估计、分支比的争论和检验见 [`event-arrivals.md`](event-arrivals.md) 第 5 节。

### 4.9 波动

$$
\mathrm{RV}(t-h, t] = \sum_{n} (\Delta \log m_n)^2,
\qquad
N^{\text{mid}}(t-h, t] = \#\{\text{中间价变动}\}
$$

$N^{\text{mid}}$ 是大 tick 股票的波动代理。短期波动决定「τ 内能走几档」，是幅度模型和障碍触达概率的核心输入。**用成交价算会被 bid-ask bounce 抬高**（Roll 1984），要用中间价。

### 4.10 时间间隔（ACD）

距上一笔成交、上一次中间价变动、上一次盘口更新的时间。ACD(1,1)（Engle 和 Russell 1998）：

$$
x_i = \psi_i\,\eta_i,\qquad \psi_i = \omega + a\,x_{i-1} + b\,\psi_{i-1},\qquad \eta_i \overset{\text{iid}}{\sim} \text{均值为 1 的正分布}
$$

特征 $x_i / \psi_i$（意外的快或慢）。成交密集意味着信息到达；中间价很久没动，说明队列长、价格「卡住」。误差分布、季节调整和检验见 [`event-arrivals.md`](event-arrivals.md) 第 4 节。

### 4.11 日内时段

距开盘分钟数（分段、样条或 $\sin(2\pi u), \cos(2\pi u)$，$u$ 为当日时间占比）、距收盘时间、是否临近集合竞价或公告时点。季节性标准化：$x_t / \bar x_{\text{同时段}}$。

### 4.12 其他慢变量

| 变量 | 定义 | 用法 |
|---|---|---|
| Kyle $\lambda$ | 滚动回归 $\Delta m = c + \lambda\,\mathrm{TI} + \epsilon$，或 $\Delta m = c + \beta^{\text{OFI}}\,\mathrm{OFI} + \epsilon$ | 状态变量；和 OFI 交互（$\mathrm{OFI} \times \hat\lambda$ ≈ 预期冲击） |
| Amihud | $\mathrm{ILLIQ} = \operatorname{mean}_d \big(\lvert r_d \rvert / \text{成交额}_d\big)$ | 跨股票分组和标准化 |
| VPIN | $\mathrm{VPIN} = \frac{1}{nV}\sum_{\tau=1}^{n} \lvert V^{\text{sell}}_\tau - V^{\text{buy}}_\tau \rvert$（成交量桶） | **有争议**（Andersen 和 Bondarenko 2014），秒级信号价值有限 |

### 4.13 汇总

| 变量 | 类型 | 一句话直觉 | 优先级 |
|---|---|---|---|
| 最优档 OBI | 状态 | 薄的一侧先被打穿 | 高（大 tick 最高） |
| 多档 / 距离加权不平衡 | 状态 | 深档压力 | 中高（小 tick 更重要） |
| Microprice $G^*$ | 状态 | 条件期望的未来中间价 | 高 |
| 价差 $S_{\text{tk}}$ | 状态 | 流动性和 tick 约束 | 高（条件变量） |
| 深度 | 状态 | 冲击 $\propto 1/\text{深度}$ | 中 |
| $\delta/m$、$P(S_{\text{tk}} = 1)$ | 慢变 | 大 / 小 tick 机制不同 | 高（分组） |
| OFI | 流量 | 最优档净供需 | 高 |
| 多档 OFI | 流量 | 深档订单流 | 中 |
| 跨资产滞后 OFI | 流量 | 领先 / 共同因子 | 中（极短 horizon） |
| 成交不平衡（意外部分） | 流量 | 主动买卖压力 | 中 |
| 队列消耗速度 | 流量 | 哪侧耗尽更快 | 高（大 tick） |
| 撤单率 | 流量 | 提供者撤退 | 中 |
| 事件强度 | 流量 | 活跃度、自激 | 中 |
| 波动、$N^{\text{mid}}$ | 流量 | 能走多远 | 高（幅度、标准化） |
| 时间间隔 | 流量 | 信息到达速度 | 中 |
| 日内时段 | 慢变 | U 形季节性 | 中 |
| Kyle $\lambda$ | 慢变 | 冲击系数 | 中（交互） |
| VPIN、Amihud | 慢变 | 毒性 / 非流动性 | 低 |

## 5. 跨股票标准化

目标：同一个特征值在不同股票、不同时段代表同样的市场含义，才能合并训练。

1. **价格类一律以 tick 计**：$\Delta m/\delta$、$S_{\text{tk}}$、$G^*$、障碍宽度。小 tick 股票再并行一套以波动为单位的版本 $\Delta m / \sigma_h$。
2. **量类除以自身的典型量**：$\mathrm{OFI}/\mathrm{AD}$；队列量除以同档、同时段的历史平均；成交量除以同时段的历史平均。
3. **比值类天然无量纲**：OBI、$\mathrm{TI}^{\text{norm}}$、撤单不平衡、$\log(T^b/T^a)$。
4. **滚动 z-score，只用过去的数据**：$z_t = (x_t - \hat\mu_{t^-}) / \hat\sigma_{t^-}$，$\hat\mu_{t^-}, \hat\sigma_{t^-}$ 用 $t$ 之前 $N$ 日的数据估计。不要一次性对全训练集标准化。
5. **时间尺度**：「平均中间价变动间隔」是重要的股票属性。τ 用物理时间（和执行决策间隔一致），不同股票在同一个 τ 内能走多远，交给这个属性去解释。
6. **股票属性当输入**：$\delta/m$、$P(S_{\text{tk}} = 1)$、$\overline{S}_{\text{tk}}$、平均队列 / 平均成交、日均成交。
7. **提醒**：Sirignano 和 Cont（2019）发现，大样本深度网络上做归一化或按 tick 拆分训练集没有带来提升；但对逻辑回归、GBDT 这类较小的模型，手工标准化通常仍然必要。
8. **截面秩变换要小心**：同一时刻对所有股票做秩变换是合法的，但会抹掉整体市场状态，要同时保留原始标准化值。

## 6. 经济学和金融学模型

### 6.1 Kyle（1985）：订单流携带信息，冲击系数 = 深度的倒数

单期模型：资产价值 $v \sim \mathcal{N}(p_0, \Sigma_0)$，知情者下单 $x$，噪声交易 $u \sim \mathcal{N}(0, \sigma_u^2)$，做市商只看到总订单流 $y = x + u$ 并按期望定价。均衡是线性的：

$$
x = \beta\,(v - p_0),\qquad p = p_0 + \lambda\,y,\qquad
\beta = \frac{\sigma_u}{\sqrt{\Sigma_0}},\qquad
\lambda = \frac{\sqrt{\Sigma_0}}{2\,\sigma_u}
$$

**指导**：带符号订单流（OFI、TI）应是主特征；$\lambda$ 随状态变（信息越不对称、噪声交易越少，$\lambda$ 越大），所以 OFI 要和 $1/\text{深度}$、$\hat\lambda$ 交互；多期版本里知情者把信息逐步释放（拆单），所以较长窗口的订单流也有信息。

### 6.2 Glosten-Milgrom（1985）：价差来自逆向选择

价值 $V \in \{V_L, V_H\}$ 各占一半，比例 $\mu$ 的交易者知情，其余随机买卖。做市商零利润，报价等于条件期望：

$$
\text{ask} = \mathbb{E}[V \mid \text{买}],\qquad \text{bid} = \mathbb{E}[V \mid \text{卖}],\qquad
P(V_H \mid \text{买}) = \frac{1 + \mu}{2}
\ \Longrightarrow\
\text{ask} - \text{bid} = \mu\,(V_H - V_L)
$$

**指导**：**挂单被成交本身就是坏消息**。评估挂单要看成交后的 markout；信号和自己方向相反还挂单，等于主动接受逆向选择。价差宽度和信息不对称相关，可以当状态变量。

### 6.3 价格冲击：线性、平方根律、传播子

- 短窗口内 OFI 和价格变化近似线性（4.1 节）。
- 大母单的总冲击对规模呈凹的**平方根律**（Tóth 等 2011）：

$$
\mathcal{I}(Q) \approx Y\,\sigma_D\,\sqrt{\frac{Q}{V_D}},\qquad Y = O(1)
$$

$\sigma_D$ 是日波动，$V_D$ 是日成交量。

- 冲击是瞬时的、会衰减（传播子模型，Bouchaud 等 2004）：

$$
m_t = m_{-\infty} + \sum_{u < t} G(t - u)\,\varepsilon_u\,f(v_u) + \text{噪声},\qquad G(\ell) \sim \ell^{-\beta_G}
$$

**指导**：短 horizon 用线性 OFI，长窗口用 $\varepsilon\sqrt{v}$ 之类的凹变换；**训练数据里的 OFI 不含自己的冲击，上线后要扣除**；执行速度要同时考虑信号带来的 alpha 和冲击成本。

### 6.4 订单流长记忆

订单符号的自相关按幂律衰减（Lillo 和 Farmer 2004）：

$$
C(\ell) = \mathbb{E}[\varepsilon_t\,\varepsilon_{t+\ell}] \sim c\,\ell^{-\gamma},\qquad \gamma \approx 0.6 < 1
$$

$\sum_\ell C(\ell)$ 发散，这是长记忆，主要来自大单拆分执行。但价格接近随机游走，因为流动性做了反向调整：传播子 $G(\ell) \sim \ell^{-\beta_G}$ 的衰减正好抵消了符号的持续性，要让价格扩散，两个指数需满足 $\beta_G = \frac{1 - \gamma}{2} \approx 0.2$（Bouchaud 等 2004）。**「下一笔成交方向」高度可预测，但这不等于价格可预测**，这正是旧标签的陷阱。

### 6.5 价格发现

交易的信息含量可以用收益和成交符号的 VAR 中、交易新息的累计价格响应来度量（Hasbrouck 1991）；多个市场交易同一资产时有「信息份额」（Hasbrouck 1995）。**指导**：主导价格发现的关联品种（期货、ETF）的滞后变量是天然的领先特征。

### 6.6 队列位置的价值

价格-时间优先下，排在前面的订单更值钱：成交时赚价差和被逆向选择的权衡（越靠后，逆向选择越重），加上「随时可以撤」的期权价值（Moallemi 和 Yuan）。**指导**：撤单还是保留要看自己的队位，好队位撤单成本高；大 tick 股票队列长，队位价值更突出。

### 6.7 Roll（1984）价差估计和 bid-ask bounce

成交价 $p_t = m_t + \frac{S}{2}\varepsilon_t$，中间价是随机游走、符号独立时：

$$
\mathrm{Cov}(\Delta p_t, \Delta p_{t-1}) = -\frac{S^2}{4}
\quad\Longrightarrow\quad
\hat S = 2\sqrt{-\mathrm{Cov}(\Delta p_t, \Delta p_{t-1})}
$$

**指导**：成交价序列天然有负自相关，用它做标签或特征会引入假的均值回复。**旧标签的核心缺陷**：上一笔在买一、下一笔在卖一，就记为「涨」，其实中间价没动。

### 6.8 tick 约束

大 tick：价差几乎总是 1 档，队列长，变动稀少，冲击近似永久，不平衡预测力强；小 tick：价差多档，盘口稀疏，冲击依赖历史，可预测性弱（Eisler 等 2012；Briola 等 2025）。**指导**：按 tick 类别分组或交互；「一档」障碍对大 tick 是主要事件，对小 tick 可能只是噪声，小 tick 用 $k$ 档或 $\sigma$ 做单位。

### 6.9 信号驱动的最优执行

加入马尔可夫信号 $\alpha_t$（如 OU 型的不平衡信号 $d\alpha_t = -\kappa\alpha_t\,dt + \eta\,dW_t$）后，最优交易速度是对 Almgren-Chriss 基准的修正，修正项和信号的期望累计漂移成正比（Lehalle 和 Neuman 2019；Cartea 和 Jaimungal 2016）。**指导**：信号输出最好是**带时间尺度的期望漂移和不确定性**，而不只是类别。

## 7. 预测目标

### 7.1 候选

**(A) 下一次中间价变动方向**：$y = \operatorname{sign}(m_{\tau_1} - m_t)$，$\tau_1$ 是 $t$ 之后中间价首次变化的时刻。

**(B) 固定窗口收益**：$y = (m_{t+h} - m_t)/\delta$。

**(C) FI-2010 平滑标签**：

$$
m^+_t = \frac{1}{k}\sum_{i=1}^{k} m_{t+i},\qquad
l_t = \frac{m^+_t - m_t}{m_t},\qquad
y_t = \begin{cases} +1 & l_t > \theta_{\text{lab}} \\ -1 & l_t < -\theta_{\text{lab}} \\ 0 & \text{其他} \end{cases}
\qquad (\theta_{\text{lab}} = 0.002)
$$

**(D) 三重障碍**：上障碍 $m_t + b\delta$，下障碍 $m_t - b\delta$（$b$ 默认 1），时间障碍 $t + \tau$：

$$
y_t = \begin{cases} +1 & \tau^{\uparrow} < \min(\tau^{\downarrow}, t+\tau) \\ -1 & \tau^{\downarrow} < \min(\tau^{\uparrow}, t+\tau) \\ 0 & \text{都没在 } t+\tau \text{ 前触到} \end{cases},
\qquad
\tau^{\uparrow} = \inf\{s > t : m_s \ge m_t + b\delta\}
$$

**(E) 首达时间 / 竞争风险**：$T = \min(\tau^\uparrow, \tau^\downarrow) - t$，事件类型 $J \in \{\uparrow, \downarrow\}$，在 $\tau_{\max}$ 处删失。建模 cause-specific hazard 和累积发生函数：

$$
h_j(u \mid x) = \lim_{\Delta \to 0}\frac{P(u \le T < u + \Delta,\ J = j \mid T \ge u, x)}{\Delta},
\qquad
\mathrm{CIF}_j(\tau \mid x) = P(T \le \tau,\ J = j \mid x) = \int_0^\tau h_j(u \mid x)\,\mathcal{S}(u^- \mid x)\,du
$$

$\mathcal{S}(u \mid x) = \exp\big(-\int_0^u (h_\uparrow + h_\downarrow)\big)$ 是「还没触达」的概率。离散时间版本见 [`ml-pipeline.md`](ml-pipeline.md) 1.5 节。

**(F) 幅度（hurdle 结构）**：

$$
P(\Delta = j) = P(\text{动}) \cdot P(\text{方向} \mid \text{动}) \cdot P\big(\lvert\Delta\rvert = j \mid \text{动}, \text{方向}\big),\qquad j \in \{1, 2, \ge 3\} \text{ tick}
$$

**(G) 分位数**：$Q_{\alpha}\big(m_{t+h} - m_t \mid x\big)$，用 pinball loss $\ell_\alpha(e) = \max(\alpha e, (\alpha - 1)e)$ 训练（$\alpha$ 是分位数水平）。

**(H) Markout**：$d\,(m_{t_f + h} - p_{\text{fill}})/\delta$，$t_f$ 是成交时刻（以 tick 计；以基点计的版本见 [`ml-pipeline.md`](ml-pipeline.md) 3.2 节）。

### 7.2 对比

| 目标 | 优点 | 缺点 | 适合执行吗 |
|---|---|---|---|
| (A) 下一次变动方向 | 去掉了「不变」；对应大 tick 的机制；基准多 | **没有时间尺度**；没有幅度 | 部分适合，要配「何时动」的模型 |
| (B) 固定窗口收益 | 有幅度，能直接和成本比 | 大 tick 股票零值大量堆积；异方差 | 适合当期望漂移，要配分布输出 |
| (C) 平滑标签 | 降低标签噪声，类别相对平衡 | 见 7.3 节 | **不推荐** |
| (D) 三重障碍 | 贴近「先赚一档还是先亏一档」；有时间尺度 | τ 和 $b$ 要选；「没动」混合了「安静」和「来回」 | **适合，好的起点** |
| (E) 首达 / 竞争风险 | 一个模型给出一组 τ 的概率和预期等待时间；处理删失 | 实现更复杂 | **最适合** |
| (F) 幅度 | 对应「涨跌分开、幅度另建模」 | 极端档样本少 | 第二阶段 |
| (G) 分位数 | 给出不确定性和尾部 | 价格离散时退化，大 tick 几乎总是 0 或 ±1 | 主要适合小 tick 或较长 $h$ |
| (H) Markout | 最贴近执行的经济结果 | 只在成交条件下定义，有样本选择 | **当评估指标**，不当通用标签 |

### 7.3 平滑标签的问题

文献指出的：

1. **阈值不是按经济含义选的**，是为了平衡类别（LOBCAST 原文）。同一个 $\theta_{\text{lab}}$ 只在 $k = 5$ 时类别平衡。
2. **类别分布对 horizon 极其敏感**：$k = 1$ 时上 / 平 / 下 = 18 / 63 / 19，$k = 10$ 时 37 / 25 / 38。改 $k$ 实际上是在改问题。
3. **只平滑未来**会让信号不一致，用于交易会产生冗余交易（DeepLOB 原文，因此 DeepLOB 改用 $l_t = (m^+_t - m^-_t)/m^-_t$，$m^-_t$ 是过去 $k$ 个中间价的均值）。
4. **刻画的是趋势，实用性存疑**；高预测精度不等于可执行信号（Briola 等 2025）。
5. FI-2010 数据已经预处理、归一化、打好标签，无法回溯原始盘口。

实现中要注意的：

6. 用 $m^-_t$ 的版本里，过去均值和特征窗口完全重叠，标签的一部分是「过去 $k$ 步的趋势」，准确率虚高，对「从现在起」的交易没有价值。
7. 相邻样本的标签共享未来价格，必须 purge 和 embargo，按日前向滚动。
8. 阈值用收益率表示，跨股票不可比：同样的 $\theta_{\text{lab}}$ 对低价股可能小于 1 tick。用 tick 或 $\sigma$ 做单位。

### 7.4 对当前方案（中间价三重障碍、幅度另建模、涨跌分开）的具体建议

1. **障碍以中间价为基准、以 tick 为单位，并区分价差状态**。价差 = 1 档时，中间价最小变动是 $\delta/2$（价差扩成 2 档）或 $\delta$（整体平移）。障碍定义成 $\lvert m_s - m_t \rvert \ge \delta$，把「只有价差开合」视为没触到或单独一类。也可以用 microprice 做障碍基准，它是鞅，假触发更少。
2. **τ 的主定义用物理时间，等于执行层实际的决策间隔**（如 1 / 5 / 30 秒）。跨股票的可比性靠把「平均中间价变动间隔」当特征解决（统一口径见 [`modeling-guide.md`](modeling-guide.md) 第 4 节）。
3. **记录首达时间，升级为竞争风险模型**（7.1 节 (E)）。提醒：Lo、MacKinlay、Zhang（2002）发现用首达时间构造的「假想成交」不能很好地代表真实成交，用「中间价首次触及」去近似「挂单能否成交」要谨慎。
4. **「没动」细分**：「完全安静」（中间价和价差都没变）和「有活动但没触到」。「没动」只说明 τ 内价格漂移小、等待的价格成本低，**不说明挂单会成交**：完全安静时挂单的成交概率 $q_0$ 往往很低。由 [`ml-pipeline.md`](ml-pipeline.md) 2.8 节的成本公式，「没动」这条分支下挂单比吃单划算的条件是 $q_0\,(S + f_t - f_m) > (1 - q_0)\,c_{\text{wait}}$（$S$ 为价差）：成交了省下整个价差加费率差，没成交要付等待的代价。需要成交模型给出 $q_0$。
5. **幅度用 hurdle 结构**（7.1 节 (F)），涨跌分开；样本不够时共享大部分参数、只让方向相关的项分开。
6. **评估面向执行**：概率校准；MCC 或 macro-F1；覆盖率-精度曲线；挂单 markout、吃单和挂单切换的收益；按 tick 类别、时段、波动分层。
7. **基线**：「OBI + 逻辑回归 + 下一次中间价变动」（Gould 和 Bonart）和 microprice 本身就是强基线，新模型必须在执行指标上打败它们。

## 参考文献

**变量**

- Cont, Kukanov, Stoikov (2014). The price impact of order book events. *J. Financial Econometrics* 12(1):47–88. https://arxiv.org/abs/1011.6402
- Xu, Gould, Howison (2019). Multi-level order-flow imbalance in a limit order book. https://arxiv.org/abs/1907.06230
- Cont, Cucuringu, Zhang (2023). Cross-impact of order flow imbalance in equity markets. *Quantitative Finance* 23(10):1373–1393. https://arxiv.org/abs/2112.13213
- Stoikov (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative Finance* 18(12):1959–1966.
- Gould, Bonart (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. https://arxiv.org/abs/1512.03492
- Cartea, Donnelly, Jaimungal (2018). Enhancing trading strategies with order book signals. *Applied Mathematical Finance* 25(1):1–35.
- Dayri, Rosenbaum (2015). Large tick assets: implicit spread and optimal tick size. *Market Microstructure and Liquidity* 1(1). https://arxiv.org/abs/1207.6325
- Eisler, Bouchaud, Kockelkoren (2012). The price impact of order book events: market orders, limit orders and cancellations. *Quantitative Finance*. https://arxiv.org/abs/0904.0900
- Huang, Lehalle, Rosenbaum (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110:107–122.
- Cont, Stoikov, Talreja (2010). A stochastic model for order book dynamics. *Operations Research* 58(3):549–563.
- Cont, de Larrard (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Mathematics* 4(1):1–25.
- Bacry, Mastromatteo, Muzy (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1). https://arxiv.org/abs/1502.04592
- Engle, Russell (1998). Autoregressive conditional duration. *Econometrica* 66(5):1127–1162.
- Lee, Ready (1991). Inferring trade direction from intraday data. *Journal of Finance* 46(2):733–746.
- Chakrabarty, Pascual, Shkilko (2015). Evaluating trade classification algorithms. *J. Financial Markets* 25:52–79.
- Easley, López de Prado, O'Hara (2012). Flow toxicity and liquidity in a high-frequency world. *RFS* 25(5):1457–1493.
- Andersen, Bondarenko (2014). VPIN and the flash crash. *J. Financial Markets* 17.
- Amihud (2002). Illiquidity and stock returns. *J. Financial Markets* 5(1):31–56.

**经济学模型**

- Kyle (1985). Continuous auctions and insider trading. *Econometrica* 53(6):1315–1335.
- Glosten, Milgrom (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *JFE* 14(1):71–100.
- Roll (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance* 39(4):1127–1139.
- Hasbrouck (1991). Measuring the information content of stock trades. *Journal of Finance* 46(1):179–207.
- Hasbrouck (1995). One security, many markets. *Journal of Finance* 50(4):1175–1199.
- Hasbrouck, Seppi (2001). Common factors in prices, order flows, and liquidity. *JFE* 59(3):383–411.
- Lillo, Farmer (2004). The long memory of the efficient market. *Studies in Nonlinear Dynamics & Econometrics* 8.
- Bouchaud, Gefen, Potters, Wyart (2004). Fluctuations and response in financial markets. *Quantitative Finance* 4(2):176–190.
- Tóth 等 (2011). Anomalous price impact and the critical nature of liquidity. *Physical Review X* 1:021006.
- Chinco, Clark-Joseph, Ye (2019). Sparse signals in the cross-section of returns. *Journal of Finance* 74(1):449–492.
- Moallemi, Yuan. A model for queue position valuation in a limit order book.（工作论文，期刊未核实）
- Lehalle, Neuman (2019). Incorporating signals into optimal trading. *Finance and Stochastics* 23(2):275–311.
- Cartea, Jaimungal (2016). Incorporating order-flow into optimal execution. *Mathematics and Financial Economics* 10:339–364.

**目标与标签**

- Ntakaris 等 (2018). Benchmark dataset for mid-price forecasting of limit order book data. *Journal of Forecasting*. https://arxiv.org/abs/1705.03233
- Zhang, Zohren, Roberts (2019). DeepLOB. *IEEE TSP* 67(11). https://arxiv.org/abs/1808.03668
- Prata 等 (2024). LOBCAST. *Artificial Intelligence Review*. https://arxiv.org/abs/2308.01915
- Briola, Bartolucci, Aste (2025). Deep limit order book forecasting: a microstructural guide. *Quantitative Finance* 25(7):1101–1131.
- Kolm, Turiel, Westray (2023). Deep order flow imbalance. *Mathematical Finance* 33(4):1044–1081.
- Lo, MacKinlay, Zhang (2002). Econometric models of limit-order executions. *JFE* 65(1):31–71.
- Fine, Gray (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA* 94(446):496–509.
- López de Prado (2018). *Advances in Financial Machine Learning*. Wiley.
- Bouchaud, Bonart, Donier, Gould (2018). *Trades, Quotes and Prices*. Cambridge University Press.

未核实：Xu、Gould、Howison 的期刊卷期；Gould 和 Bonart 的期刊版本；加权中间价最早出处的细节。
