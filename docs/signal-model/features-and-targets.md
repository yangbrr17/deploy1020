# 变量与预测目标

2026-10 调研整理。tick 级盘口信号有哪些输入变量、背后的经济学含义、预测目标怎么选。总览见 [`modeling-guide.md`](modeling-guide.md)。

## 结论

1. **最强的短期输入是订单流（OFI 类）和盘口不平衡（queue imbalance、microprice）。** 约一分钟的窗口内，OFI 和同期中间价变化近似线性，斜率和深度成反比（Cont、Kukanov、Stoikov 2014）。多档 OFI 用第一主成分合成 integrated OFI 后，同期解释力的样本外 R² 从约 65% 升到约 84%（Cont、Cucuringu、Zhang 2023）。
2. **tick 大小是第一位的分组变量。** 大 tick 股票（价差几乎总是 1 档、队列长）的中间价变动本质上是「哪一侧队列先被打穿」，不平衡类变量很强；小 tick 股票盘口稀疏，可预测性弱得多。
3. **「下一笔成交价涨跌」这个标签被 bid-ask bounce 严重污染。** 模型学到的多半是「下一笔是主动买还是主动卖」，不是价值变化。标签要用中间价或 microprice。
4. **FI-2010 那种平滑标签不适合执行。** 阈值是为了平衡类别选的，没有经济含义；对 horizon 很敏感；平均值也不可交易。
5. **三重障碍是好的起点，升级成「首达时间 + 竞争风险」更好。** 一个模型给出任意 τ 下先涨 / 先跌的概率。
6. **原始的 OFI 回归是「同期」关系，不是预测。** 拿来预测时信号弱得多，有效 horizon 只有约 2 次平均价格变动（Kolm 等 2023）。

## 1. 记号

```
k = 1..K          档位，k = 1 为最优档
bid_k, ask_k      第 k 档买价 / 卖价
qb_k, qa_k        第 k 档买量 / 卖量
n                 事件序号（每次盘口更新或成交是一个事件）
h                 回看窗口长度（时钟时间或事件数）
δ                 最小价位变动（tick size）
mid  = (bid_1 + ask_1) / 2
S    = ask_1 − bid_1               价差
S_tk = S / δ                       价差，以 tick 计
sign_i ∈ {+1, −1}                  第 i 笔成交的主动方向（+1 = 主动买）
v_i                                第 i 笔成交量
```

## 2. 变量分四类

| 类型 | 含义 | 典型变量 | 计算 | 注意 |
|---|---|---|---|---|
| **状态量** | 某一时刻盘口的形状 | 队列不平衡、多档不平衡、microprice、价差、深度 | 在 t 时刻的快照上直接算 | 附带「距上次更新的时间」 |
| **流量** | 一个窗口内发生了什么 | OFI、多档 OFI、成交不平衡、撤单率、队列消耗速度、事件强度、波动、间隔 | 在 (t − h, t] 内累积 | 多个窗口（多尺度）；只用 t 之前的数据 |
| **慢变属性** | 股票级、日级的结构参数 | tick 相对价格、平均价差、价差 = 1 档的时间占比、平均队列 / 平均成交、Amihud | 过去 N 日滚动估计 | 用于跨股票标准化、分组、交互 |
| **跨资产** | 其他资产的流量和状态 | 指数期货、ETF、板块的滞后 OFI 和收益 | 和流量一样，但**必须滞后** | 只在极短 horizon 有增量 |

## 3. 盘口状态量

### 3.1 最优档队列不平衡（OBI / queue imbalance）

```
QI_1 = (qb_1 − qa_1) / (qb_1 + qa_1)        取值 [−1, 1]
或   I = qb_1 / (qb_1 + qa_1)                取值 [0, 1]
```

- **直觉**：买一厚、卖一薄，卖一更容易先被吃光或撤光，中间价更可能上移。
- **经济含义**：大 tick 股票的价格变化由「队列耗尽」驱动，不平衡衡量两侧队列的「耗尽竞赛」，也部分反映流动性提供者的意愿。
- **实证**：逻辑回归 + QI 预测下一次中间价变动方向，大 tick 股票上增益很大、小 tick 上中等（Gould 和 Bonart 2016）。量不平衡能预测下一笔市价单的方向，纳入挂单策略能降低逆向选择（Cartea、Donnelly、Jaimungal 2018）。延迟会侵蚀不平衡信息的价值（Lehalle 和 Mounjid）。
- **用法**：做非线性变换（分箱、样条或 atanh），**和价差交互**。价差大于 1 档时 QI 的含义变了：价差内可能出现新挂单，中间价可以移动半个 tick。

### 3.2 多档不平衡

```
按档位加权：
QI_K = Σ_{k=1..K} w_k (qb_k − qa_k) / Σ_{k=1..K} w_k (qb_k + qa_k)      w_k = exp(−λ(k−1)) 或 1/k

按价格距离加权（小 tick 股票推荐）：
D_bid(d) = [mid − d·δ, mid) 区间内的买量
D_ask(d) = (mid, mid + d·δ] 区间内的卖量
QI(d)    = (D_bid(d) − D_ask(d)) / (D_bid(d) + D_ask(d))                d = 1, 2, 5, 10
```

深档决定「打穿一档之后还会走多远」，也反映深处的供需意图。小 tick 股票档位之间可能有空价位，「第 k 档」不等于「距离 k tick」，按价格距离分桶更稳。

### 3.3 Microprice（Stoikov 2018）

```
简单版（加权中间价）：
  Mw = I × ask_1 + (1 − I) × bid_1
  (Mw − mid) / δ = (I − 0.5) × S_tk
  问题：不是鞅，噪声大；卖一只剩 1 股被撤后，Mw 反而下移

Stoikov 的 microprice：
  P_micro = lim E[mid(τ_i) | 当前状态]，τ_i 是中间价变动的时刻
  P_micro = mid + g(I, S)
  估计：把 I 分成 n 箱、S 分成 m 个状态，估计
        Q  = 中间价不变时的状态转移
        R1 = 中间价变动 k 的概率
        R2 = 中间价变动后的新状态
        g_1 = (1 − Q)^(−1) R1 k
        B   = (1 − Q)^(−1) R2，g_{i+1} = B g_i
        对称化 g(I, S) = −g(1 − I, S)
  特征：G* = (P_micro − mid) / δ
```

- microprice 是「在当前盘口状态下，中间价最终会去哪」的条件期望，按构造是鞅，是更合理的公允价估计。
- 它把不平衡和价差映射成「以价格计的预期漂移」，**可以直接和吃单成本（半个价差 + 费用）比较**。
- G\* 本身就是一个强特征，也是一个基线模型。有研究在部分市场没复现出它的优势，要在自己的数据上验证。

### 3.4 价差

```
S_tk；1{S_tk = 1}；S_tk 相对日内同时段均值；窗口内 S_tk = 1 的时间占比
```

既是特征，也是**条件变量**：模型按 `S_tk = 1` 和 `S_tk > 1` 分段，或做交互。在 Glosten-Milgrom 里价差补偿逆向选择；大 tick 股票的价差被「钉」在 1 档，真实的隐含价差小于 1 档（Dayri 和 Rosenbaum 2015）。

### 3.5 深度

```
Depth_k = Σ_{j=1..k} (qb_j + qa_j)
标准化：Depth_k / 同档位、同时段过去 N 日的平均深度
        qb_1 / 中位数市价单规模          # 要几笔典型市价单才能打穿
```

价格冲击系数约和深度成反比：深度越浅，同样的订单流引起的价格变化越大（Kyle λ 的微观对应）。

### 3.6 tick 相对价格（大 tick vs 小 tick）

```
tick_to_price = δ / mid                    （bp）
tick_to_vol   = δ / σ                      σ 为某尺度上的价格波动
P(S = 1)      = 价差等于 1 档的时间占比
mean_S_tk     = 平均价差 / δ               （Briola 等按此分大 / 中 / 小 tick：≲ 1.5 / 中间 / ≳ 3）
```

tick 越大，价格离散化越严重：价差被钉在 1 档，队列很长，价格变动稀少、多是 ±1 档。**这是最重要的跨股票条件变量**，用于分组建模、交互项和标签设计（障碍宽度）。

## 4. 订单流量

### 4.1 OFI（Cont、Kukanov、Stoikov 2014）

```
单个事件 n 的贡献（最优档）：
e_n =  qb_1(n)   × 1{bid_1(n) >= bid_1(n−1)}
     − qb_1(n−1) × 1{bid_1(n) <= bid_1(n−1)}
     − qa_1(n)   × 1{ask_1(n) <= ask_1(n−1)}
     + qa_1(n−1) × 1{ask_1(n) >= ask_1(n−1)}

窗口：OFI(t − h, t] = Σ_{n ∈ (t−h, t]} e_n
```

- **直觉**：买一增加（加挂、价格上移）是买压，买一减少（撤单、被吃）是卖压，卖一对称。OFI 把限价单、市价单、撤单统一成「最优档的净供需变化」。
- **经济含义**：短窗口内价格变化主要由 OFI 驱动，关系线性，斜率约与深度成反比；基于成交量的价量关系噪声更大。
- **注意**：原始实证是**同期**回归，不是预测。

### 4.2 多档 OFI、标准化 OFI 和 integrated OFI

```
多档：OFI^m 用第 m 档的价格和量，按同样规则计算，m = 1..M
      向量 MLOFI = (OFI^1, …, OFI^M)

深度标准化（Cont、Cucuringu、Zhang 2023）：
  Q^{M,h}(t) = 前 M 档在窗口内的平均深度
  ofi^m(t)   = OFI^m(t − h, t] / Q^{M,h}(t)

integrated OFI：
  w_1     = 历史多档 ofi 向量的第一主成分载荷
  ofi^I   = w_1' × ofi / ||w_1||_1
```

- 每多加一档，同期拟合都改善（Xu、Gould、Howison 2019）。
- 多档 OFI 高度相关，第一主成分解释 89% 以上的方差。同期 R²：最优档 OFI 样本外 64.6%，integrated OFI 83.8%。
- **预测场景下，integrated OFI 没有显著好于最优档 OFI**。作者推测主成分丢了「档位距离」的信息。所以预测时可以把各档 OFI 分开输入，让模型自己学权重。
- 小 tick 股票档位会重排，可以改用「距中间价的 tick 距离」定义档位。

### 4.3 跨资产 OFI

```
r_i(t, t+f] = α_i + Σ_j Σ_k β_{i,j,k} × ofi_j^{(kh)}(t) + ε        LASSO 估计（Cont、Cucuringu、Zhang 2023）

实用简化：
  市场 / 板块因子 OFI：指数期货、ETF 或板块成分的 OFI 加权和或第一主成分，取滞后
  特质 OFI：本股 OFI 对因子 OFI 回归后的残差
```

- 订单流有共同因子，解释收益共性的约三分之二（Hasbrouck 和 Seppi 2001）。
- **同期**跨资产项在用 integrated OFI 后没有增量；**滞后**的跨资产 OFI 能改善约 1 分钟的预测，但衰减很快（Cont、Cucuringu、Zhang 2023；Chinco、Clark-Joseph、Ye 2019）。
- 用法：只加少数稳定、有经济含义的领先变量（指数期货、ETF、板块），严格滞后，在 ≤ 1 分钟的 horizon 上验证。

### 4.4 成交不平衡

```
TI(t − h, t]  = Σ_i sign_i × v_i
TI_norm       = (V_buy − V_sell) / (V_buy + V_sell)
笔数不平衡     = (N_buy − N_sell) / (N_buy + N_sell)
凹变换         = Σ_i sign_i × sqrt(v_i)               # 对应冲击的凹性
「意外」部分   = sign_i − E[sign_i | 过去的符号]        # 去掉长记忆中可预测的部分
```

在 Kyle 模型里净订单流携带私人信息。但成交符号有长记忆，而价格接近鞅：可预测的那部分已经被流动性的反向调整抵消。**真正有信息的是「意外」的订单流。**

### 4.5 成交方向怎么判断

| 方法 | 做法 | 评价 |
|---|---|---|
| 交易所主动方标志 | 直接用 | 有就用 |
| Lee-Ready（1991） | 成交价 > 中间价算买，< 算卖；等于中间价时用 tick test。现代逐笔数据用「成交前最后一个报价」 | 逐笔数据优先 |
| Tick rule | 只看成交价相对前一个不同成交价的涨跌 | 和 Lee-Ready 准确率相近 |
| BVC（Easley 等） | 按 bar 切分，`V_buy = V × Z(ΔP / σ_ΔP)` | 只适合缺逐笔方向、bar 较粗的场景；**秒级以下基本不适用** |

### 4.6 队列消耗速度

```
depl_bid = (最优买档被成交的量 + 被撤的量) / h
add_bid  = 最优买档新增挂单量 / h
T_bid    = qb_1 / max(depl_bid − add_bid, ε)       # 预计耗尽时间
特征：log(T_bid / T_ask)，或 1{T_ask < T_bid}
```

大 tick 股票的中间价变动 = 某侧队列被耗尽。**哪侧跑得快比哪侧现在短更有信息量。** 理论依据是排队模型：queue-reactive 模型里事件到达强度只依赖当前队列状态（Huang、Lehalle、Rosenbaum 2015）；Cont、Stoikov、Talreja（2010）和 Cont、de Larrard（2013）给出了「给定盘口时中间价上涨的概率」的解析式。这些模型假设事件间隔独立指数分布，和实证不符，当特征的启发有用，当精确概率要谨慎。

### 4.7 撤单率

```
cancel_rate_bid = 最优（或前 k）买档撤单量 / 平均队列量
cancel_to_add   = 撤单量 / 新增挂单量
撤单不平衡       = (cancel_ask − cancel_bid) / (cancel_ask + cancel_bid)
```

一侧集中撤单，说明那一侧的流动性提供者觉得被逆向选择了：买方撤单预示下跌。OFI 已经隐含了撤单，单独拆出来可以区分「被吃掉」和「撤走」的不同含义。

### 4.8 事件强度（Hawkes）

```
λ_e(t) = μ_e + Σ_{e'} Σ_{t_i^{e'} < t} φ_{e,e'}(t − t_i^{e'})         φ(Δ) = α × exp(−β Δ)

廉价近似：对每类事件 e（主动买、主动卖、买一加挂、卖一加挂、买一撤单、卖一撤单、中间价上、中间价下）
          维护指数衰减计数 C_e(t; β) = Σ_{t_i < t} exp(−β (t − t_i))，取多个半衰期
特征：C_buyMO / C_sellMO；C_e 相对日内同时段均值
```

事件会自激（市价单引发市价单）和互激（撤单引发市价单），强度上升意味着信息可能正在到达。直接拟合 Hawkes 成本高、容易误设；**衰减计数是很好的替代，天然多尺度**。

### 4.9 波动

```
RV(t − h, t] = Σ (Δ log mid)²
N_mid        = 窗口内中间价变动次数         # 大 tick 股票的波动代理
```

短期波动决定「τ 内能走几档」，是幅度模型和障碍触达概率的核心输入，也用于标准化。**用成交价算会被 bid-ask bounce 抬高**，要用中间价或 microprice。

### 4.10 时间间隔

```
dt_trade = 距上一笔成交的时间
dt_mid   = 距上一次中间价变动的时间
dt_book  = 距上一次盘口更新的时间
ACD：ψ_i = ω + a × x_{i−1} + b × ψ_{i−1}；特征 x_i / ψ_i（意外的快或慢）
```

成交密集意味着信息到达或大单在执行；中间价很久没动，说明队列长、价格「卡住」（Engle 和 Russell 1998）。

### 4.11 日内时段

距开盘分钟数（分段、样条或 sin/cos）、距收盘时间、是否临近集合竞价或公告时点。成交量和波动的日内 U 形是知情交易者和流动性交易者策略性集中交易的结果（Admati 和 Pfleiderer 1988）。也可以把每个特征除以「同一时段的历史均值」做季节性标准化。

### 4.12 其他慢变量

| 变量 | 定义 | 用法 |
|---|---|---|
| Kyle λ | 滚动回归 `Δmid = c + λ × 带符号成交量`；或 `Δmid = c + β × OFI`，β ≈ 常数 / 深度 | 状态变量；和 OFI 交互（OFI × λ ≈ 预期冲击） |
| Amihud | `mean(|r_d| / 成交额_d)` | 跨股票分组和标准化，不做秒级特征 |
| VPIN | 成交量桶内 `|V_sell − V_buy|` 的均值 / 桶容量 | **有争议**（Andersen 和 Bondarenko 2014：在闪崩之后而不是之前见顶，预测力主要来自和交易强度的机械关系）。秒级信号价值有限 |

### 4.13 汇总

| 变量 | 类型 | 一句话直觉 | 优先级 |
|---|---|---|---|
| 最优档 QI | 状态 | 薄的一侧先被打穿 | 高（大 tick 最高） |
| 多档 / 距离加权不平衡 | 状态 | 深档压力 | 中高（小 tick 更重要） |
| Microprice G\* | 状态 | 条件期望的未来中间价 | 高 |
| 价差（tick） | 状态 | 流动性和 tick 约束 | 高（条件变量） |
| 深度 | 状态 | 冲击 ∝ 1 / 深度 | 中 |
| tick / 价格、P(S = 1) | 慢变 | 大 / 小 tick 机制不同 | 高（分组） |
| OFI | 流量 | 最优档净供需 | 高 |
| 多档 OFI | 流量 | 深档订单流 | 中 |
| 标准化 / integrated OFI | 流量 | 多档合一、无量纲 | 高（同期）、中（预测） |
| 跨资产滞后 OFI | 流量 | 领先 / 共同因子 | 中（极短 horizon） |
| 成交不平衡 | 流量 | 主动买卖压力 | 中 |
| 队列消耗速度 | 流量 | 哪侧耗尽更快 | 高（大 tick） |
| 撤单率 | 流量 | 提供者撤退 | 中 |
| 事件强度 | 流量 | 活跃度、自激 | 中 |
| 波动 / 中间价变动次数 | 流量 | 能走多远 | 高（幅度、标准化） |
| 时间间隔 | 流量 | 信息到达速度 | 中 |
| 日内时段 | 慢变 | U 形季节性 | 中 |
| Kyle λ | 慢变 | 冲击系数 | 中（交互） |
| VPIN、Amihud | 慢变 | 毒性 / 非流动性 | 低 |

## 5. 跨股票标准化

目标：同一个特征值在不同股票、不同时段代表同样的市场含义，才能合并训练。

1. **价格类一律以 tick 计**：价差、中间价变化、microprice 偏移、障碍宽度。小 tick 股票再并行一套以波动为单位的版本（`Δmid / σ_h`）。
2. **量类除以自身的典型量**：OFI 除以前 M 档平均深度；队列量除以同档、同时段的历史平均；成交量除以同时段的历史平均。
3. **比值类天然无量纲**：QI、TI_norm、撤单不平衡、`log(T_bid / T_ask)` 直接用。
4. **滚动 z-score，只用过去的数据**。不要一次性对全训练集标准化。
5. **时间尺度标准化**：窗口和 horizon 用事件时间，或「平均中间价变动间隔」的倍数。有效 horizon 约为 2 次平均价格变动，所以 τ 应按各股的价格变动频率定标。
6. **股票属性当输入**：tick_to_price、P(S = 1)、平均价差、平均队列 / 平均成交、日均成交。
7. **提醒**：Sirignano 和 Cont（2019）发现，对大样本深度网络做归一化或按 tick 拆分训练集**没有**带来提升，网络似乎能自己学到。但对逻辑回归、GBDT 这类较小的模型，手工标准化通常仍然必要。
8. **截面秩变换要小心**：同一时刻对所有股票做秩变换是合法的，但会抹掉整体市场状态，要同时保留原始的标准化值。

## 6. 经济学和金融学概念

| 概念 | 要点 | 对建模的指导 |
|---|---|---|
| **Kyle（1985）** | 一个知情者、噪声交易者、竞争性做市商；价格线性响应净订单流，斜率 λ 是市场深度的倒数；知情者把信息逐步释放进价格 | 带符号订单流（OFI、TI）应是主特征；冲击系数随状态变，OFI 要和 1 / 深度、λ 交互；知情者拆单，长窗口订单流也有信息 |
| **Glosten-Milgrom（1985）** | 只要有知情交易者，即使做市商零利润，也会有正的价差；每笔成交都会更新报价 | **挂单被成交本身就是坏消息**（逆向选择），评估挂单要看成交后的 markout；信号和自己方向相反还挂单，等于主动接受逆向选择 |
| **价格冲击** | 短窗口内 OFI 和价格变化近似线性（CKS 2014）；大母单的总冲击对规模呈凹的平方根律（Tóth 等 2011）；冲击是瞬时的、会衰减（传播子模型，Bouchaud 等 2004） | 短 horizon 用线性 OFI，长窗口用 `sign × sqrt` 之类的凹变换；**训练数据里的 OFI 不含自己的影响，上线后要扣除自身冲击**；执行速度要同时考虑信号带来的 alpha 和冲击成本 |
| **订单流长记忆** | 订单符号的自相关按幂律衰减（Lillo 和 Farmer 2004），主要来自大单拆分执行；价格却接近随机游走 | **「下一笔成交方向」高度可预测，但这不等于价格可预测**，这正是旧标签的陷阱；特征要包含订单流的「意外部分」 |
| **价格发现** | 交易的信息含量可以用 VAR 中交易新息的累计价格响应来量（Hasbrouck 1991）；多市场有信息份额（Hasbrouck 1995） | 主导价格发现的关联品种（期货、ETF）的滞后变量是天然的领先特征 |
| **队列位置的价值** | 价格-时间优先下，排在前面的订单更值钱：成交时赚价差和被逆向选择的权衡（越靠后逆向选择越重），加上动态的期权价值（Moallemi 和 Yuan） | 撤单还是保留要看自己的队位，好队位撤单成本高；大 tick 股票队列长，队位价值更突出 |
| **Roll 价差估计** | 交易成本使成交价变化负序列相关：`价差 = 2 × sqrt(−Cov(Δp_t, Δp_{t−1}))`（Roll 1984） | 成交价序列天然有负自相关，用它做标签或特征会引入假的均值回复 |
| **Bid-ask bounce** | 成交价在买一和卖一之间跳，不是价值变化 | **旧标签的核心缺陷**：上一笔在买一、下一笔在卖一，就记为涨，其实中间价没动 |
| **tick 约束** | 大 tick：价差几乎总是 1 档，队列长，变动稀少，冲击近似永久，不平衡预测力强；小 tick：价差多档，盘口稀疏，冲击依赖历史，可预测性弱 | 按 tick 类别分组或交互；「一档」障碍对大 tick 是主要事件，对小 tick 可能只是噪声，小 tick 用 k 档或 σ 做单位 |
| **信号驱动的最优执行** | 加入马尔可夫信号后，最优策略是对 Almgren-Chriss 基准的修正（Lehalle 和 Neuman 2019；Cartea 和 Jaimungal 2016） | **信号输出最好是带时间尺度的期望漂移和不确定性**，而不只是类别，这样才能直接嵌入执行优化 |

## 7. 预测目标

### 7.1 候选

```
(A) 下一次中间价变动方向
    y = sign(mid(τ_1) − mid(t))，τ_1 = t 之后中间价首次变化的时刻
(B) 固定窗口的中间价收益
    y = (mid(t + h) − mid(t)) / δ
(C) FI-2010 平滑标签
    m_plus = (1/k) Σ_{i=1..k} mid(t + i)
    l = (m_plus − mid(t)) / mid(t)
    y = +1 若 l > α；−1 若 l < −α；否则 0          （α = 0.002）
(D) 三重障碍
    上障碍 mid(t) + b·δ，下障碍 mid(t) − b·δ，时间障碍 t + τ
    y = +1 先触上；−1 先触下；0 都没触到
(E) 首达时间 / 竞争风险
    T = min(T_up, T_down)，事件类型 J ∈ {up, down}，在 τ_max 处删失
    建模 cause-specific hazard：h_up(s | x)、h_down(s | x)
    累积发生函数 CIF_up(τ | x) = P(T <= τ, J = up | x)
(F) 幅度分档
    {<= −2, −1, 0, +1, >= +2} tick；或 P(动) × P(方向 | 动) × P(|Δ| = j | 动, 方向)
(G) 分位数
    q_α(mid(t + h) − mid(t) | x)，pinball loss
(H) Markout
    side × (mid(t_fill + h) − p_fill) / δ
```

### 7.2 对比

| 目标 | 优点 | 缺点 | 适合执行吗 |
|---|---|---|---|
| (A) 下一次变动方向 | 去掉了「不变」；对应大 tick 的机制；基准多 | **没有时间尺度**：下一次可能 1 毫秒后，也可能 10 分钟后；没有幅度 | 部分适合，要配「何时动」的模型 |
| (B) 固定窗口收益 | 有幅度，能直接和成本比 | 大 tick 股票零值大量堆积；异方差；h 难选 | 适合当期望漂移，要配分布输出 |
| (C) 平滑标签 | 降低标签噪声，类别相对平衡 | 见下 | **不推荐** |
| (D) 三重障碍 | 贴近「先赚一档还是先亏一档」；有时间尺度 τ；「没动」有明确含义 | τ 和 b 要选；「没动」混合了「安静」和「来回」 | **适合，好的起点** |
| (E) 首达 / 竞争风险 | 一个模型给出任意 τ 的概率和预期等待时间；处理删失；和排队模型的理论对应 | 实现更复杂 | **最适合**：「多快」和「往哪」一起给 |
| (F) 幅度分档 | 对应「涨跌分开、幅度另建模」 | 极端档样本少 | 第二阶段 |
| (G) 分位数 | 给出不确定性和尾部 | 价格离散时分位数退化，大 tick 几乎总是 0 或 ±1 | 主要适合小 tick 或较长 h |
| (H) Markout | 最贴近执行的经济结果 | 只在成交条件下定义，有样本选择 | **适合当评估指标和决策阈值**，不宜当通用标签 |

### 7.3 平滑标签的问题

文献指出的：

1. **阈值不是按经济含义选的**，是为了平衡类别（LOBCAST 原文）。同一个 α 只在 k = 5 时类别平衡。
2. **类别分布对 horizon 极其敏感**：k = 1 时上 / 平 / 下 = 18 / 63 / 19，k = 10 时 37 / 25 / 38。改 k 实际上是在改问题。
3. **只平滑未来**会让信号不一致，用于交易会产生冗余交易（DeepLOB 原文）。
4. **刻画的是趋势，实用性存疑**；高预测精度不等于可执行信号（Briola 等 2025）。
5. FI-2010 数据本身已经预处理、归一化、打好标签，无法回溯原始盘口，留有大量过拟合空间。

实现中要注意的：

6. 「未来均值 vs 过去均值」的版本里，过去均值和特征窗口完全重叠，标签的一部分是「过去 k 步的趋势」，准确率虚高，对「从现在起」的交易没有价值。
7. 相邻样本的标签共享未来价格，必须 purge 和 embargo，按日前向滚动；随机切分会严重高估。
8. 阈值用收益率表示，跨股票不可比：同样的 α 对低价股可能小于 1 tick。用 tick 或 σ 做单位。

### 7.4 对当前方案（中间价三重障碍、幅度另建模、涨跌分开）的具体建议

1. **障碍以中间价为基准、以 tick 为单位，并区分价差状态**。价差 = 1 档时，中间价最小变动是 0.5 档（价差扩成 2 档）或 1 档（整体平移）。障碍定义成「中间价变动 ≥ 1 tick」，把「只有价差开合」单独记一类或视为没触到。也可以用 microprice 做障碍基准，它是鞅，假触发更少。
2. **τ 用事件时间或「平均中间价变动间隔」的倍数**，保证跨股票可比；同时保留多个 τ。
3. **记录首达时间，升级为竞争风险模型**：离散时间多项 hazard（每格 {up, down, 无事件}，样本按「仍在风险中」展开）。连续时间可用 cause-specific Cox 或 Fine-Gray。提醒：Lo、MacKinlay、Zhang（2002）发现用首达时间构造的「假想成交」不能很好地代表真实成交，用「中间价首次触及」去近似「挂单能否成交」要谨慎。
4. **「没动」细分**：「完全安静」（中间价和价差都没变）和「有活动但没触到」（价差开合、来回）。前者适合挂单等待，后者说明波动在升高。
5. **幅度用 hurdle 结构**：`P(τ 内动) × P(方向 | 动) × P(幅度 = 1, 2, ≥3 | 动, 方向)`。涨跌分开能捕捉不对称，但样本要够，或者共享大部分参数、只让方向相关的项分开。
6. **评估面向执行**：概率校准；MCC 或 macro-F1；覆盖率-精度曲线；挂单 markout、吃单和挂单切换的收益；按 tick 类别、时段、波动分层。
7. **基线**：「QI + 逻辑回归 + 下一次中间价变动」（Gould 和 Bonart）和 microprice，本身就是强基线。新模型必须在执行指标上打败它们。

## 参考文献

**变量**

- Cont, Kukanov, Stoikov (2014). The price impact of order book events. *J. Financial Econometrics* 12(1):47–88. https://arxiv.org/abs/1011.6402
- Xu, Gould, Howison (2019). Multi-level order-flow imbalance in a limit order book. https://arxiv.org/abs/1907.06230
- Cont, Cucuringu, Zhang (2023). Cross-impact of order flow imbalance in equity markets. *Quantitative Finance* 23(10):1373–1393. https://arxiv.org/abs/2112.13213
- Su 等 (2021). The price impact of generalized order flow imbalance. https://arxiv.org/abs/2112.02947
- Stoikov (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative Finance* 18(12):1959–1966.
- Gould, Bonart (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. https://arxiv.org/abs/1512.03492
- Cartea, Donnelly, Jaimungal (2018). Enhancing trading strategies with order book signals. *Applied Mathematical Finance* 25(1):1–35.
- Lehalle, Mounjid. Limit order strategic placement with adverse selection risk and the role of latency. https://arxiv.org/abs/1610.00261
- Dayri, Rosenbaum (2015). Large tick assets: implicit spread and optimal tick size. *Market Microstructure and Liquidity* 1(1). https://arxiv.org/abs/1207.6325
- Curato, Lillo (2015). Modeling the coupled return-spread high frequency dynamics of large tick assets. *J. Stat. Mech.* P01028. https://arxiv.org/abs/1310.4539
- Eisler, Bouchaud, Kockelkoren (2012). The price impact of order book events: market orders, limit orders and cancellations. *Quantitative Finance*. https://arxiv.org/abs/0904.0900
- Huang, Lehalle, Rosenbaum (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110:107–122.
- Cont, Stoikov, Talreja (2010). A stochastic model for order book dynamics. *Operations Research* 58(3):549–563.
- Cont, de Larrard (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Mathematics* 4(1):1–25.
- Bacry, Mastromatteo, Muzy (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1). https://arxiv.org/abs/1502.04592
- Engle, Russell (1998). Autoregressive conditional duration. *Econometrica* 66(5):1127–1162.
- Zhang, Mykland, Aït-Sahalia (2005). A tale of two time scales. *JASA* 100(472):1394–1411.
- Lee, Ready (1991). Inferring trade direction from intraday data. *Journal of Finance* 46(2):733–746.
- Chakrabarty, Pascual, Shkilko (2015). Evaluating trade classification algorithms. *J. Financial Markets* 25:52–79.
- Easley, López de Prado, O'Hara (2012). Flow toxicity and liquidity in a high-frequency world. *RFS* 25(5):1457–1493.
- Easley, López de Prado, O'Hara (2016). Discerning information from trade data. *JFE* 120(2):269–285.
- Andersen, Bondarenko (2014). VPIN and the flash crash. *J. Financial Markets* 17.
- Amihud (2002). Illiquidity and stock returns. *J. Financial Markets* 5(1):31–56.

**经济学概念**

- Kyle (1985). Continuous auctions and insider trading. *Econometrica* 53(6):1315–1335.
- Glosten, Milgrom (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *JFE* 14(1):71–100.
- Roll (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance* 39(4):1127–1139.
- Hasbrouck (1991). Measuring the information content of stock trades. *Journal of Finance* 46(1):179–207.
- Hasbrouck (1995). One security, many markets. *Journal of Finance* 50(4):1175–1199.
- Hasbrouck, Seppi (2001). Common factors in prices, order flows, and liquidity. *JFE* 59(3):383–411.
- Lillo, Farmer (2004). The long memory of the efficient market. *Studies in Nonlinear Dynamics & Econometrics* 8.
- Bouchaud, Gefen, Potters, Wyart (2004). Fluctuations and response in financial markets. *Quantitative Finance* 4(2):176–190.
- Bouchaud, Farmer, Lillo (2009). How markets slowly digest changes in supply and demand. https://arxiv.org/abs/0809.0822
- Tóth 等 (2011). Anomalous price impact and the critical nature of liquidity. *Physical Review X* 1:021006.
- Admati, Pfleiderer (1988). A theory of intraday patterns. *RFS* 1(1):3–40.
- Chordia, Roll, Subrahmanyam (2002). Order imbalance, liquidity, and market returns. *JFE* 65(1):111–130.
- Chinco, Clark-Joseph, Ye (2019). Sparse signals in the cross-section of returns. *Journal of Finance* 74(1):449–492.
- Moallemi, Yuan. A model for queue position valuation in a limit order book. （工作论文，期刊未核实）
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

**教科书和综述**

- Bouchaud, Bonart, Donier, Gould (2018). *Trades, Quotes and Prices*. Cambridge University Press.
- Gould 等 (2013). Limit order books. *Quantitative Finance* 13(11):1709–1742. https://arxiv.org/abs/1012.0349

未核实：Xu、Gould、Howison 的期刊卷期；Gould 和 Bonart 的期刊版本；Capponi 和 Cont（2020）「跨冲击不足总冲击 1%」的原文；加权中间价最早出处的细节。
