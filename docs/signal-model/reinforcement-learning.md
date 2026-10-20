# 强化学习与模拟器

2026-10 调研整理。强化学习（RL）在执行里放在哪一层、怎么建 MDP、选什么算法、模拟器怎么做、怎么评估。监督信号本身见 [`modeling-guide.md`](modeling-guide.md)。

## 结论

1. **RL 不做预测，只做「下单方式」这一层的小决策。** 排程层（规则、VWAP 曲线或 Almgren-Chriss）每隔几十秒到几分钟给出一个切片：目标量 q\* 和时间窗 T。RL 只决定这段时间里怎么下单：吃单、挂一档、挂深档、撤单重挂，还是不动。一个切片就是一个 episode，窗口结束时剩余量按规则清扫。
2. **进度带是硬约束，不放进奖励。** 由排程层和「动作屏蔽」保证。样本少时，奖励里的软惩罚学不出可靠的「永不越界」。
3. **路线：监督模块 → contextual bandit → 离线 RL → 模拟器里的在线 RL。** 前两步就可能拿到大部分收益；PPO / DQN 只在校准过的模拟器里做研究对照，产出的策略再蒸馏成决策树。
4. **要尽早在现行规则里加一点受控的随机化**，并记录每个动作被选中的概率（倾向得分）。否则日志全是确定性规则产生的，任何离线评估和离线 RL 都没法评估规则从没选过的动作。
5. **难点是模拟器，不是算力。** 除了 JAX-LOB 和生成式 LOB 模型，其余都能在 CPU 上做。
6. **统计功效是上线评估的瓶颈。** 母单成本的标准差很大，主要指标要换成子单级的。

## 1. 分层

```
母单（方向、总量、当天完成、进度带 [L(t), U(t)]）
  └─ 排程层（规则 / VWAP 曲线 / Almgren-Chriss，不交给 RL）
       每个切片 k：目标量 q*_k，时间窗 [t_k, t_k + T]
         └─ 下单方式层（bandit / RL 的候选）
              每个决策点（事件驱动，或每 1–10 秒）：吃 / 挂一档 / 挂深 / 撤单重挂 / 不动
              输入：盘口特征、信号概率、队列位置估计、切片剩余量和剩余时间、进度偏离
         └─ 风控壳（硬约束）：进度带、价格保护、单笔上限、撤单频率、kill switch
```

好处：动作空间小、反馈快，样本量比「一天一个 episode」高几个数量级；RL 出错的最坏后果有上界，最多是一个切片的清扫成本。

Karpe 等（2020）发现，在模拟器里训练出的 RL 在部分场景下收敛到接近 TWAP 的轨迹，说明排程层面 RL 的增量可能有限。这也支持「排程不交给 RL」。

## 2. MDP 设计

### 状态

| 类别 | 特征 | 说明 |
|---|---|---|
| 切片私有变量 | 切片剩余比例 `q_rem / q*`，剩余时间比例 `τ_rem / T` | 最核心的两个（Nevmyvaka 等 2006） |
| 进度 | 进度偏离 `已成交/Q − target(t)`，到上下界的距离 | 归一化到带宽 |
| 盘口 | 价差（tick 数）、一档和前 N 档量、OBI、`microprice − mid`、近期成交方向和强度、撤单率、短窗波动 | 大 tick 和小 tick 股票分开处理 |
| 信号 | 监督模型输出的 `P(先涨)`、`P(先跌)`，多个时长 | **只当输入**，RL 不重新学预测。可以同时输入信号最近的校准度 |
| 自身挂单 | 有没有挂单、挂在哪档、**队列位置估计**（前方量 / 该档总量）、已挂时长、已部分成交量 | 有 L3 数据才能精确算队列位置；L2 只能用概率模型估 |
| 时段 | 开盘后、收盘前的分钟数，临近集合竞价 | 非平稳的主要来源 |
| 成本上下文 | 当前吃单成本、预估清扫成本 | 来自监督模块 |

原则：先用 10–30 个手工特征，不直接喂原始盘口张量；按股票自身尺度归一化（tick、日均成交量、典型价差），多只股票共享一个策略；只用决策时刻已经到达的信息。

### 动作（离散，推荐起点）

| 动作 | 含义 | 备注 |
|---|---|---|
| `TAKE` | 吃对手一档 | 数量 = min(切片剩余, 对手一档量 × α) |
| `JOIN_L1` | 挂在己方一档，排队尾 | |
| `IMPROVE` | 价差大于 1 档时挂进价差内 | 价差 = 1 档时屏蔽 |
| `POST_DEEP` | 挂二档或更深 | 成交概率低 |
| `CANCEL` / `REPOST` | 撤单，或撤了按最新盘口重挂 | 重挂会丢掉队列位置 |
| `HOLD` | 不动 | 默认动作 |

- **数量先固定成规则**，只学「方式」；第二阶段再加 2–3 个数量档。
- **动作屏蔽**：接近下限只允许 `TAKE` / `IMPROVE`；接近上限禁止 `TAKE`；撤单频率超限禁止 `CANCEL`。SB3-Contrib 的 Maskable PPO 有现成实现，bandit / DQN 只要在 argmax 时过滤。
- **宏动作**：把「挂在 X 档、持有 τ 秒、没成交就撤」定义成一个动作，多步交互压成一步，可以直接用 contextual bandit。
- 价格本质上是离散的 tick，一开始不建议用连续动作（Pan 等 2022 的混合动作是为此设计的）。

### 奖励

`s = +1` 买、`−1` 卖；成本以 bp 计，奖励 = −成本。

```
# 切片级成本（推荐做奖励）：相对决策时刻的中间价 mid_0
cost_slice = s * Σ q_i * (p_i − mid_0) / (q* * mid_0) * 1e4             # 已成交部分
           + s * q_rem_T * (p_cleanup − mid_0) / (q* * mid_0) * 1e4      # 窗口结束时剩余量清扫
reward     = − cost_slice

# markout：成交后价格走势，诊断逆向选择
markout_h = s * (mid(t_i + h) − p_i) / p_i * 1e4        # h = 1s / 5s / 30s，正数是好

# 风险敏感（可选）
penalty = λ * Var(cost) 或 CVaR_α(cost)
```

- **参考价用决策时刻的中间价，不用母单到达价。** 到达价包含切片之前的全部漂移，切片内的动作控制不了，方差很大。
- **清扫成本必须进奖励。** 否则策略会一直挂深档等便宜价，把未成交的风险推给排程层（reward hacking）。
- **评估时可以把切片内的市场漂移当控制变量做方差缩减，训练时不要。** 否则会抹掉信号带来的择时收益。

### Episode

| 方案 | 定义 | 评价 |
|---|---|---|
| 全天母单 | 开盘到收盘 | 样本 = 母单数，太少；不采用 |
| **排程切片** | 1–5 分钟，结束时清扫 | 样本 = 母单数 × 切片数，方差小，**推荐** |
| 单个子单 | 从下单到成交或撤单 | 适合宏动作 bandit |
| 事件驱动 | 每次盘口变化触发 | 最贴近真实，对模拟器要求最高；第二阶段 |

### 必须同时比较的基线

1. 现行规则（如固定挂一档 + 超时吃单）。
2. 同样用信号的固定阈值规则：`P(不利方向) > θ` 就吃单，否则挂一档；阈值网格搜索。只作次要对照。
3. **同一监督信号 + 成本决策层**（[`ml-pipeline.md`](ml-pipeline.md) 2.8 节），或等价的「预测再优化」：`argmax_a [P_fill(a) × 价差收益(a) − (1 − P_fill(a)) × 清扫成本 − 逆向选择(a)]`（Fabre 和 Ragel 2023）。**RL 必须显著好于这一条，而不只是好于第 1、2 条。**
4. 解析解：Cartea 和 Jaimungal（2015）限价 + 市价；Cartea 和 Jaimungal（2016）、Lehalle 和 Neuman（2019）带信号。参数从数据校准。

## 3. 算法选择

| 方法 | 要在线探索 | 能纯离线 | 可解释 | 依赖模拟器 | 适用 |
|---|---|---|---|---|---|
| 监督「预测再优化」（成交概率、markout、清扫成本 + 一步决策） | 否 | 是 | **高** | 低 | **第一步** |
| Contextual bandit（LinUCB、Thompson、DR 离线学习） | 小幅（安全集内） | 有日志时可以 | 高 | 低 | **第二步** |
| 行为克隆 | 否 | 是 | 中 | 无 | 离线 RL 的初始化和安全基线 |
| 离线 RL（离散 CQL / BCQ、IQL、TD3+BC），或拟合 Q 迭代 + GBDT | 否 | 是 | 低到中 | 中 | 第三步，可选 |
| DQN / DDQN / PPO | 是 | 否 | 低 | **高** | 模拟器内的研究对照 |
| SAC / TD3 | 是 | 否 | 低 | 高 | 连续动作和离散 tick 不匹配，不推荐起步 |
| 风险敏感 Q-learning + 决策树蒸馏 | 是 | 否 | **高** | 高 | 可能的最终上线形态（Vyetrenko 和 Xu 2019） |

为什么先走监督和 bandit：

- **不能在线探索**：真实交易里的探索有真实成本。即使探索，也只在「安全动作集」里小比例随机（比如 2–10% 的决策在 2–3 个合理动作里随机），并记录倾向得分。
- **样本少**：bandit 只需估计「状态 → 每个动作的即时收益」，是回归问题，比时序差分学习的样本效率高得多。
- **可解释**：线性或树模型的 bandit、成交概率曲线，都能直接拿给交易员和模型验证看。
- **什么时候需要多步 RL**：撤单重挂、队列位置的期权价值显著时，一步模型会低估「排队等待」的价值。那时再上离线 RL，用行为克隆初始化，用离线策略评估（OPE）和模拟器双重评估。

### 分阶段路线

```
阶段 0  日志
        子单级：决策时刻的状态、动作、倾向得分、成交明细、撤单、markout(1s/5s/30s)、清扫成本
        在现行规则里加安全集内的随机化（ε ≈ 2–10%）
阶段 1  监督模块 + 一步决策
        成交概率 P_fill(a, h | x)：生存分析，或处理删失的分类器
        markout 模型、清扫成本模型
        策略 = argmax 期望节省；和「信号规则版」比
阶段 2  Contextual bandit（宏动作）
        动作 = {TAKE, JOIN_L1, IMPROVE, POST_DEEP} × 持有 τ 秒后清扫
        离线学习和评估：IPW / DR（Open Bandit Pipeline）；奖励模型用线性或 GBDT
阶段 3  离线 RL（可选）
        d3rlpy：DiscreteBC → DiscreteCQL / DiscreteBCQ
        评估：FQE / DR-OPE（SCOPE-RL）+ 校准过的模拟器
阶段 4  模拟器内的在线 RL（研究）
        Maskable PPO / DDQN（Stable-Baselines3、CleanRL），参数随机化
        蒸馏成深度 4–6 的决策树，作为候选上线形态
```

## 4. 模拟器

### 历史回放的局限

| 问题 | 后果 | 缓解 |
|---|---|---|
| 没有自身冲击 | 低估吃单成本 | 叠加冲击核（瞬时冲击 + 衰减）；限制单笔不超过一档量的某个比例 |
| 不知道队列位置 | **高估被动成交**，尤其是假设「挂了就排第一」 | L3 数据精确追踪；L2 用概率队列模型，同时报告保守模型的结果 |
| 没有反事实 | 评估不了「当时如果挂在 X 档」 | 交互式模拟器 |
| 逆向选择缺失 | 高估被动成交的收益 | 用真实子单的 markout 校准；回放里用「穿价成交」而不是「触价成交」 |
| 延迟 | 高估抢到一档的能力 | 用实测延迟分布 |
| 非平稳 | 过拟合 | 滚动样本外；跨时期压力测试 |

### 模拟器分层

| 层 | 类型 | 代表 | 能模拟什么 | CPU |
|---|---|---|---|---|
| 1 | 回放 + 队列模型 + 延迟 | hftbacktest | 队列位置、延迟；没有冲击 | 很快 |
| 2 | 回放 + 冲击叠加 | 自己写传播子 / 衰减核 | 近似冲击 | 快 |
| 3 | Queue-reactive 类 | Huang、Lehalle、Rosenbaum（2015）；Noble、Rosenbaum、Souilmi（2026）的扩展（有 C++ 代码） | 订单流强度依赖当前队列状态，会对我们的单子做出反应；2026 版复现凹冲击和部分回复 | 快 |
| — | Hawkes 过程 | `tick` 库；状态依赖 Hawkes（Morariu-Patrichi 和 Pakkanen 2022） | 订单流的聚集和自激 | 中 |
| — | 解析 SDE 环境 | mbt_gym | 只用来检查 RL 管线能不能复现解析最优解 | 很快 |
| — | 多主体 | ABIDES、ABIDES-Gym | 完全交互，适合压力测试；校准难、慢 | 慢 |
| — | GPU 回放 | JAX-LOB | 有撮合，背景订单流是回放 | 需要 GPU |
| — | 生成式 / 世界模型 | LOBS5、TRADES、MarS | 条件生成订单流，可对注入的订单做出反应 | 需要 GPU，研究跟踪用 |

综述：Jain、Firoozye、Kochems、Treleaven（2024），*Limit Order Book Simulations: A Review*。

### 怎么验证模拟器

- **分布层面（stylized facts）**：价差、深度、OBI 分布，事件间隔，收益厚尾，波动聚集，订单流符号的长记忆，日内 U 型。指标可以直接借 LOB-Bench（Nagy 等 2025）。
- **冲击响应（对执行最重要）**：单笔市价单后的价格响应曲线；元订单冲击的平方根律、凹性和完成后的部分回复；被动单的成交概率曲线和真实子单对比。
- **sim-to-real 校验（上线门槛）**：
  1. **自身订单回放**：把历史上真实发出的子单注入模拟器，比较成交率、成交时间分布和 markout 曲线。
  2. **策略排名一致性**：3–5 个已有实盘结果的规则策略，在模拟器里的排名要和实盘一致（Spearman 相关）。
  3. **敏感性**：扰动队列模型、冲击系数和延迟，结论应当不变。
  4. **参数随机化**：训练时随机化冲击强度、撤单率、延迟，避免过拟合单一校准值。

## 5. 评估与上线

```
离线回测（回放 + 队列模型）
  → 模拟器评估（含敏感性）
  → 离线策略评估（IPW / DR / FQE，需要随机化日志）
  → 影子运行（只记录会怎么做，不下单）
  → 小流量随机对照
  → 逐步放量 + 持续监控
```

**指标**：

- 母单级：相对到达价的执行差（IS）、相对区间 VWAP、完成率、进度带越界次数、成本的 95% 分位或 CVaR。
- 子单级：被动成交比例、价差捕获、markout（1s / 5s / 30s）、清扫成本、撤单率、平均挂单时长。
- 按股票流动性、价差（大 / 小 tick）、时段、波动状态分层报告。

### 要多少订单才能测出差异

每组需要的母单数（双侧 α = 5%，power = 80%）：`n = 2 × (1.96 + 0.84)² × σ² / Δ²`，Δ 是想测出的成本差异

| 单笔成本标准差 σ | Δ = 1 bp | Δ = 2 bp | Δ = 3 bp |
|---|---|---|---|
| 10 bp | 1,570 | 392 | 174 |
| 30 bp | 14,128 | 3,532 | 1,570 |
| 50 bp | 39,244 | 9,811 | 4,360 |

让试验可行的办法：

1. **主指标换成子单级的**（价差捕获、markout、清扫成本），母单 IS 只当护栏（不显著变差即可）。
2. **换基准**：相对区间 VWAP 或决策时刻中间价，剔除大部分市场漂移。
3. **按母单随机化**；按子单或切片随机化时，用按母单聚类的稳健标准误，有效样本量按 `1 + (m − 1)ρ` 折算。
4. **分层随机化 + CUPED / 回归调整**，用事前可得的变量（股票、规模、价差、波动、时段）。
5. **序贯检验**（如 mSPRT）做护栏，提前停掉明显变差的组。

### 模型风险和监管（通用）

- 算法交易监管（如欧盟 MiFID II RTS 6）要求：上线前有开发和测试方法、测试环境与生产隔离、受控部署、年度自评、压力测试、重大变更管理、kill 功能、事前和实时风控。
- 模型风险管理（如英国 PRA SS1/23；美国 SR 11-7，二手资料称 2026 年 4 月已被新指引取代，原文未核对）要求独立验证、持续监控、有效挑战。
- **再训练包络**：事先书面定义哪些变化算常规再训练（同特征、同结构、同动作、同奖励、窗口滚动、超参数在预定范围内），走自动验证门槛 + 签批；包络外的变化按实质性更新处理。
- **冠军-挑战者**：新版本先影子运行，再小流量对照，再替换。
- **监控**：状态分布、动作分布、成本、信号校准度的漂移；超阈值自动回退到规则策略。
- **硬约束和模型分离**：进度带、价格保护、频率限制、kill switch 都在模型之外。

## 6. 算力

| 组件 | 规模 | CPU（量级估计，未实测） |
|---|---|---|
| 监督模块（GBDT / 逻辑回归） | 10^6–10^7 样本 | 分钟级 |
| Bandit 离线学习 + OPE | 10^5–10^7 决策 | 分钟级 |
| 离线 RL（2 层 MLP，256 单元） | 10^6 转移 | 小时级 |
| DQN / PPO 在快速模拟器上 | 10^6–10^7 步 | 够用，瓶颈在模拟器速度，要多核并行 |
| DQN / PPO 在 ABIDES 上 | 每个 episode 模拟全天多主体 | 慢，只适合评估和小实验 |
| JAX-LOB、生成式模型 | — | 需要 GPU，不可行 |

## 7. 风险和对策

| 风险 | 表现 | 对策 |
|---|---|---|
| 模拟器偏差 | 被动成交过于乐观、冲击被低估 | 保守队列模型并报告区间；自身订单回放；排名一致性；参数随机化 |
| 覆盖度不足 | 日志由确定性规则产生，评估不了新动作 | 尽早加安全集内随机化，记录倾向得分 |
| 信号漂移 | 信号失效时 RL 过度依赖 | 输入信号的近期校准度；信号失效时回退规则 |
| 奖励漏洞 | 挂深档等便宜价，或频繁撤挂 | 清扫成本进奖励；撤单频率进动作屏蔽 |
| 统计功效不足 | 测不出 1 bp 改进 | 子单级指标、方差缩减、聚类稳健推断 |

开源仓库见 [`repos.md`](repos.md)。

## 参考文献

**执行 RL**

- Nevmyvaka, Feng, Kearns (2006). Reinforcement learning for optimized trade execution. *ICML 2006*, 673–680. https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf
- Hendricks, Wilcox (2014). A reinforcement learning extension to the Almgren-Chriss model for optimal trade execution. *IEEE CIFEr*. https://arxiv.org/abs/1403.2229
- Bacoyannis, Glukhov, Jin, Kochems, Song (2018). Idiosyncrasies and challenges of data driven learning in electronic trading. *NeurIPS 2018 Workshop*. https://arxiv.org/abs/1811.09549 （立场文章：非平稳、低信噪比、模拟器、奖励设计）
- Ning, Lin, Jaimungal (2021). Double deep Q-learning for optimal execution. *Applied Mathematical Finance* 28(4):361–380. https://arxiv.org/abs/1812.06600
- Lin, Beling (2020). An end-to-end optimal trade execution framework based on proximal policy optimization. *IJCAI 2020*. https://www.ijcai.org/Proceedings/2020/627
- Karpe, Fang, Ma, Wang (2020). Multi-agent reinforcement learning in a realistic limit order book market simulation. *ICAIF 2020*. https://arxiv.org/abs/2006.05574
- Vyetrenko, Xu (2019). Risk-sensitive compact decision trees for autonomous execution in presence of simulated market response. https://arxiv.org/abs/1906.02312
- Fang 等 (2021). Universal trading for order execution with oracle policy distillation. *AAAI 2021*. https://arxiv.org/abs/2103.10860
- Zhang 等 (2023). Towards generalizable reinforcement learning for trade execution. *IJCAI 2023*. https://arxiv.org/abs/2307.11685
- Pan 等 (2022). Learn continuously, act discretely: hybrid action-space RL for optimal execution. https://arxiv.org/abs/2207.11152
- Fabre, Ragel (2023). Interpretable ML for high-frequency execution. https://arxiv.org/abs/2307.04863
- Hafsi, Vittori (2024). Optimal execution with reinforcement learning. https://arxiv.org/abs/2411.06389
- Cheridito, Weiss (2025). Reinforcement learning for trade execution with market and limit orders. https://arxiv.org/abs/2507.06345

**解析基线**

- Almgren, Chriss (2001). Optimal execution of portfolio transactions. *Journal of Risk*. DOI 10.21314/JOR.2001.041
- Cartea, Jaimungal (2015). Optimal execution with limit and market orders. *Quantitative Finance* 15(8):1279–1291.
- Cartea, Jaimungal (2016). Incorporating order-flow into optimal execution. *Mathematics and Financial Economics* 10:339–364.
- Lehalle, Neuman (2019). Incorporating signals into optimal trading. *Finance and Stochastics* 23(2):275–311. https://arxiv.org/abs/1704.00847
- Cartea, Jaimungal, Penalva (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

**模拟器**

- Huang, Lehalle, Rosenbaum (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110:107–122. https://arxiv.org/abs/1312.0563
- Noble, Rosenbaum, Souilmi (2026). Bridging the reality gap in limit order book simulation. https://arxiv.org/abs/2603.24137
- Bacry, Mastromatteo, Muzy (2015). Hawkes processes in finance. https://arxiv.org/abs/1502.04592
- Morariu-Patrichi, Pakkanen (2022). State-dependent Hawkes processes and their application to limit order book modelling. *Quantitative Finance* 22(3):563–583. https://arxiv.org/abs/1809.08060
- Byrd, Hybinette, Balch (2019). ABIDES. https://arxiv.org/abs/1904.12066
- Amrouni 等 (2021). ABIDES-Gym. *ICAIF 2021*. https://arxiv.org/abs/2110.14771
- Jerome, Sánchez-Betancourt, Savani, Herdegen (2023). Model-based gym environments for limit order book trading. *ICAIF 2023*. https://arxiv.org/abs/2209.07823
- Frey 等 (2023). JAX-LOB. *ICAIF 2023*. https://arxiv.org/abs/2308.13289
- Nagy 等 (2023). Generative AI for end-to-end limit order book modelling (LOBS5). *ICAIF 2023*. https://arxiv.org/abs/2309.00638
- Nagy 等 (2025). LOB-Bench. *ICML 2025*. https://arxiv.org/abs/2502.09172
- Berti, Prenkaj, Velardi (2025). TRADES. *ECAI 2025*. https://arxiv.org/abs/2502.07071
- Li 等 (2025). MarS: a financial market simulation engine powered by generative foundation model. *ICLR 2025*. https://arxiv.org/abs/2409.07486
- Jain, Firoozye, Kochems, Treleaven (2024). Limit order book simulations: a review. https://arxiv.org/abs/2402.17359

**离线 RL**

- Kumar, Zhou, Tucker, Levine (2020). Conservative Q-learning for offline RL. *NeurIPS 2020*.
- Fujimoto, Meger, Precup (2019). Off-policy deep RL without exploration (BCQ). *ICML 2019*.
- Kostrikov, Nair, Levine. Offline RL with implicit Q-learning (IQL). （会议年份未核实）

未核实：Vyetrenko 等 *Get Real* 的会议信息；MDPI *FinTech* 2(3):23（2023）一篇执行 RL 论文的作者；SR 11-7 被取代一事的监管原文。
