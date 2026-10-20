# 执行层：信号怎么用

2026-10 整理。监督信号做出来以后，怎么接进执行算法：信号决定什么、进度带怎么放、阈值怎么来、分步怎么上线。建模本身见 [`modeling-guide.md`](modeling-guide.md)，成本公式见 [`ml-pipeline.md`](ml-pipeline.md) 2.8 节。

## 场景假设

- 做的是**执行**，不是方向性交易：母单的买卖方向给定，当天做完。
- 执行有进度上下限 $[L(t), U(t)]$（已成交比例 $p(t)$ 满足 $L(t) \le p(t) \le U(t)$）。带子里面，算法可以自己决定快慢和下单方式。
- 所以问题不是「会涨还是会跌」，而是「**对我这一边，现在做便宜，还是等一等便宜**」。

> 这里把上下限理解成**执行进度**的区间。如果是**参与率**的上下限，第 2 节的规则要相应改写。

## 1. 信号决定什么

信号不决定买卖方向，只决定两件事：这一刻用什么方式下单（吃单、挂在最优价、挂得更深、撤单），以及在进度带里加快还是放慢。

| 母单方向 | 信号：马上要涨 | 信号：马上要跌 | 信号：短时间不会动 |
|---|---|---|---|
| 买入 | 吃单，或者加快 | 挂单，或者撤了往下挂，或者放慢 | 价格层面等待不吃亏；挂不挂取决于成交概率 |
| 卖出 | 挂单，或者撤了往上挂，或者放慢 | 吃单，或者加快 | 同上 |

关于最后一列：「没动」只说明等待的**价格成本**低，不说明挂单会成交。完全安静时市价单少、队列长，挂单的成交概率往往很低，挂了不成交就会落后于进度。所以要配一个成交模型，比较 $q_0\,(s/2 - f_m)$ 和 $(1 - q_0)\,c_{\text{wait}}$。

信号的价值主要在于避开「挂单刚成交，价格就往不利方向走」（逆向选择）。

## 2. 进度带怎么放进去

以买单为例（卖单对称）：

| 位置 | 规则 |
|---|---|
| 带子中间、离下限远 | 按成本决策层来（见下一节） |
| 接近下限或快收盘 | 落后的代价 c_wait 变大，信号弱也要吃单 |
| **接近上限（已经超前）** | 不吃单。信号对母单有利（买单预测下跌）时，**撤掉被动单等更低的价**，因为这时挂着的单会被逆向选择成交，而且本来就不急；信号不利（预测上涨）时保留被动单 |
| 跌破下限 | 统一主动成交，补回进度 |

- 这和 Markov（2012）的「不确定带」做法一致：带子上方用便宜的、机会性的方式，带子下方逐渐变积极，跌破下限就统一主动成交。Cartea 和 Jaimungal（2015）也是平时挂单赚价差、进度落后时用市价单补。
- **硬约束写在代码里，模型之外。** 不管信号怎么说，都不能越过进度带和事前风控的限制。
- **防操纵**：盘口挂单量可以被人为制造（挂假单）。要限制单次盘口读数对进度的影响幅度。
- 没有找到直接处理「硬性进度带」的论文，上面这些是最接近的。

## 3. 阈值：从成本推，不用固定值

原则：对买单，**等待的预期成本大于马上吃单的成本，就吃单**。

公式统一用 [`ml-pipeline.md`](ml-pipeline.md) 2.8 节的版本：吃单成本 $C_{\text{take}} = s/2 + f_t\ (+\ \text{impact})$，挂单成本按「先涨 / 先跌 / 没动」三个结局加权，每个结局下再乘以该结局的成交概率 $q_k$：

$$
C_{\text{post}} \approx \sum_{k \in \{\uparrow, \downarrow, 0\}} p_k\Big[q_k\big(-\tfrac{s}{2} + f_m\big) + (1 - q_k)\,\big(\text{结局 } k \text{ 下没成交时的追价成本}\big)\Big]
$$

- 所以阈值不是固定的 0.5，它随价差、排队长度、手续费、离进度带边界的远近而变。
- 进度的影响通过 $c_{\text{wait}}$ 进入：越接近下限、越接近收盘，$c_{\text{wait}}$ 越大，相当于 Cont 和 Kukanov 的「落后进度惩罚」$\lambda$，也就是进度带约束的影子价格。一个简单的参数化：

$$
c_{\text{wait}}(t) = c_0 \cdot \exp\Big(-\kappa\,\frac{p(t) - L(t)}{U(t) - L(t)}\Big) \cdot \frac{1}{\max(T_{\text{close}} - t,\ \epsilon)}
$$

$c_0, \kappa$ 用历史执行数据校准；进度越靠近下限、离收盘越近，等待越贵。

## 4. 延迟

信号的价值随延迟很快衰减（Lehalle 和 Mounjid）。要测清楚从收到行情到发出订单要多久，并把这个延迟放进回测：特征只用决策时刻已经到达的信息，成交按「延迟之后的盘口」撮合。

## 5. 评估

- **对照组不是 TWAP**，而是同一套执行逻辑、不用信号的版本。
- **主检验用子单级指标**：被动成交的 markout（1 s / 5 s / 30 s）、价差捕获、清扫成本，相对决策时刻中间价。**母单 IS 当护栏**，做非劣效检验（母单 IS 方差太大，测 1 bp 差异要上万笔母单）。
- 还要看完成率和进度带越界次数。
- **剥离当天价格走势**：两种执行方式的差异常常主要来自当天的价格走势，要把它单独拆出来。
- **历史回放会高估挂单的效果**：回放里我们的订单不影响价格，也不改变排队位置。要用带队列位置和延迟的模拟器，最终靠小规模实盘对照确认（见 [`reinforcement-learning.md`](reinforcement-learning.md) 第 4 节）。

## 6. 分步计划

和 [`modeling-guide.md`](modeling-guide.md) 第 8 节一致：

| 步骤 | 做什么 | 怎么判断做成了 |
|---|---|---|
| 1 | 数据层 + 换标签：中间价首达三分类（物理时间 τ = 决策间隔）；旧特征、多项 LR | 和旧模型比经济指标和信号十分位图 |
| 2 | 加特征：样条 + 交互 LR，校准，成本决策层 + 进度带规则 | 同一标签下 log loss 技能分 > 0；仿真成本改善 |
| 3 | LightGBM、竞争风险 hazard、成交模型、幅度模型 | 稳定好于第 2 步；成交模型在真实成交上校准合格 |
| 4 | 影子运行，再拿小部分订单随机对照 | 子单级指标显著改善，母单 IS 不显著变差 |
| 5（可选） | 强化学习决定下单方式 | 显著好于「同一信号 + 成本决策层」 |

## 7. 落地前要确认的事

1. 进度上下限是执行进度的区间，还是参与率的区间？
2. 算法多久做一次决定？这决定 τ。
3. 从收到行情到发出订单，延迟要求多少？
4. 有没有自己的子单和成交记录？能不能算挂单成交后的 markout？
5. 算法能不能撤单重挂、挂到更深的档位？撤单频率有没有限制？
6. 手续费和返佣是多少？主要交易的股票价差一般几档？

## 参考文献

- Cartea, Jaimungal (2015). Optimal execution with limit and market orders. *Quantitative Finance* 15(8):1279–1291.
- Cartea, Jaimungal (2016). Incorporating order-flow into optimal execution. *Mathematics and Financial Economics* 10(3):339–364.
- Cartea, Donnelly, Jaimungal (2018). Enhancing trading strategies with order book signals. *Applied Mathematical Finance* 25(1):1–35.
- Lehalle, Neuman (2019). Incorporating signals into optimal trading. *Finance and Stochastics* 23(2):275–311. https://arxiv.org/abs/1704.00847
- Lehalle, Mounjid. Limit order strategic placement with adverse selection risk and the role of latency. https://arxiv.org/abs/1610.00261
- Cont, Kukanov. Optimal order placement in limit order markets. https://arxiv.org/abs/1210.1625
- Markov (2012). On the design of sell-side limit and market order tactics. *Journal of Trading* 7(3). https://arxiv.org/abs/1409.1442
- Fabre, Ragel (2023). Interpretable ML for high-frequency execution. https://arxiv.org/abs/2307.04863
- Huang, Lehalle, Rosenbaum (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110(509):107–122.
