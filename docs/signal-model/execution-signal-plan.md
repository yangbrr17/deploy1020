# 执行用的 OBI / OFI 信号：建模方案

2026-10 调研后整理。指标定义和特征清单见 [`obi-ofi-basics.md`](obi-ofi-basics.md)。文献都核对过出处（期刊、年份、链接），少数打不开原文的标了「未核实」。

## 场景假设

- 做的是**执行**，不是方向性交易：母单的买卖方向是给定的，当天必须做完。
- 执行有进度的上下限，比如进度不低于 25%、不高于 75%。在这个带子里，算法可以自己决定快慢和下单方式。
- 所以问题不是「会涨还是会跌」，而是「对我这一边，现在做便宜，还是等一等便宜」。

> 这里把 25% / 75% 理解成**执行进度**的上下限。如果其实是**参与率**的上下限，第 5 节要改。

## 结论

1. **信号不决定买还是卖，买卖方向由母单定。** 信号决定两件事：这一刻用什么方式下单（吃单、挂在买一卖一、挂得更深、撤单），以及在进度带里加快还是放慢。
2. **标签改成「τ 内中间价先涨一档 / 先跌一档 / 没动」。** 这样「大多数 tick 价格不变」的问题自然就没了，同时有了方向和时间。
3. **阈值从成本推出来，不再固定。** 对买单，等待的预期成本高于马上吃单的成本，就吃单。阈值随价差、排队长度、离进度带边界多远而变。
4. **涨和跌分开处理。** 信号的价值主要在于避开「挂单刚成交，价格就往不利方向走」。
5. **模型先用 LightGBM，CPU 就够。** 概率要校准。评估看执行成本，不看准确率；对照组是不用信号的同一套逻辑，不是 TWAP。
6. **强化学习以后可以用在下单方式的小决策上**，前提是先有可信的模拟器和严格的测试。全天的进度安排不交给它。

## 1. 信号在执行里决定什么

| 母单方向 | 信号：马上要涨 | 信号：马上要跌 | 信号：短时间不会动 |
|---|---|---|---|
| 买入 | 吃单，或者加快 | 挂单，或者撤了往下挂，或者放慢 | 挂在买一，等成交 |
| 卖出 | 挂单，或者撤了往上挂，或者放慢 | 吃单，或者加快 | 挂在卖一，等成交 |

最后一列：在执行里，「短时间不会动」不是没用的信息。它说明等待不吃亏，可以安心挂单赚半个价差。

## 2. 标签用中间价，不用成交价

成交价在买一和卖一之间来回跳，会制造假的涨跌（Roll 1984）。标签用中间价，或者用自己要成交的那一边的价格：买单看卖一，卖单看买一。

## 3. 预测目标

### 把方向和时间放在一个标签里

在算法下一次做决定之前的这段时间 τ 里，看哪件事先发生：

- 中间价先涨一档；
- 中间价先跌一档；
- τ 结束了都没动。

这相当于「三重障碍」标签：上下各一档，加一条时间线（López de Prado 2018）。

和其他做法的关系：

- 只看「下一次中间价变动是涨还是跌」，是 Gould 和 Bonart（2016）的做法。它在事件时间里只统计价格真的动了的时刻，所以不存在「大多数不变」的问题，但它不告诉你多久会动。
- 加上 τ 以后，同时有了方向和时间。
- 更严格的写法是「竞争风险」模型：买一被吃光和卖一被吃光，看哪个先发生。Cont 和 de Larrard（2013）用排队模型给出了「下一次是涨的概率」和「距离下一次变动的时间」的公式，可以当对照。

### 幅度

- 在「动了」的前提下，再预测动多少：1 档、2 档、3 档以上，分几类。
- 或者用分位数回归给出一个区间。LightGBM 有现成的分位数目标；要完整的分布可以用 NGBoost。
- **涨和跌的幅度分开建模。** 对买单，涨是成本，跌只有在挂单真的成交时才是收益，两边的价值不对称。Zhang、Zohren、Roberts（2019）就用买卖两边各自的价格分别做了分位数预测。

### 多个时长

τ 取几个值：1、2、5 次中间价变动，再加 1 秒、5 秒、30 秒。Kolm、Turiel、Westray（2023）在 115 只纳斯达克股票上发现，这类信号的有效预测长度大约是 2 次平均价格变动，再往后基本是噪声。

## 4. 阈值怎么定

**原则：** 对买单，等待的预期成本大于马上吃单的成本，就吃单。

```
马上吃单的成本 ≈ 半个价差 + 手续费 + 冲击

等待的预期成本 ≈ P(先涨) × 涨幅
                − P(挂单成交) × 半个价差
                + 挂单成交后价格继续不利的损失（逆向选择）
                + 落后于进度带的代价
```

- 这些量都来自模型：涨的概率和幅度、挂单成交的概率、成交后的价格走势（markout）。
- **所以阈值不是固定的 0.5 或者别的数**，它随价差、排队长度、手续费、离进度带边界的远近而变。
- 现成的公式模板：Cont 和 Kukanov 推导了挂单和吃单怎么分配，里面有一个「落后进度的惩罚」参数 λ。这里 λ 就是进度带的影子价格：越接近下限、越接近收盘，λ 越大。

## 5. 进度带怎么放进去

- **在带子中间、离下限远：** 按信号来。
- **接近下限或者快收盘：** 落后的代价变大，信号弱也要吃单。
- **接近上限：** 只挂单；信号不利时撤单等待。
- 这和 Markov（2012）的「不确定带」做法一致：带子上方用便宜的、机会性的方式，带子下方逐渐变积极，跌破下限就统一主动成交。Cartea 和 Jaimungal（2015）也是平时挂单赚价差、进度落后时用市价单补。
- **硬约束写在代码里，模型之外。** 不管信号怎么说，都不能越过进度带和事前风控的限制。
- **防操纵：** 盘口挂单量可以被人为制造（挂假单）。要限制单次盘口读数对进度的影响幅度。

没有找到直接处理「硬性进度带」的论文，上面这些是最接近的。

## 6. 模型

- **主模型：** LightGBM。每个 τ 一个三分类（先涨 / 先跌 / 没动），幅度另一个模型。
- **概率要校准：** 阈值直接用到概率，所以要用留出的交易日做校准（isotonic 或 Platt）。
- **按股票类型分开看：** 价差常年只有一档的股票（大价位股票）信号最有用；价位相对价格很小的股票效果弱很多（Gould 和 Bonart 2016；Xu、Gould、Howison 2019）。
- **所有股票共用一个模型：** 特征标准化以后共用一个模型，见 [`pooled-model.md`](pooled-model.md)。
- **延迟：** 信号的价值随延迟很快衰减（Lehalle 和 Mounjid；Stoikov 和 Waeber 估算，约 1 毫秒的延迟下，每股能比 TWAP 省约三分之一个价差）。要测清楚从收到行情到发出订单要多久。
- **深度模型以后再说：** LOBCAST 基准测试了 15 个深度模型，换到新数据上效果都明显下降，单个样本的推理时间约 0.7–3.9 毫秒（Prata 等 2024）。

## 7. 怎么评估

- **按交易日滚动：** 用前面的日子训练，后面的日子测试。标签有时间窗口，切分时要在训练集和测试集之间留空档（purge / embargo）。
- **诊断指标：** AUC、对数损失或 Brier 分数、概率校准（预测 60% 时是否真的约 60% 会涨）、按股票类型拆开看。
- **主要的验收标准是执行成本：**
  - 用了信号以后，被动成交在 1 秒、5 秒、30 秒后的价格走势（markout）有没有变好。
  - 相对到达价的执行差。
  - 完成率，以及有没有守住进度带。
- **对照组不是 TWAP**，而是同一套执行逻辑、不用信号的版本。
- **剥离当天价格走势。** 比较两种执行方式时，差异常常主要来自当天的价格走势，而不是冲击成本。要把走势的贡献单独拆出来，否则会把运气当成算法的效果。
- **历史回放会高估挂单的效果：** 回放里我们的订单不会影响价格，也不会改变排队位置。要估计这部分，需要基于排队的模拟器（queue-reactive 模型，Huang、Lehalle、Rosenbaum 2015），最终靠小规模实盘对照确认。

## 8. 强化学习

**用来预测方向和幅度，不适合。** 这是监督学习的问题：有现成的标签。强化学习解决的是「一连串决策，每一步的动作会影响后面的状态和收益」。用它做预测，样本效率更低、结果更不稳定。

**用来做执行决策，可以考虑，但是第二阶段的事。** 它应该是**使用信号**的那一层，不是产生信号的那一层。

**研究证据：**

- 收益通常是几个 bp，比较的基准大多是 TWAP 这类弱基准。
- 结果多数来自不考虑自身冲击的历史回放或模拟器。
- 没有找到公开的实盘对照结果。
- 业界公开的做法（如 Bacoyannis 等 2018）是用强化学习决定限价单放在哪里，动作范围有限，外面套着常规的排程和风控。

**限制：**

1. **样本少。** 全天的订单每只股票每年只有大约 250 个「回合」，所以只能用在几秒到几分钟的小决策上，多只股票合起来训练。
2. **算力不是问题，模拟器才是。** 简单的强化学习用 CPU 几分钟就能训练，小型神经网络几个小时。真正花时间的是一个可信的市场模拟器：历史回放反映不了自身订单的冲击和排队位置。
3. **监管要求严。** 算法交易监管（如欧盟 MiFID II RTS 6）一般要求上线前充分测试、改动前重新测试、保留设计记录。强化学习每重新训练一次，都可能算一次重大改动。模型风险管理（如 PRA SS1/23、SR 11-7）也要求独立验证和可解释。

**建议的顺序：**

- **现在：** 用信号直接改进基于规则的下单方式，这一步可能就拿到大部分收益。
- **之后的前提：** 有自己的子单和成交记录；能测出 1 bp 级别差异的 TCA；考虑排队位置的回放；用自己的成交校准冲击模型；和「同样用信号的规则版本」比，而不只是和 TWAP 比。
- **再之后：** 简单的强化学习，离线训练后固定下来，不在线学习。先影子运行，再拿小部分订单随机对照。
- **最后：** 只有在实盘里显著好于规则版本时，才考虑更复杂的强化学习。

## 9. 分步计划

| 步骤 | 做什么 | 怎么判断做成了 |
|---|---|---|
| 1 | 换标签：中间价，τ 内先涨 / 先跌 / 没动。模型还是逻辑斯蒂回归，特征不变 | 和原模型在同一批测试日上比较 |
| 2 | 加特征，换 LightGBM，校准概率，多个 τ | 对数损失、校准、按股票类型的结果都变好 |
| 3 | 幅度模型；用成本推出阈值，加入进度带逻辑 | 用历史成交估算 markout 和执行差，和不用信号的版本比 |
| 4 | 影子运行，再拿小部分订单随机对照 | 实盘的执行差有统计上显著的改善 |
| 5（可选） | 用强化学习决定下单方式 | 显著好于第 4 步的规则版本 |

## 10. 落地前要确认的事

1. 进度上下限是执行进度的上下限，还是参与率的上下限？
2. 算法多久做一次决定，下一次决定前的时间大概多长？这决定 τ 取多少。
3. 从收到行情到发出订单，延迟要求多少？
4. 有没有自己的子单和成交记录？能不能算挂单成交后的价格走势？
5. 算法能不能撤单重挂、挂到更深的档位？
6. 手续费和返佣是多少？主要交易的股票价差一般几档？
7. 原模型具体用了哪些特征、阈值是怎么定的、在什么数据上评估的？

## 参考文献

**信号怎么用于执行**

- Cartea, Jaimungal (2016). Incorporating order-flow into optimal execution. *Mathematics and Financial Economics* 10(3):339–364. https://ora.ox.ac.uk/objects/uuid:88a23aea-ceb6-4bb0-8b59-fd594d8db92c
- Cartea, Donnelly, Jaimungal (2018). Enhancing trading strategies with order book signals. *Applied Mathematical Finance* 25(1):1–35.
- Cartea, Jaimungal (2015). Optimal execution with limit and market orders. *Quantitative Finance* 15(8):1279–1291. https://ideas.repec.org/a/taf/quantf/v15y2015i8p1279-1291.html
- Lehalle, Neuman (2019). Incorporating signals into optimal trading. *Finance and Stochastics* 23(2):275–311. https://arxiv.org/abs/1704.00847
- Lehalle, Mounjid. Limit order strategic placement with adverse selection risk and the role of latency. https://arxiv.org/abs/1610.00261 （期刊信息未核实）
- Cont, Kukanov. Optimal order placement in limit order markets. https://arxiv.org/abs/1210.1625 （期刊信息未核实）
- Markov (2012). On the design of sell-side limit and market order tactics. *Journal of Trading* 7(3). https://arxiv.org/abs/1409.1442
- Stoikov, Waeber. Reducing transaction costs with low-latency trading algorithms. （期刊信息未核实）

**标签、幅度、时间**

- Gould, Bonart (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. *Market Microstructure and Liquidity* 2(2). https://arxiv.org/abs/1512.03492
- Cont, de Larrard (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Mathematics* 4(1):1–25. https://arxiv.org/abs/1104.4596
- Kolm, Turiel, Westray (2023). Deep order flow imbalance. *Mathematical Finance* 33(4):1044–1081. https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html
- Xu, Gould, Howison (2019). Multi-level order-flow imbalance in a limit order book. https://arxiv.org/abs/1907.06230
- Cont, Kukanov, Stoikov (2014). The price impact of order book events. *J. Financial Econometrics* 12(1):47–88. https://arxiv.org/abs/1011.6402
- Stoikov (2018). The micro-price. *Quantitative Finance* 18(12):1959–1966.
- Zhang, Zohren, Roberts (2019). Extending deep learning models for limit order books to quantile regression. https://arxiv.org/abs/1906.04404
- Briola, Bartolucci, Aste (2025). *Quantitative Finance* 25(7):1101–1131. https://arxiv.org/html/2403.09267v2
- Huang, Lehalle, Rosenbaum (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110(509):107–122.
- Arroyo, Cartea, Moreno-Pino, Zohren (2024). 挂单成交时间的生存分析. *Quantitative Finance* 24(1):35–57.
- Engle, Russell (1998). Autoregressive conditional duration. *Econometrica* 66(5):1127–1162.
- Roll (1984). 买卖价差引起的价格负相关. *Journal of Finance*.
- López de Prado (2018). *Advances in Financial Machine Learning*.
- Prata 等 (2024). LOBCAST 基准. *Artificial Intelligence Review*. https://arxiv.org/html/2308.01915v2

**强化学习**

- Nevmyvaka, Feng, Kearns (2006). Reinforcement learning for optimized trade execution. *ICML*.
- Ning, Lin, Jaimungal (2021). Double deep Q-learning for optimal execution. *Applied Mathematical Finance*. https://ar5iv.arxiv.org/html/1812.06600
- Lin, Beling (2020). *IJCAI*. https://www.ijcai.org/Proceedings/2020/0627.pdf
- Bacoyannis 等 (2018). 限价单放置的强化学习引擎. https://arxiv.org/abs/1811.09549
- Hambly, Xu, Yang (2023). Recent advances in reinforcement learning in finance. *Mathematical Finance* 33(3). https://arxiv.org/abs/2112.04553
- ABIDES-Gym. https://arxiv.org/abs/2110.14771 （原仓库 2025 年 6 月已归档）
