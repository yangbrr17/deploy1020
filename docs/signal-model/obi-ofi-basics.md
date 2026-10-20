# OBI / OFI：定义和特征

2026-10 整理。tick 级盘口信号的基本量和特征清单。标签、阈值、模型、评估见 [`execution-signal-plan.md`](execution-signal-plan.md)。

## 起点和问题

常见的起点：tick 级数据，用逻辑斯蒂回归预测「下一笔成交来的时候价格涨还是跌」，用一个固定阈值分涨跌。问题：

1. 只有方向，没有幅度；涨跌的不对称也没区分。
2. 没有衡量「下一步」是多久之后。
3. 大多数 tick 价格不变，二分类把「不变」硬塞进涨或跌。

## 两个指标

缩写容易混：一般 **OFI** 指 Order Flow Imbalance，**OBI**（也叫 queue imbalance）指 Order Book Imbalance。

### Order Book Imbalance（盘口不平衡）

某一时刻盘口的状态。最简单的只看买一卖一：

```
OBI = (买一挂单量 − 卖一挂单量) ÷ (买一挂单量 + 卖一挂单量)      取值 −1 到 1
```

买一的量远多于卖一，下一次中间价更可能往上走。可以扩展到前几档，越远的档位权重越小。

### Order Flow Imbalance（订单流不平衡）

一段时间内盘口的变化，来自 Cont、Kukanov、Stoikov（2014）。每来一个盘口事件，算一次：

```
e = [买一价没降] × 新的买一量 − [买一价没升] × 旧的买一量
  − [卖一价没升] × 新的卖一量 + [卖一价没降] × 旧的卖一量
```

方括号里成立取 1，否则取 0。把一段时间里的 e 加起来就是这段时间的 OFI。直观意思：买方挂单增加、卖方挂单被吃掉或撤掉，都算买方压力。

结论：一段时间内中间价的变化和 OFI 大致成线性关系，斜率和盘口深度成反比。所以 **OFI 除以平均深度** 之后，不同股票可以放在一起比较，这是「所有股票共用一个模型」的基础（见 [`pooled-model.md`](pooled-model.md)）。

### 微观价格（microprice）

用盘口不平衡给中间价加权，比中间价更接近「下一刻的公允价」。简单版本：

```
微观价格 = 买一价 × 卖一量 / (买一量 + 卖一量) + 卖一价 × 买一量 / (买一量 + 卖一量)
```

Stoikov（2018）给了更严格的定义。

## 预测目标的几个原则

1. **预测中间价，不要预测成交价。** 成交价在买一和卖一之间来回跳（bid-ask bounce）。「下一笔成交价比这一笔高」很多时候只是因为下一笔是主动买，不是价格真的涨了。准确率看起来很高，但没用。成交方向单独当一个特征。
2. **方向和时间一起定。** 具体做法（τ 内先涨 / 先跌 / 没动）见 [`execution-signal-plan.md`](execution-signal-plan.md) 第 3 节。
3. **幅度的几种做法：**
   - 按价位档数分类：跌 2 档及以上 / 跌 1 档 / 不变 / 涨 1 档 / 涨 2 档及以上。
   - 两段式：先预测「会不会动」，再预测「往哪动、动多少」。
   - 直接回归价格变化（以价位档或 bp 为单位），或者用分位数回归给出一个区间。
4. **时间的几种做法：**
   - 同时预测几个时间长度：事件时间（下 1、5、20 笔）和时钟时间（1 秒、5 秒、30 秒），看信号多快衰减。
   - 单独预测「距离下一次价格变动还有多久」：持续时间模型（ACD，Engle 和 Russell 1998）或 Hawkes 过程。
   - 把时间间隔当特征：距离上一笔成交过了多久、最近事件来得多快。

## 特征

在逻辑斯蒂回归里先逐类加，看每一类带来多少提升：

- 多档 OBI：前 1、3、5 档。
- 多个窗口的 OFI：最近 10、50、200 个事件，以及最近 1 秒、5 秒、30 秒。除以平均深度做标准化。
- 买卖价差（以价位档为单位）。价差是 1 档和大于 1 档时，OBI 的作用很不一样。
- 微观价格减中间价。
- 最近成交的方向和主动买卖量的差（成交方向用 Lee-Ready 方法判断）。
- 撤单速度、买一卖一被吃掉的速度。
- 最近的波动率、距离上一个事件的时间、一天里的时段。
- 股票层面的特征（给通用模型用）：流动性、波动率、价位相对价格的大小、平均价差档数。

## 模型阶梯

| 顺序 | 模型 | 说明 |
|---|---|---|
| 0 | 逻辑斯蒂回归，换新标签，加上面的特征 | 基准。OBI + 逻辑回归本身就是文献里的标准基准（Gould 和 Bonart 2016） |
| 1 | LightGBM / XGBoost | 只用 CPU，训练和预测都快。能抓住非线性，比如价差不同时 OBI 的作用不同。可以做多分类、回归、分位数回归 |
| 2 | 多个时间长度一起预测 | Kolm、Turiel、Westray（2023）发现用 OFI 做输入比直接用原始盘口好，并且同时预测多个时间长度 |
| 3 | 深度模型（如 DeepLOB，Zhang、Zohren、Roberts 2019） | 直接读原始盘口，基本需要显卡；换一段新数据后效果下降明显。放到最后 |

## 数据上的坑

- **时间戳对齐：** 成交和报价可能来自不同的数据流，时间戳精度和延迟不同。只能用预测时刻之前已经到达的信息。
- **按时间切分：** 不要随机打乱，否则未来的信息会漏进训练集。
- **确认数据粒度：** 有几档盘口；有没有逐笔挂单、撤单记录（L3），还是只有快照；时间戳精确到多少。

## 参考文献

- Cont, Kukanov, Stoikov (2014). The price impact of order book events. *J. Financial Econometrics* 12(1):47–88. https://arxiv.org/abs/1011.6402
- Gould, Bonart (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. *Market Microstructure and Liquidity* 2(2). https://arxiv.org/abs/1512.03492
- Stoikov (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative Finance* 18(12):1959–1966.
- Xu, Gould, Howison (2019). Multi-level order-flow imbalance in a limit order book. https://arxiv.org/abs/1907.06230
- Kolm, Turiel, Westray (2023). Deep order flow imbalance: extracting alpha at multiple horizons from the limit order book. *Mathematical Finance* 33(4):1044–1081.
- Zhang, Zohren, Roberts (2019). DeepLOB: deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*.
- Engle, Russell (1998). Autoregressive conditional duration. *Econometrica* 66(5):1127–1162.
- López de Prado (2018). *Advances in Financial Machine Learning*.
