# 深度学习：总览、通用设计和训练

2026-10 调研整理。从 GBDT 往深度学习走时，选哪几个模型、所有网络共用的设计原则、训练时看什么指标、怎么和 GBDT 公平比较。每个模型的具体设计（数学、逐层结构、参数怎么选、怎么分析、可运行的代码）在子页面里。总览见 [`modeling-guide.md`](modeling-guide.md)。运行环境：Intel i9 CPU，没有 GPU。

## 结论

1. **公开基准上的高分基本不可信，也不可迁移。** TransLOB 在 FI-2010 上自报 F1 约 87%，统一复现只有 59%；换到新的 NASDAQ 数据后，15 个模型的 F1 全部落在 48%–61%（LOBCAST）。
2. **能不能预测，主要看股票的微观结构，网络结构影响小得多。** 大 tick 股票 10 个事件的 MCC 约 0.29，小 tick 约 0.11（Briola 等 2025）。真实数据上模型之间只差 1–5 个 F1 点；输入表示、标签、按股票分层、评估口径造成的差距是 10–30 个点。
3. **输入表示比结构更重要。** 逐档 OFI 稳定好于原始 10 档价量（Kolm 等 2023；Lucchese 等 2024）。
4. **我们的推荐是 [OFI-TCN-GRU](dl/ofi-tcn-gru.md)**：约 5 万参数，可以逐事件增量推理，纯 numpy 单步约 160 µs。**[BiN-CTABL](dl/bin-ctabl.md) 当深度学习基线**，[DeepLOB](dl/deeplob.md) 和 [TLOB](dl/tlob.md) 当对照。
5. **i9 CPU 够用**：这几个小网络在 CPU 上训练是小时级；TLOB 这类百万参数的模型训练会慢很多，但仍然可行。

## 1. 精选的模型

| 模型 | 代表什么 | 输入 | 参数 | CPU 推理（实测，单线程 batch 1） | 结论 | 页面 |
|---|---|---|---|---|---|---|
| **DeepLOB**（2019） | 原始盘口 + CNN + LSTM，事实上的标准基线 | $100 \times 40$ 原始价量 | 14.2 万（官方代码） | 5.5 ms | 原文 FI-2010 F1 83.4，复现 71.1；NASDAQ 约 59.5 | [详细](dl/deeplob.md) |
| **BiN-CTABL**（2019 / 2020） | 双线性网络 + 双向归一化，最轻、最稳 | $10 \times 40$ 原始价量 | 1.1 万 | 0.42 ms | 复现结果和原文一致；稳健性最高 | [详细](dl/bin-ctabl.md) |
| **Deep OFI**（2023） | 订单流输入 + 多 horizon 回归 | 逐档 OF / OFI 序列 | 未核实 | — | 订单流输入显著好于原始盘口；有效 horizon 约 2 次价格变动 | [详细](dl/deep-ofi.md) |
| **TLOB**（2025） | 双注意力 Transformer，最新 | $128 \times 40$ | 114 万 | 9.4 ms | 短 horizon 上纯 MLP 就够了，长 horizon 上注意力才有优势；加入价差阈值后 F1 明显下降 | [详细](dl/tlob.md) |
| **OFI-TCN-GRU**（推荐） | 综合以上结论 | 逐档 OFI + 相对价格 + 事件元数据 | 5.0 万 | 整窗 5.7 ms；**流式单步 0.16 ms（numpy）** | 待在自有数据上验证 | [详细](dl/ofi-tcn-gru.md) |

（推理时间在 4 核 Xeon 2.8 GHz、PyTorch 2.5 上测得，i9 单核会更快。DeepLOB 和 TLOB 每来一个事件都要整窗重算；OFI-TCN-GRU 只算一步。）

没有收进来的：TransLOB（复现差）、DLA（种子间方差大）、Axial-LOB、LiT（相对 DeepLOB 提升不到 1 个点）、Mamba / 状态空间模型（没有找到用于 LOB 方向预测的同行评审工作）、Chronos / TimesFM 这类基础模型（没有 LOB 方向任务的公开基准，2 亿参数和亚毫秒推理不相容）、图神经网络（跨资产信息用「滞后 OFI 当 GBDT 特征」性价比更高）。

## 2. 基准为什么不可信

**FI-2010**：5 只芬兰股票 10 个交易日，已经预处理、归一化、打好标签，**无法还原原始盘口，也就没法回测**；标签是平滑标签，类别比例随 horizon 漂移（$k = 1$ 时上 / 平 / 下 = 18 / 63 / 19，$k = 10$ 时 37 / 25 / 38）；论文之间切分方式混用。

**LOBCAST**（Prata 等 2024）统一复现 15 个模型：多个模型的复现结果远低于自报；换到新数据后全部明显变差；约一半的超参组合发散（F1 ≤ 33%）；前三名都不用长窗口；集成没有超过最好的单模型；回测显示盈利远没有保证。

**Briola 等（2025）**：

| Horizon（事件数） | 小 tick MCC | 中 tick MCC | 大 tick MCC |
|---|---|---|---|
| 10 | ~0.11 | ~0.13 | ~0.29 |
| 50 | ~0.04 | ~0.085 | ~0.36 |
| 100 | ~0.01 | ~0.036 | ~0.26 |

大 tick 股票好预测的原因：最优档队列厚，**成交前队列会先变薄**，这本身就是方向信息。他们还定义了「完整交易正确率」

$$
p_T = \frac{C_T}{P_T + T_T - C_T}
$$

（$C_T$ 是开平仓都对的交易数，$P_T$ 是目标序列里的潜在交易数，$T_T$ 是预测执行出来的交易数），实测只有 0.04–0.15。**提高置信度阈值能抬高 MCC，却会把 $p_T$ 压到接近 0。**

## 3. 所有网络共用的设计原则

### 3.1 标签

用首达三分类（τ 内先涨 ≥ 1 tick / 先跌 ≥ 1 tick / 没动），τ 用物理时间、等于执行的决策间隔；幅度以「动了」为条件、涨跌分开。不用 FI-2010 那种平滑标签。定义见 [`features-and-targets.md`](features-and-targets.md) 7.1 节。

### 3.2 输入表示（按优先级）

| 方案 | 维数 / 事件 | 评价 |
|---|---|---|
| **逐档 OFI / OF** | 10–20 | 平稳，直接编码供需变化，证据最强。**主输入** |
| 相对中间价的价格（以 tick 计）+ $\log(1 + \text{量})$ | 20–40 | 平移不变，跨股票可共享 |
| 以中间价为锚的量网格（按 tick 距离排列） | 20 | 空间均匀，小 tick 股票优先（Lucchese 等 2024） |
| 事件元数据 | 5–10 | 事件类型、$\log \Delta t$、成交方向和规模、价差 |
| 静态上下文 | 10–30 | 时段、日内波动、跨资产滞后 OFI、股票属性；在输出头之前拼接 |
| 原始 10 档价量 | 40 | 非平稳，只作对照 |

### 3.3 归一化

- 按股票、按日滚动：用过去 $N$ 天（如 5 天）的统计量做 z-score，**不用当日或未来的统计量**。
- 量类先 $\log(1 + x)$，再用中位数和四分位距做稳健标准化，截断到 $\pm 5$。
- OFI 除以滚动平均深度。
- 可以在网络里再加一层 BiN（[BiN-CTABL](dl/bin-ctabl.md) 4.3 节）。
- 多股票共用一个模型时加股票 embedding。

### 3.4 损失

样本按唯一性加权的交叉熵，**不加类权重、不用 focal loss**（两者都破坏校准，而执行层直接用概率算成本）：

$$
\mathcal{L} = -\frac{1}{\sum_i w_i}\sum_i w_i \sum_j \log \hat\pi_{j, y_{ij}}(x_i) + \lambda_{\text{mag}}\,\mathcal{L}_{\text{mag}}
$$

### 3.5 切分

按日：训练 → 空出 purge → 早停验证 → 校准 → 测试，walk-forward 滚动。**早停用的验证集和校准窗口要分开。**

## 4. 训练时看什么指标

### 4.1 训练过程

| 指标 | 异常迹象 |
|---|---|
| 训练 / 验证 loss（总的、分 horizon、分任务） | 训练 loss 一直降、验证 loss 1–3 个 epoch 后回升：在记日内模式，LOB 上很常见 |
| 分任务 loss 占比 | 某个头的 loss 不降：权重失衡或标签有问题 |
| 梯度范数 | 爆炸或 NaN |
| 预测类别分布 | 塌缩到「没动」 |
| 种子间方差（至少 3 个种子） | 大于 2 个 log loss 技能点：结论不可靠 |

### 4.2 预测质量（每个 horizon 单独报）

$$
\text{Skill}_h = 1 - \frac{\mathrm{LL}_{\text{model}, h}}{\mathrm{LL}_{\text{prior}, h}},\qquad
\mathrm{IC}_h = \operatorname{Spearman}\big(\hat\pi^\uparrow - \hat\pi^\downarrow,\ \Delta m_{t \to t+h}\big)
$$

$\mathrm{LL}_{\text{prior}}$ 用训练期的类频率。IC 先按日算，再看均值、$t$ 值、正值天数占比。另外看 MCC、macro-F1、Brier、classwise-ECE（定义见 [`ml-pipeline.md`](ml-pipeline.md) 3.1 节），以及置信度前 $k\%$ 子集上的 precision 和信号翻转频率。

### 4.3 稳定性

- 按日的指标序列、滚动 20 日均值；看有没有在特定日期崩坏。
- 按 tick 类别、流动性、时段、波动分层。**可以只在可预测的子集上启用深度学习信号。**
- 定期重训，比较「新模型」和「旧模型继续用」，估计衰减速度。

### 4.4 和 GBDT 公平比较

1. 样本、标签、切分、采样时钟完全一致；GBDT 拿到相同的信息（把序列手工汇总成多尺度 OFI 累计等特征）。
2. 主指标：各 horizon 的技能分和按日 IC。**按日**做 block bootstrap 或 Diebold-Mariano 检验：设 $d_t = \mathrm{LL}^{\text{GBDT}}_t - \mathrm{LL}^{\text{DL}}_t$ 是第 $t$ 天的损失差，

$$
\mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2_{\text{HAC}}(d) / n_{\text{days}}}} \ \overset{a}{\sim}\ \mathcal{N}(0, 1)
$$

   不能把逐样本当独立（样本高度相关）。
3. 增量检验：「GBDT + 网络输出当特征」对比「GBDT」。增量显著的话，网络不单独上线也有价值。
4. 在固定的 p99 延迟预算内比。
5. **上线门槛是执行模拟里的成本改善**（含逆向选择和成交概率），不是 F1。

## 5. i9 CPU 上的训练和推理

| 模型 | 参数 | 训练（$10^7$ 样本、30 epoch，量级估计） | 线上推理 |
|---|---|---|---|
| BiN-CTABL | 1.1 万 | 1–2 小时 | 整窗，< 0.5 ms |
| OFI-TCN-GRU | 5.0 万 | 几小时以内 | 流式单步，numpy 0.16 ms，C++ 预计几十微秒 |
| DeepLOB | 14.2 万 | 半天到一天 | 整窗，约 5 ms |
| TLOB | 114 万 | 一天以上；建议先用 $H = 32$、$T = 64$、$L = 2$ 的小版本 | 整窗，约 9 ms |

- `torch.set_num_threads(物理核数)`；数据预先存成按日分区的 parquet / numpy 内存映射，避免数据加载成为瓶颈。
- 样本高度重叠，每隔 5–10 个事件取一个窗口，训练量降一个数量级，信息损失很小。
- 混合精度在 CPU 上收益有限，用 fp32。
- 推理用 numpy 或 C++ 手写前向；ONNX Runtime 的动态 int8 量化可以试，但要和 fp32 比 log loss。

## 6. 推进顺序

1. **数据层**：事件级盘口重建 → 逐档 OFI、相对价格、量网格 → 首达标签 → 按日分区。GBDT 和深度学习共用。
2. **GBDT 基线**（[`ml-pipeline.md`](ml-pipeline.md)）。
3. **深度学习 v1**：[OFI-TCN-GRU](dl/ofi-tcn-gru.md)，同时跑 [BiN-CTABL](dl/bin-ctabl.md) 作基线。
4. **融合**：网络输出当 GBDT 特征，检验增量；按 tick 类别 / 时段决定是否启用。
5. **上线**：流式推理；影子运行；按日监控，定期重训。
6. **第二阶段（可选）**：[TLOB](dl/tlob.md) 的小版本，看长 τ 上注意力有没有增量。

## 参考文献

- Prata 等 (2024). LOB-based deep learning models for stock price trend prediction: a benchmark study (LOBCAST). *Artificial Intelligence Review* 57. https://arxiv.org/abs/2308.01915
- Briola, Bartolucci, Aste (2025). Deep limit order book forecasting: a microstructural guide. *Quantitative Finance* 25(7):1101–1131. https://arxiv.org/abs/2403.09267
- Lucchese, Pakkanen, Veraart (2024). The short-term predictability of returns in order book markets: a deep learning perspective. *International Journal of Forecasting* 40(4):1587–1621. https://arxiv.org/abs/2211.13777
- Kolm, Turiel, Westray (2023). Deep order flow imbalance. *Mathematical Finance* 33(4):1044–1081.
- Sirignano, Cont (2019). Universal features of price formation in financial markets: perspectives from deep learning. *Quantitative Finance* 19(9):1449–1459. https://arxiv.org/abs/1803.06917
- Ntakaris 等 (2018). Benchmark dataset for mid-price forecasting of limit order book data (FI-2010). *Journal of Forecasting*. https://arxiv.org/abs/1705.03233
- Wu, Mahfouz, Magazzeni, Veloso (2021). Towards robust representation of limit orders books for deep learning models. https://arxiv.org/abs/2110.05479
- DeLise (2024). The negative drift of a limit order fill. https://arxiv.org/abs/2407.16527
- Diebold, Mariano (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics* 13(3):253–263.
- 各模型的文献见子页面。
