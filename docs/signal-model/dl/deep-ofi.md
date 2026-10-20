# Deep OFI：用订单流做输入、多 horizon 回归

Kolm、Turiel、Westray（2023），*Deep order flow imbalance: extracting alpha at multiple horizons from the limit order book*，*Mathematical Finance* 33(4):1044–1081。返回 [深度学习总览](../deep-learning.md)。

**一句话**：不把原始盘口价量直接喂给网络，先把它变成**平稳的逐档订单流**（OF / OFI），再用普通的 LSTM 类网络**回归多个 horizon 的中间价收益**（「alpha 期限结构」）。结论：订单流输入显著好于原始盘口；有效预测 horizon 约为 **2 次平均价格变动**；「信息丰富」的股票更可预测。

> **可得性说明**：这篇论文的全文没能打开（出版社和 SSRN 都拒绝访问）。下面的内容来自摘要、引言的转载，以及几篇复现和引用它的论文（Lucchese 等 2024；Briola 等 2025；arXiv:2311.02088）。**网络的层数和单元数、回看窗口长度、horizon 网格、学习率和 batch、$R^2$ 的具体数值都没能核实，一律不填数字。**它在这里的价值是「输入表示」和「目标设计」的思路，而不是具体网络。

## 1. 为什么选它

- **它回答的是我们最关心的问题**：输入怎么表示、预测什么、多长 horizon 有效。三条结论都被独立研究证实：
  - Lucchese 等（2024）在同一批股票上复现：订单流输入在 88%–89% 的情形下进入最优模型集，原始盘口输入（DeepLOB）只有 5%–11%；
  - 另一项研究在中国市场上发现，除 MLP 外，各基线模型用 OFI 输入都好于原始盘口。
- **回归而不是分类**：直接预测收益向量，避开了平滑标签和阈值。
- **按股票自己的时间尺度定义 horizon**：不同流动性的股票可以比较。

## 2. 主要创新点

1. **平稳输入**：原始价量的水平是非平稳的，网络很难学；订单流是「变化量」，天然平稳。
2. **多 horizon 回归（alpha 期限结构）**：一个网络同时输出 $K$ 个 horizon 的收益预测，看信号在多长时间内有效、多快衰减。
3. **horizon 用「平均价格变动间隔」缩放**：不同股票在同一个缩放后的 horizon 上，经历的价格变动次数大致相同。
4. **大横截面**：115 只 NASDAQ 股票、2019 年全年、不降采样，逐只训练，再对结果做横截面回归，看哪些股票特征决定可预测性。

## 3. 输入

第 $l$ 档（$l = 1, \dots, 10$）在第 $t$ 个盘口事件后的状态 $(p^{(l)}_{a,t}, v^{(l)}_{a,t}, p^{(l)}_{b,t}, v^{(l)}_{b,t})$。逐档订单流（Cont、Kukanov、Stoikov 2014 推广到多档）：

$$
\mathrm{bOF}^{(l)}_t = \begin{cases}
v^{(l)}_{b,t} & p^{(l)}_{b,t} > p^{(l)}_{b,t-1} \\
v^{(l)}_{b,t} - v^{(l)}_{b,t-1} & p^{(l)}_{b,t} = p^{(l)}_{b,t-1} \\
-v^{(l)}_{b,t-1} & p^{(l)}_{b,t} < p^{(l)}_{b,t-1}
\end{cases}
\qquad
\mathrm{aOF}^{(l)}_t = \begin{cases}
-v^{(l)}_{a,t-1} & p^{(l)}_{a,t} > p^{(l)}_{a,t-1} \\
v^{(l)}_{a,t} - v^{(l)}_{a,t-1} & p^{(l)}_{a,t} = p^{(l)}_{a,t-1} \\
v^{(l)}_{a,t} & p^{(l)}_{a,t} < p^{(l)}_{a,t-1}
\end{cases}
$$

$$
\mathrm{OFI}^{(l)}_t = \mathrm{bOF}^{(l)}_t - \mathrm{aOF}^{(l)}_t
$$

**直觉**：买一价上移，新的买一量全部算作新增买压；价格不变时看量的增减；价格下移，原来的买一量全部算作消失的买压。卖方对称，符号相反。

**三种输入表示对比**：

| 表示 | 每个时刻的维数 | 说明 |
|---|---|---|
| 原始盘口 | 40 | 10 档 × {卖价, 卖量, 买价, 买量} |
| OF | 20 | 每档 $(\mathrm{aOF}^{(l)}, \mathrm{bOF}^{(l)})$，买卖两路分开 |
| OFI | 10 | 每档一个数 |

**归一化**：用训练集上逐特征的均值和标准差做 z-score：$\tilde x_{t,j} = (x_{t,j} - \hat\mu_j^{\text{train}}) / \hat\sigma_j^{\text{train}}$。（Lucchese 等的复现改成了滚动窗口统计量，我们也建议用滚动版，见 [`features-and-targets.md`](../features-and-targets.md) 第 5 节。）

## 4. 输出和 horizon

不平滑的中间价收益：

$$
r_{t, t+h} = \frac{m_{t+h} - m_t}{m_t},\qquad m_t = \tfrac{1}{2}\big(p^{(1)}_{a,t} + p^{(1)}_{b,t}\big)
$$

网络输出一个向量 $\hat{\mathbf{r}}_t = (\hat r_{t, t+h_1}, \dots, \hat r_{t, t+h_K})$，最后一层是线性映射到 $\mathbb{R}^K$。horizon 按每只股票的平均价格变动间隔缩放（这个形式是根据二手资料推断的）：

$$
h^{(s)}_k = c_k\,\bar\Delta^{(s)},\qquad \bar\Delta^{(s)} = \frac{\text{交易时长}}{\#\{\text{中间价变动}\}}
$$

**评价指标**是样本外 $R^2$，每个 horizon 单独算：

$$
R^2_{\mathrm{OS}, h} = 1 - \frac{\sum_t (r_{t,t+h} - \hat r_{t,t+h})^2}{\sum_t (r_{t,t+h} - \bar r_h)^2}
$$

$\bar r_h$ 是基准（股票的平均收益）。$R^2_{\mathrm{OS}} > 0$ 说明比「预测平均值」好。

## 5. 模型和训练

论文比较了 ARX、MLP、LSTM、LSTM-MLP、三层 stacked LSTM、CNN-LSTM（类似 DeepLOB）六种网络，每种都配三种输入。据引用文献转述，CNN-LSTM 最好，其次是 LSTM-MLP、LSTM、stacked LSTM；摘要的说法是「简单的现成网络」在平稳输入上就能达到前沿水平。

| 项 | 已知信息 |
|---|---|
| 损失 | 多输出 MSE（由 $R^2_{\mathrm{OS}}$ 的定义推断） |
| 正则 | L2 正则 + 早停 |
| 样本外方案 | 滚动窗口：4 周训练、1 周验证、1 周测试，覆盖 48 周 |
| 建模粒度 | 每只股票单独训练 |
| 层数、单元数、窗口长度、学习率、batch | **未核实** |

## 6. 主要结论和怎么分析

1. **订单流输入胜过原始盘口**：「models trained on order flow significantly outperform most models trained directly on order books」。
2. **有效 horizon 约为 2 次平均价格变动**：精度在这附近见顶，之后衰减。**这决定了我们的 τ 不该太长**，也说明信号对延迟很敏感。
3. **「信息丰富」的股票更可预测**。信息丰富度（Briola 等给出的定义）：

$$
\mathrm{IR} = \log\frac{\#\{\text{盘口更新}\}}{\#\{\text{价格变动}\}}
$$

   每次价格变动之间有越多的盘口更新，网络就有越多信息可用。Briola 等指出 **IR 和 tick 大小几乎一一对应**：大 tick 股票 IR 高（例如 5.16），小 tick 股票低（1.4–2.2）。所以「信息丰富」和「大 tick」说的是同一件事。
4. **第一个 horizon 的 $R^2$ 可能为负**（引用文献在其他市场复现时也观察到）。

**我们怎么用这些结论分析自己的模型**：

- 画出 $R^2_{\mathrm{OS}, h}$（或我们的 log loss 技能分）随 $h / \bar\Delta$ 变化的曲线，找到峰值位置，用来定执行的决策间隔和 τ。
- 按 IR（或 tick 大小）分组看技能分，决定信号在哪些股票上启用。

## 7. 已知问题

- **一手细节缺失**：用之前应取得出版社正式版核对。
- **时钟不一致**：输入用事件时钟，horizon 用物理时间，样本里实际包含的事件数随日内活跃度变化。
- **收益不可交易**：中间价到中间价的收益没有扣价差，有效 horizon 又只有约 2 次价格变动，对延迟要求极高。
- **$R^2_{\mathrm{OS}}$ 对基准敏感**：基准用训练期均值还是测试期均值，结果会不同。

## 8. 在我们的场景里怎么用

- **输入**：完全采用它的做法，逐档 OF / OFI 作为网络的主输入。这就是我们推荐网络 [OFI-TCN-GRU](ofi-tcn-gru.md) 的输入。
- **输出**：把「多 horizon 回归」换成「多 horizon 首达分类 + 条件幅度」，因为执行决策要的是概率和成本；回归的期望收益可以作为辅助头。
- **评估**：沿用「按 $h / \bar\Delta$ 画技能曲线」和「按 IR 分组」的分析方法。

## 9. 代码

只实现能核实的部分：逐档订单流、训练集 z-score、多 horizon 回归头、$R^2_{\mathrm{OS}}$。LSTM 的宽度、窗口 $W$、horizon 数 $K$ 是**占位值**。

```python
import numpy as np
import torch
import torch.nn as nn


def order_flow(pa, va, pb, vb):
    """Multi-level order flow. pa, va, pb, vb: (N, L) arrays, one row per order-book update.
    Returns bOF, aOF: (N-1, L); OFI = bOF - aOF."""
    pb0, pb1, vb0, vb1 = pb[:-1], pb[1:], vb[:-1], vb[1:]
    pa0, pa1, va0, va1 = pa[:-1], pa[1:], va[:-1], va[1:]
    bOF = np.where(pb1 > pb0, vb1, np.where(pb1 == pb0, vb1 - vb0, -vb0))
    aOF = np.where(pa1 > pa0, -va0, np.where(pa1 == pa0, va1 - va0, va1))
    return bOF, aOF


class OFLSTM(nn.Module):
    """LSTM regression on order-flow windows -> alpha term structure.
    x: (B, W, F), F = 2L (OF) or L (OFI); returns (B, K) return forecasts.
    hidden / num_layers / W / K are placeholders, not taken from the paper."""

    def __init__(self, n_features=20, hidden=64, num_layers=1, n_horizons=10):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        seq, _ = self.lstm(x)
        return self.head(seq[:, -1])


def r2_os(y, y_hat, y_bench):
    """R^2_OS per horizon. y, y_hat: (N, K); y_bench: (K,)."""
    return 1 - ((y - y_hat) ** 2).mean(0) / ((y - y_bench) ** 2).mean(0)


if __name__ == '__main__':
    rng = np.random.default_rng(0); N, L, W, K = 2000, 10, 100, 10
    mid = 100 + np.cumsum(rng.integers(-1, 2, N)) * 0.01
    pb = mid[:, None] - 0.005 - 0.01 * np.arange(L); pa = mid[:, None] + 0.005 + 0.01 * np.arange(L)
    vb = rng.integers(1, 50, (N, L)) * 100.0; va = rng.integers(1, 50, (N, L)) * 100.0
    bOF, aOF = order_flow(pa, va, pb, vb)
    X = np.stack([aOF, bOF], -1).reshape(N - 1, 2 * L)
    X = (X - X[:1000].mean(0)) / (X[:1000].std(0) + 1e-8)               # z-score with train statistics
    win = torch.tensor(np.stack([X[i - W:i] for i in range(W, 1500)]), dtype=torch.float32)
    print(OFLSTM(2 * L, n_horizons=K)(win).shape)                        # (1400, 10)
```

## 参考文献

- Kolm, Turiel, Westray (2023). Deep order flow imbalance: extracting alpha at multiple horizons from the limit order book. *Mathematical Finance* 33(4):1044–1081. https://doi.org/10.1111/mafi.12413 ；SSRN 3900141
- Cont, Kukanov, Stoikov (2014). The price impact of order book events. *J. Financial Econometrics* 12(1):47–88.
- Xu, Gould, Howison (2019). Multi-level order-flow imbalance in a limit order book. https://arxiv.org/abs/1907.06230
- Lucchese, Pakkanen, Veraart (2024). The short-term predictability of returns in order book markets: a deep learning perspective. *International Journal of Forecasting* 40(4):1587–1621. https://arxiv.org/abs/2211.13777
- Briola, Bartolucci, Aste (2025). Deep limit order book forecasting: a microstructural guide. *Quantitative Finance* 25(7):1101–1131. https://arxiv.org/abs/2403.09267
- arXiv:2311.02088（转述了原文的 $R^2_{\mathrm{OS}}$ 定义和滚动窗口设定）
- Kolm, Westray. Improving deep learning of alpha term structures from the order book. SSRN 4770476.（后续工作：简单 LSTM 很难被多头 LSTM 或 Seq2Seq 超越）
