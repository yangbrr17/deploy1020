# DeepLOB：卷积 + Inception + LSTM

Zhang、Zohren、Roberts（IEEE TSP 2019），arXiv:1808.03668；多步变体 Zhang、Zohren（2021），arXiv:2105.10430。返回 [深度学习总览](../deep-learning.md)。

**一句话**：直接读最近 100 个盘口快照的 10 档价量（$100 \times 40$），用三组卷积依次「把每档的价和量配对 → 把买卖两侧合并 → 把 10 档整合成一列」，再用 Inception 做多尺度时间平滑，最后用 LSTM 抓时间依赖。**它是这个领域被引用最多、被复现最多的基线。**

## 1. 为什么选它

- **事实上的标准基线**：几乎所有后续论文都和它比；LOBCAST 在 2022 年 NASDAQ 数据上复现，它排第一（F1 59.5）。
- **结构设计有清晰的微观结构含义**（第 2 节），是理解「网络怎么读盘口」最好的例子。
- **代价**：输入是原始价量，非平稳；论文和官方代码有多处不一致（第 8 节），复现前要先选定版本；CPU 推理是毫秒级。

## 2. 主要创新点（每一层为什么这样设计）

| 设计 | 理由 |
|---|---|
| 只用原始 10 档价量 | 不依赖手工特征，由数据学 |
| 第 1 层卷积 $1 \times 2$、步长 $1 \times 2$ | 把每档的 $(p^{(i)}, v^{(i)})$ 配对汇总。不用步长的话，同一组参数会同时作用于 $(p^{(i)}, v^{(i)})$ 和 $(v^{(i)}, p^{(i+1)})$，含义错乱 |
| 第 2 个 $1 \times 2$、步长 $1 \times 2$ | 把同一档的卖方和买方合并，作者类比为「对每一档算一个 microprice」：$p^{\text{micro}} = I\,p_a + (1 - I)\,p_b$，$I = \frac{v_b}{v_a + v_b}$ |
| $1 \times 10$ 卷积 | 把 10 档整合成 1 列 |
| $4 \times 1$ 时间卷积 | 等价于 FIR 滤波器 $y(n) = \sum_{k=0}^{M} b_k\,x(n-k)$，做可学习的平滑 |
| 卷积块不用池化 | 实验显示池化会欠拟合 |
| Inception（$1\times1 \to 3\times1$、$1\times1 \to 5\times1$、池化 → $1\times1$） | 多尺度的「可学习移动平均」，自动学出不同衰减速度 |
| 用 LSTM(64) 代替全连接 | 64 单元全连接要 63 万参数，LSTM 约 6 万，还能刻画时间依赖 |
| 滚动 5 日 z-score | 应对金融序列的 regime shift |
| 迁移测试 | 在没参与训练的 5 只股票上精度相近，说明学到了跨股票的通用特征 |

## 3. 输入和标签

**输入**：

$$
X = [x_1, \dots, x_{100}]^\top \in \mathbb{R}^{100 \times 40},\qquad
x_t = \big[p_a^{(i)}(t),\ v_a^{(i)}(t),\ p_b^{(i)}(t),\ v_b^{(i)}(t)\big]_{i=1}^{10}
$$

$T = 100$ 个最近的盘口状态（事件计数，不是时钟时间）。PyTorch 张量形状 `(B, 1, 100, 40)`。LSE 数据用前 5 个交易日的均值和标准差逐特征 z-score：

$$
\tilde x_{d, j} = \frac{x_{d, j} - \mu_j^{(d-5:d-1)}}{\sigma_j^{(d-5:d-1)}}
$$

**标签**（多步论文附录的写法，$p_t$ 是中间价）：

$$
m_-(t) = \frac{1}{k}\sum_{i=0}^{k-1} p_{t-i},\qquad
m_+(t) = \frac{1}{k}\sum_{i=1}^{k} p_{t+i}
$$

$$
\text{FI-2010：}\ l_t = \frac{m_+(t) - p_t}{p_t},\qquad
\text{LSE：}\ l_t = \frac{m_+(t) - m_-(t)}{m_-(t)},\qquad
y_t = \begin{cases} +1 & l_t > \alpha \\ -1 & l_t < -\alpha \\ 0 & \text{其他} \end{cases}
$$

FI-2010 用 $\alpha = 0.002$；LSE 的 $\alpha$ 按训练集类别平衡选（如某只股票在 $k = 20/50/100$ 时为 $0.25/0.50/0.75 \times 10^{-4}$）。原论文 $m_-$ 的公式有 $k+1$ 项却除以 $k$，是笔误。

**我们的看法**：LSE 版本的 $m_-$ 正好是输入窗口里过去价格的平均，标签有一部分是「过去的趋势」，准确率会虚高（Lucchese 等的批评）。我们用首达三分类。

## 4. 数学

### 4.1 卷积层

二维卷积（输入通道 $C_{\text{in}}$，输出通道 $C_{\text{out}}$，核 $k_h \times k_w$）：

$$
Y_{c}(t, w) = b_c + \sum_{c'=1}^{C_{\text{in}}}\sum_{i=0}^{k_h-1}\sum_{j=0}^{k_w-1} K_{c, c'}(i, j)\, X_{c'}(t + i,\ s_w w + j)
$$

$s_w$ 是宽度方向的步长。激活一律是 LeakyReLU：$\phi(z) = \max(z, 0.01 z)$（斜率 0.01 由验证集网格搜索确定）。时间方向（$k_h = 4$）用 same padding，所以时间维保持 100。

### 4.2 Inception 模块

$$
\mathrm{Inc}(H) = \Big[\ \underbrace{\mathrm{Conv}_{3\times1}\big(\mathrm{Conv}_{1\times1}(H)\big)}_{\text{短窗平滑}}\ \Big\Vert\ \underbrace{\mathrm{Conv}_{5\times1}\big(\mathrm{Conv}_{1\times1}(H)\big)}_{\text{长窗平滑}}\ \Big\Vert\ \underbrace{\mathrm{Conv}_{1\times1}\big(\mathrm{MaxPool}_{3\times1}(H)\big)}_{\text{局部极值}}\ \Big]
$$

$\Vert$ 表示通道拼接。三个分支各 64 通道，拼成 192 通道。

### 4.3 LSTM

对 Inception 输出的序列 $z_1, \dots, z_T \in \mathbb{R}^{192}$：

$$
\begin{aligned}
i_t &= \sigma(W_i z_t + U_i h_{t-1} + b_i), & f_t &= \sigma(W_f z_t + U_f h_{t-1} + b_f), & o_t &= \sigma(W_o z_t + U_o h_{t-1} + b_o),\\
\tilde c_t &= \tanh(W_c z_t + U_c h_{t-1} + b_c), & c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t, & h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

取最后时刻 $h_T \in \mathbb{R}^{64}$，接 $\hat p = \operatorname{softmax}(W h_T + b)$。

### 4.4 Adam 的 $\epsilon = 1$

论文用 Adam、学习率 0.01、$\epsilon = 1$。Adam 的更新是

$$
\theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

当梯度量级远小于 1 时 $\sqrt{\hat v} \ll 1$，更新约为 $\eta\,\hat m$，**相当于学习率 0.01 的动量 SGD**。官方代码用的是学习率 $10^{-4}$、默认 $\epsilon \approx 10^{-7}$，两者实际步长可能差几个数量级。复现时不能只抄学习率。

## 5. 结构和参数量

官方 TF 代码（卷积 32 通道、Inception 64 通道），输出形状写作（时间, 宽, 通道）：

| # | 层 | 核 / 步长 | 通道 | 输出 | 参数 |
|---|---|---|---|---|---|
| 1 | Conv | $1\times2$ / $1\times2$ | 1 → 32 | (100, 20, 32) | $2 \cdot 32 + 32 = 96$ |
| 2–3 | Conv × 2 | $4\times1$ / 1 | 32 → 32 | (100, 20, 32) | $2 \times (4 \cdot 32 \cdot 32 + 32) = 8{,}256$ |
| 4 | Conv | $1\times2$ / $1\times2$ | 32 → 32 | (100, 10, 32) | 2,080 |
| 5–6 | Conv × 2 | $4\times1$ / 1 | 32 → 32 | (100, 10, 32) | 8,256 |
| 7 | Conv | $1\times10$ / 1 | 32 → 32 | (100, 1, 32) | 10,272 |
| 8–9 | Conv × 2 | $4\times1$ / 1 | 32 → 32 | (100, 1, 32) | 8,256 |
| 10 | Inception（3 个分支） | — | 32 → 3 × 64 | (100, 1, 192) | 39,232 |
| 11 | Dropout 0.2（只在代码里） | — | — | (100, 192) | 0 |
| 12 | LSTM | — | 192 → 64 | (64,) | $4[64(192 + 64) + 64] = 65{,}792$ |
| 13 | Dense + softmax | — | 64 → 3 | (3,) | 195 |
| | **合计** | | | | **142,435**（Keras 口径） |

卷积参数 $= k_h k_w C_{\text{in}} C_{\text{out}} + C_{\text{out}}$。论文图里画的宽度是卷积 16、Inception 32，按这个算是 60,691，对应论文写的「约 6 万」。**论文结果对应哪种宽度，从论文和代码都无法确定。**（PyTorch 的 LSTM 有两组偏置，同样结构多 256 个参数。）

多步变体（编码器同上，$T = 50$；解码器 LSTM 展开 5 步对应 5 个 horizon）：Seq2Seq 176,419 个参数，Attention 版 177,699 个，和论文表格一致。

## 6. 训练设置

| 项 | 论文 | 官方代码 | 我们建议 |
|---|---|---|---|
| 损失 | 交叉熵 | 交叉熵 | 交叉熵 × 唯一性权重，不加类权重 |
| 优化器 | Adam，lr 0.01，$\epsilon = 1$ | Adam，lr $10^{-4}$ | AdamW，lr $10^{-3}$，warmup + cosine；或复现时照搬论文 |
| Batch | 32 | 128 / 64 | 256 |
| 早停 | 验证准确率 20 个 epoch 不提升 | 保存验证 loss 最好的 | 验证 **log loss**，patience 5 |
| Epoch | FI-2010 约 100，LSE 约 40 | 200 / 50 | ≤ 30 |
| 硬件 | 1 块 Tesla P100 | — | i9 CPU 可以训练，但慢（第 7 节） |

## 7. 参数怎么选、结果怎么分析

**超参数和敏感性**：

- 论文只网格搜索了 LeakyReLU 的斜率。$T = 100$ 是直接设定的。
- 多步论文改用 $T = 50$，注意力权重集中在最近的观测上，说明 $T = 100$ 并非必要。
- LOBCAST 发现约一半的超参组合会发散（F1 ≤ 33%），前三名的模型都不用长度 100 的输入。
- Briola 等（2025）用 DeepLOB 做了 135 组实验：**可预测性主要由 tick 大小决定**，大 tick 股票 MCC 在 10 / 50 / 100 个事件上是 0.29 / 0.36 / 0.26，小 tick 是 0.11 / 0.04 / 0.01。

**报告的结果**（F1，%）：

| 数据 | 设定 | 结果 | 来源 |
|---|---|---|---|
| FI-2010 Setup 2 | k = 10 / 20 / 50 | 83.40 / 72.82 / 80.35 | 原文 |
| FI-2010 | k = 10 / 20 / 50 / 100 | 71.1 / 62.4 / 75.4 / 77.6 | LOBCAST 复现 |
| LSE 2017，训练过的 5 只 | k = 20 / 50 / 100 | 70.15 / 63.49 / 60.65 | 原文 |
| LSE 2017，没见过的 5 只 | k = 20 / 50 / 100 | 68.48 / 62.84 / 60.77 | 原文 |
| NASDAQ 2021 / 2022 | 平均 | 59.5 / 59.5 | LOBCAST |
| TSLA / INTC / BTC | h = 10 | 36.3 / 68.1 / 68.1 | TLOB 论文 |

多步变体在 FI-2010 的 k = 100 上：DeepLOB 76.76、Seq2Seq 79.16、Attention 81.49。短 horizon 上和 DeepLOB 相当，长 horizon 更好。

**怎么分析训练好的模型**：

- 看第 1、2 层卷积核的权重：价和量的系数符号是否符合直觉（买量大 → 偏向上涨）。
- 看 Inception 三个分支的输出方差，判断模型更依赖短窗还是长窗。
- **多步 Attention 版**：看注意力权重 $\alpha_{t,i}$ 在过去时刻上的分布，通常集中在最近几个时刻。

**推理时间**：论文单次前向 0.253 ms（GPU）。我们在 4 核 Xeon 2.8 GHz、PyTorch 2.5、单线程、batch 1 实测：**官方宽度 p50 5.5 ms / p99 9.1 ms；论文宽度（16/32）p50 4.6 ms**。LSTM 是逐步计算的，而且每来一个事件都要整窗重算 100 步，这是延迟的主要来源。

**CPU 训练时间**：14 万参数、$10^7$ 个窗口、每个 epoch 在 i9 上约 1–3 小时（量级估计）；用每隔 5–10 个事件取一个窗口降采样，可以快一个数量级。

## 8. 已知问题

1. **论文和代码不一致**：通道 16/32 vs 32/64；学习率 0.01（$\epsilon = 1$）vs $10^{-4}$；batch 32 vs 128/64；FI-2010 用 z-score 还是 decimal precision 归一化；早停依据。Dropout、BatchNorm、Tanh 只出现在代码里。
2. **官方 PyTorch notebook**：时间卷积没有 padding（时间维 100 → 82）；第 2 块激活换成 Tanh；每层后接 BatchNorm；`forward` 返回 softmax 后又送进 `CrossEntropyLoss`（双重 softmax）。
3. **TF 版 Dropout 在推理时也开着**（`training=True` 写死），预测不可复现。
4. **滑窗跨越股票或日期边界**：代码把多只股票拼起来直接滑窗。
5. **泛化差**：NASDAQ 数据上 F1 只有 55–62%，TSLA 约 35%。

## 9. 在我们的场景里怎么用

- **定位**：对照组。证明「原始盘口 + 大网络」不如「OFI 输入 + 小网络」，正好支撑我们的推荐设计。
- **不建议上线**：输入非平稳、毫秒级延迟、不能增量推理。
- **可借鉴**：「先配对价量、再合并买卖、最后整合各档」的卷积顺序；多步解码同时输出多个 horizon 的思路（我们的推荐网络用多头输出实现同样的效果）。
- **代码使用**：官方仓库没有 LICENSE 文件（默认保留所有权利），只读，自己重写。下面的实现是按论文和代码重写的。

## 10. 代码

```python
import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """DeepLOB (Zhang, Zohren, Roberts 2019).
    Defaults = official TF notebooks: conv 32ch, inception 64ch, LSTM 64, LeakyReLU(0.01) everywhere,
    'same' padding on temporal convs (T preserved), no BatchNorm, Dropout(0.2) shared over time before LSTM.
      conv_ch=16, inc_ch=32   -> widths drawn in the paper's figures (~60k params)
      pytorch_notebook=True   -> official PyTorch notebook: no temporal padding (T 100->82), Tanh in block 2,
                                 BatchNorm after every activation
    x: (B, 1, T, 40) ordered [p_a^1, v_a^1, p_b^1, v_b^1, ..., p_a^10, v_a^10, p_b^10, v_b^10]
    returns logits (B, 3) -> train with nn.CrossEntropyLoss, softmax only at inference."""

    def __init__(self, conv_ch=32, inc_ch=64, lstm_hidden=64, n_classes=3, dropout=0.2,
                 pytorch_notebook=False):
        super().__init__()
        nb, C, I = pytorch_notebook, conv_ch, inc_ch
        pad_t = 0 if nb else 'same'

        def unit(cin, cout, k, s=(1, 1), pad=0, tanh=False):
            layers = [nn.Conv2d(cin, cout, k, stride=s, padding=pad),
                      nn.Tanh() if tanh else nn.LeakyReLU(0.01)]
            return layers + ([nn.BatchNorm2d(cout)] if nb else [])

        # (B,1,T,40) -> 1x2/s2 -> (B,C,T,20) -> 4x1 -> 4x1           [pair price & volume per level]
        self.block1 = nn.Sequential(*unit(1, C, (1, 2), (1, 2)),
                                    *unit(C, C, (4, 1), pad=pad_t), *unit(C, C, (4, 1), pad=pad_t))
        # (B,C,T,20) -> 1x2/s2 -> (B,C,T,10) -> 4x1 -> 4x1           [pair ask & bid per level]
        self.block2 = nn.Sequential(*unit(C, C, (1, 2), (1, 2), tanh=nb),
                                    *unit(C, C, (4, 1), pad=pad_t, tanh=nb),
                                    *unit(C, C, (4, 1), pad=pad_t, tanh=nb))
        # (B,C,T,10) -> 1x10 -> (B,C,T,1) -> 4x1 -> 4x1              [integrate 10 levels]
        self.block3 = nn.Sequential(*unit(C, C, (1, 10)),
                                    *unit(C, C, (4, 1), pad=pad_t), *unit(C, C, (4, 1), pad=pad_t))
        # Inception: three branches, each (B,I,T',1)
        self.inc1 = nn.Sequential(*unit(C, I, (1, 1)), *unit(I, I, (3, 1), pad='same'))
        self.inc2 = nn.Sequential(*unit(C, I, (1, 1)), *unit(I, I, (5, 1), pad='same'))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)), *unit(C, I, (1, 1)))
        self.dropout = 0.0 if nb else dropout
        self.lstm = nn.LSTM(3 * I, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def features(self, x):
        x = self.block3(self.block2(self.block1(x)))                   # (B, C, T', 1)
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], 1)  # (B, 3I, T', 1)
        x = x.squeeze(-1).transpose(1, 2)                             # (B, T', 3I)
        if self.dropout > 0 and self.training:                        # one channel mask shared over time
            keep = 1 - self.dropout
            x = x * torch.bernoulli(torch.full((x.size(0), 1, x.size(2)), keep, device=x.device)) / keep
        return x

    def forward(self, x):
        seq, _ = self.lstm(self.features(x))  # (B, T', H)
        return self.fc(seq[:, -1])            # (B, 3) logits


if __name__ == '__main__':
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(DeepLOB()(torch.randn(4, 1, 100, 40)).shape, n(DeepLOB()))   # (4, 3) 142,691
    print(n(DeepLOB(conv_ch=16, inc_ch=32)))                          # 60,947
```

多步变体（编码器 = 上面的卷积和 Inception；解码器 LSTM 展开 5 步，Attention 版用 Luong 点积注意力 $e_{t,i} = h_i^\top h'_t$、$\alpha_{t,i} = \operatorname{softmax}_i(e_{t,i})$、$c_t = \sum_i \alpha_{t,i} h_i$）：

```python
import torch
import torch.nn as nn


class DeepLOBSeq2Seq(nn.Module):
    """DeepLOB-Seq2Seq / DeepLOB-Attention (Zhang & Zohren 2021). Needs the DeepLOB class above.
    x: (B,1,50,40) -> probs (B,5,3), one row per horizon; train with NLL on log-probs."""

    def __init__(self, attention=False, hidden=64, n_classes=3, n_steps=5, conv_ch=32, inc_ch=64):
        super().__init__()
        self.att, self.K, self.n_steps = attention, n_classes, n_steps
        self.backbone = DeepLOB(conv_ch, inc_ch, hidden, n_classes, dropout=0.0)
        del self.backbone.lstm, self.backbone.fc
        self.encoder = nn.LSTM(3 * inc_ch, hidden, batch_first=True)
        self.decoder = nn.LSTMCell(n_classes + hidden, hidden)        # input [y_{t-1}; c]
        self.out = nn.Linear(2 * hidden, n_classes)
        if attention:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden, momentum=0.4) for _ in range(n_steps)])

    def forward(self, x):
        B = x.size(0)
        enc, (h, c) = self.encoder(self.backbone.features(x))   # enc (B,T',H)
        h, c = h[0], c[0]
        y_prev = x.new_zeros(B, self.K); y_prev[:, 0] = 1.0      # fixed one-hot start token
        ctx, outs = h, []
        for t in range(self.n_steps):
            h, c = self.decoder(torch.cat([y_prev, ctx], -1), (h, c))
            if self.att:
                a = torch.bmm(enc, h.unsqueeze(-1)).squeeze(-1).softmax(-1)   # dot-product attention
                ctx = self.bn[t](torch.bmm(a.unsqueeze(1), enc).squeeze(1))
                y = self.out(torch.cat([ctx, h], -1)).softmax(-1)
            else:
                y = self.out(torch.cat([h, ctx], -1)).softmax(-1)
            outs.append(y); y_prev = y
        return torch.stack(outs, 1)
```

## 参考文献

- Zhang, Zohren, Roberts (2019). DeepLOB: deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing* 67(11):3001–3012. https://arxiv.org/abs/1808.03668 ；代码 https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books （无 LICENSE）
- Zhang, Zohren (2021). Multi-horizon forecasting for limit order books: novel deep learning approaches and hardware acceleration using intelligent processing units. https://arxiv.org/abs/2105.10430 ；代码 https://github.com/zcakhaa/Multi-Horizon-Forecasting-for-Limit-Order-Books （MIT）
- Szegedy 等 (2015). Going deeper with convolutions (Inception). *CVPR 2015*.
- Hochreiter, Schmidhuber (1997). Long short-term memory. *Neural Computation* 9(8):1735–1780.
- Prata 等 (2024). LOBCAST. https://arxiv.org/abs/2308.01915
- Briola, Bartolucci, Aste (2025). Deep limit order book forecasting: a microstructural guide. *Quantitative Finance* 25(7):1101–1131. https://arxiv.org/abs/2403.09267
- Lucchese, Pakkanen, Veraart (2024). The short-term predictability of returns in order book markets: a deep learning perspective. *International Journal of Forecasting* 40(4):1587–1621. https://arxiv.org/abs/2211.13777

未核实：多步论文表格里 DeepLOB 本身 105,347 个参数无法用任何宽度组合复原。
