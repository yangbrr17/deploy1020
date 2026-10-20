# TLOB：双注意力 Transformer（附 MLPLOB）

Berti、Kasneci（2025），arXiv:2502.15757。返回 [深度学习总览](../deep-learning.md)。

**一句话**：每个 block 先在「时间」上做自注意力（token = 快照），再在「特征」上做自注意力（token = 嵌入后的通道），FFN 换成 MLP-Mixer 式的特征混合 / 时间混合；同时提出把**平滑窗口和预测 horizon 解耦**的新标签，并用**平均价差做阈值**检验可交易性。

## 1. 为什么选它

- **代表 2025 年的 Transformer 路线**，是目前公开的、在多个数据集（FI-2010、NASDAQ 的 TSLA 和 INTC、BTC）上都报告了结果的最新模型，代码维护得最好（MIT）。
- **它自己的实验很有价值**：
  - 阈值设成平均价差（考虑交易成本）后，F1 明显下降；
  - INTC 2012 → 2015 年可预测性下降；
  - 短 horizon 上纯 MLP（MLPLOB）就够了，长 horizon 上注意力才有优势。
- **代价**：100 万以上参数，CPU 推理是毫秒级（第 7 节），不能逐事件增量计算。

## 2. 主要创新点

1. **双注意力**：时间注意力学「哪些历史快照重要」，特征注意力学「哪些特征组合重要」。消融显示两者缺一个都会掉点（第 7 节）。
2. **MLPLOB**：只用 MLP 交替做特征混合和时间混合（MLP-Mixer 思路），各向同性设计，说明简单结构也能接近甚至超过 SoTA。
3. **新标签**：旧标签里平滑窗口就等于 horizon，$h$ 小时平滑不够、$h$ 大时平滑过度。新标签把平滑窗口 $k$ 固定、horizon $h$ 单独设置。
4. **价差阈值**：$\theta = \overline{\text{spread}} / \overline{\text{mid}}$，把交易成本写进「涨跌」的定义。

## 3. 输入和标签

- **张量** `(B, T, F)`：最近 $T$ 个快照（TLOB $T = 128$，MLPLOB $T = 384$），每个快照 $F$ 维。BTC 数据 $F = 40$（10 档），NASDAQ 数据 $F = 46$（40 维盘口 + 6 个订单字段：时间差、事件类型、数量、价格、方向、深度）。
- **采样**：NASDAQ 数据按成交量采样（每成交 500 股一个快照）；BTC 按时间（100–250 ms）。
- **归一化**：所有档位的价格共用一组均值和标准差、所有数量共用一组，用训练集统计量做 z-score；然后模型内部再过一层 BiN（见 [BiN-CTABL](bin-ctabl.md) 4.3 节）。

**新标签**（论文的写法，$p$ 是中间价）：

$$
w^{+}(t, h, k) = \frac{1}{k+1}\sum_{i=0}^{k} p_{t+h-i},\qquad
w^{-}(t, k) = \frac{1}{k+1}\sum_{i=0}^{k} p_{t-i},\qquad
l_t = \frac{w^{+} - w^{-}}{w^{-}}
$$

$$
y_t = \begin{cases} \text{Up} & l_t > \theta \\ \text{Down} & l_t < -\theta \\ \text{Stable} & \lvert l_t \rvert \le \theta \end{cases}
$$

代码里的实现略有不同：窗口是 $k$ 项（$k = 10$，若 $h < k$ 则 $k \leftarrow h$），阈值是 $\theta = \frac{1}{2}\cdot\frac{1}{N}\sum_t \lvert l_t \rvert$，并且训练、验证、测试集**各算一次**（测试集阈值用了测试集自己的统计量）。

**我们的看法**：解耦比旧标签好，但仍然是平滑标签，而且阈值是为了类别平衡定的。我们用首达三分类（见 [`features-and-targets.md`](../features-and-targets.md)）。

## 4. 数学

记 $F$ 为输入特征数，$T$ 为序列长度，$H$ 为隐藏维（默认 $H = F$），$L$ 为 block 数，头数 $n_h = 1$。

### 4.1 输入：BiN、嵌入、位置编码

$$
H^{(0)} = \mathrm{BiN}(X)\,W_e + \mathbf{1}_T b_e^\top + P \in \mathbb{R}^{T \times H},\qquad
P_{t, 2i} = \sin\frac{t}{10000^{2i/H}},\quad P_{t, 2i+1} = \cos\frac{t}{10000^{2i/H}}
$$

### 4.2 注意力子层 $\mathcal{A}_{d \to d'}$

输入 $Z \in \mathbb{R}^{N \times d}$（$N$ 个 token，每个 $d$ 维）。缩放点积注意力：

$$
Q = Z W_Q,\quad K = Z W_K,\quad V = Z W_V,\qquad
\mathrm{Attn}(Z) = \operatorname{softmax}\Big(\frac{Q K^\top}{\sqrt{d}}\Big) V
$$

softmax 按行进行，每个 query 在所有 key 上归一化。然后是 post-LN 残差和 MLP 块：

$$
U = \mathrm{LN}_d\big(\mathrm{Attn}(Z)\,W_0 + Z\big)
$$

$$
\mathrm{MLP}_{d \to d'}(U) = \mathrm{GELU}\Big(\mathrm{LN}_{d'}\big(\mathrm{GELU}(U W_1 + b_1)\,W_2 + b_2 + \mathbb{1}[d' = d]\,U\big)\Big),\qquad W_1 \in \mathbb{R}^{d \times 4d},\ W_2 \in \mathbb{R}^{4d \times d'}
$$

$$
\mathcal{A}_{d \to d'}(Z) = \mathrm{MLP}_{d \to d'}(U) + \mathbb{1}[d' = d]\,Z
$$

（官方代码里 $W_Q, W_K, W_V$ 之后 `nn.MultiheadAttention` 还有一次内部投影，参数冗余但不影响表达能力；最后的 $+Z$ 是代码特有的第二条残差。）

### 4.3 一个 TLOB block

$$
\text{时间注意力：}\quad H' = \mathcal{A}^{\text{temp}}_{H \to H_\ell}\big(H^{(\ell-1)}\big)\qquad (\text{token} = T \text{ 个快照，每个 } H \text{ 维})
$$

$$
\text{特征注意力：}\quad H^{(\ell)} = \Big[\mathcal{A}^{\text{spat}}_{T \to T_\ell}\big(H'^{\top}\big)\Big]^{\top}\qquad (\text{token} = H \text{ 个通道，每个 } T \text{ 维})
$$

- $\ell < L$ 时 $H_\ell = H$、$T_\ell = T$；最后一个 block 取 $H_L = \lfloor H/4 \rfloor$、$T_L = \lfloor T/4 \rfloor$，这一层的 MLP 兼做降维。
- 时间注意力子层里的 MLP 沿 $H$ 维混合（特征混合），特征注意力子层里的 MLP 沿 $T$ 维混合（时间混合），合起来正好是一个 MLPLOB block。
- 「特征注意力」的 token 是嵌入后的 $H$ 个通道（原始特征的线性组合），不是原始盘口特征本身。

### 4.4 分类头

$$
z = \operatorname{vec}\big(H^{(L)\top}\big) \in \mathbb{R}^{\lfloor H/4 \rfloor \lfloor T/4 \rfloor}
$$

只要维度 $n > 128$ 就接 `Linear(n, n // 4)` + GELU，最后 `Linear(n, 3)` 输出 logits。没有 dropout。

### 4.5 MLPLOB

$$
X_0 = \mathrm{GELU}\big(\mathrm{BiN}(X)\,W_e + b_e\big) \in \mathbb{R}^{T \times H}\qquad (\text{没有位置编码})
$$

$$
U_{i,:} = \mathrm{MLP}^{\text{feat}}(X_{i,:}),\ i = 1..T;\qquad
Z_{:,j} = \mathrm{MLP}^{\text{time}}(U_{:,j}),\ j = 1..H
$$

前 $L - 1$ 个 block 的隐藏宽度是 4 倍，最后一个是 2 倍并降到 $\lfloor H/4 \rfloor$、$\lfloor T/4 \rfloor$。

## 5. 结构和参数量

**TLOB，$F = H = 40$，$T = 128$，$L = 4$，$n_h = 1$**：

| # | 模块 | 输入 | 输出 | 参数 |
|---|---|---|---|---|
| 1 | BiN$(40, 128)$ | $128 \times 40$ | $128 \times 40$ | 338 |
| 2 | Linear$(40 \to 40)$ + 正弦位置编码 | $128 \times 40$ | $128 \times 40$ | 1,640 |
| 3 | Block 1 时间注意力（128 token × 40 维） | $128 \times 40$ | $128 \times 40$ | 26,280 |
| 4 | Block 1 特征注意力（40 token × 128 维） | $40 \times 128$ | $40 \times 128$ | 264,320 |
| 5–8 | Block 2、3 | | | 2 × 290,600 |
| 9 | Block 4 时间注意力，$40 \to 10$ | $128 \times 40$ | $128 \times 10$ | 21,390 |
| 10 | Block 4 特征注意力，$128 \to 32$ | $10 \times 128$ | $10 \times 32$ | 214,880 |
| 11 | 展平 | $10 \times 32$ | 320 | 0 |
| 12 | Linear$(320 \to 80)$ + GELU | 320 | 80 | 25,680 |
| 13 | Linear$(80 \to 3)$ | 80 | 3 | 243 |
| | **合计** | | | **1,135,971** |

参数量公式（$n_h = 1$）：

$$
\#\mathcal{A}_{d \to d} = \underbrace{3(d^2 + d)}_{Q, K, V} + \underbrace{4d^2 + 4d}_{\text{MHA 内部投影}} + \underbrace{d^2 + d}_{W_0} + \underbrace{2d}_{\text{LN}} + \underbrace{8d^2 + 7d}_{\text{MLP}} = 16d^2 + 17d
$$

**特征注意力子层的参数随 $T^2$ 增长**（$d = T = 128$ 时 26.4 万），这是参数的大头。

| 配置 | TLOB（$T = 128$，$L = 4$） | MLPLOB（$T = 384$，$L = 3$） |
|---|---|---|
| $F = H = 40$（BTC） | 1,135,971 | 3,012,411 |
| $F = H = 46$（NASDAQ） | 1,173,690 | 3,074,000 |
| $F = H = 144$（FI-2010） | 2,656,721 | 6,327,905 |

（论文 Table 2 写 MLPLOB $6.3 \times 10^7$、TLOB $1 \times 10^7$；按默认配置实测分别是 $6.3 \times 10^6$ 和 $2.7 \times 10^6$，论文的数字可能多写了一位或用了更大的配置。）

## 6. 训练设置（官方配置）

| 项 | 默认值 |
|---|---|
| 损失 | `CrossEntropyLoss()`，不加类权重 |
| 优化器 | Adam，eps $10^{-8}$ |
| 学习率 | TLOB $10^{-4}$；MLPLOB $3 \times 10^{-4}$ |
| 学习率调度 | 每个验证 epoch 结束时，val loss 改善不到 0.002 就减半 |
| 早停 | val loss，patience 1，最多 10 个 epoch |
| EMA | 权重的指数滑动平均，decay 0.999；验证、测试、保存都用 EMA 权重 |
| Batch | NASDAQ / BTC 128；FI-2010 32 |
| 硬件 | 论文用 RTX 3090 |

**EMA**：每个训练步之后 $\bar\theta \leftarrow 0.999\,\bar\theta + 0.001\,\theta$。它对噪声大的金融数据很有用，我们的网络也建议加上。

## 7. 参数怎么选、结果怎么分析

**超参数搜索**（论文附录）：优化器 {Adam, Lion}、序列长度 {64, 128, 256, 384, 512}、学习率 {$10^{-3}$, $3 \times 10^{-4}$, $10^{-4}$}、层数 {2, 3, 4, 6}。结论：

- 序列长度**超过 128 后进入平台期**，再长没有收益。
- 注意力头数取 1、2、4 没有差别，所以固定为 1。
- TLOB 最优：Adam、$T = 128$、$10^{-4}$、4 层。

**消融**（FI-2010，F1 %，总注意力层数固定为 8）：

| 模型 | h = 10 | h = 20 | h = 50 | h = 100 |
|---|---|---|---|---|
| 去掉特征注意力（8 层时间注意力） | 79.59 | 78.96 | 87.51 | 91.40 |
| 去掉时间注意力（8 层特征注意力） | 80.27 | 79.20 | 87.72 | 91.42 |
| **完整 TLOB** | **81.55** | **82.68** | **90.03** | **92.81** |

**按 horizon 分工**：MLPLOB 在短 horizon（10、20）上最好，TLOB 在长 horizon（50、100）上最好。短 horizon 上两者只差约 0.5，长 horizon 上 INTC 差约 7、BTC 差约 5。**对我们的执行场景（秒级、几次价格变动），短 horizon 才是重点，所以注意力的增量很可能有限。**

**报告的结果**（F1，%）：

| 数据 | h | DeepLOB | BiN-CTABL | MLPLOB | TLOB |
|---|---|---|---|---|---|
| FI-2010 | 10 / 20 / 50 / 100 | 71.1 / 62.4 / 75.4 / 77.6 | 81.1 / 71.5 / 87.7 / 92.1 | 81.6 / 84.9 / 91.4 / 92.6 | 81.6 / 82.7 / 90.0 / 92.8 |
| TSLA | 10 / 20 / 50 / 100 | 36.3 / 36.6 / 35.3 / 34.4 | 58.7 / 48.8 / 42.2 / 38.8 | 60.7 / 50.3 / 39.0 / 33.0 | 60.5 / 49.7 / 43.5 / 39.8 |
| INTC | 10 / 20 / 50 / 100 | 68.1 / 63.7 / 40.3 / 30.1 | 72.7 / 66.6 / 54.0 / 41.1 | 81.2 / 73.3 / 55.7 / 43.2 | 80.2 / 72.8 / 62.1 / 50.1 |
| BTC | 10 / 20 / 50 / 100 | 68.1 / 57.9 / 45.1 / 37.4 | 73.4 / 61.3 / 47.1 / 40.6 | 74.6 / 61.0 / 42.7 / 37.0 | 74.7 / 61.7 / 48.5 / 41.5 |

- **价差阈值**（TSLA，TLOB）：h = 50 / 100 / 200 时 F1 为 41.4 / 36.5 / 30.8，全面低于类别平衡阈值。INTC 做不了这个实验：价差相对波动太小，99.99% 的样本被判为「平稳」。**这说明：一旦把交易成本写进标签，可预测的部分大幅缩水。**
- **可预测性随时间下降**（INTC，h = 50）：2012 年 66.9，2015 年 60.2。
- 没有找到 TLOB 的第三方复现；论文没有报告种子数和方差。

**推理时间**：论文 Table 2 给出 TLOB 2.24 ms、MLPLOB 4.79 ms、BiN-CTABL 0.71 ms、DeepLOB 1.31 ms（测量的 batch 和设备没有说明）。我们在 4 核 Xeon 2.8 GHz、PyTorch 2.5、单线程、batch 1 实测：**TLOB p50 9.4 ms / p99 15.7 ms，MLPLOB p50 7.8 ms / p99 11.5 ms**。i9 多线程能快几倍，但仍是毫秒级，而且每来一个事件都要整窗重算。

**怎么分析训练好的模型**：

- 取时间注意力矩阵（$T \times T$）对一批样本平均，看权重是不是集中在最近几个快照（应当是）。
- 取特征注意力矩阵（$H \times H$），看哪些通道组合起作用；通道是原始特征的线性组合，要配合嵌入矩阵 $W_e$ 解读。
- 比较 TLOB 和 MLPLOB 在不同 horizon 上的差距，判断注意力是否值得它的延迟。

## 8. 已知问题

1. **BiN 的位置**：图注说「每个 block 前都有 BiN」，代码只在输入处用一次。
2. **标签实现和论文不一致**（第 3 节）：窗口项数、阈值公式不同，且测试集阈值用了测试集自己的统计量。
3. **NASDAQ 数据的 horizon 列号错位**（静态读代码发现，未运行确认）：读取函数按 5 个 horizon 列写，但只生成了 4 列，按当前代码 h = 10 会读到特征列。用之前要自己核对。
4. **订单类型嵌入被 `.detach()`**：`Embedding(3, 1)` 永远不会被训练。
5. **默认训练预算很紧**：最多 10 个 epoch、patience 1，复现时要确认是否收敛。
6. **单股训练**：NASDAQ 数据每次只用一只股票训练。

## 9. 在我们的场景里怎么用

- **不建议作为第一版上线模型**：毫秒级延迟、不能增量推理、100 万以上参数、短 horizon 上相对 MLP 没有优势。
- **适合作为第二阶段的研究对照**：在 GBDT 和 [OFI-TCN-GRU](ofi-tcn-gru.md) 之后，用同样的数据和标签训练一个小版本（$H = 32$，$T = 64$，$L = 2$），看注意力在较长的 τ 上有没有增量。
- **借鉴的设计**：EMA 权重、BiN 输入归一化、「序列长度超过 128 后进入平台期」的结论、MLPLOB 说明纯 MLP 在短 horizon 上就够了。
- **CPU 训练**：$H = 32$、$T = 64$、$L = 2$ 的小版本约 15 万参数，i9 上 $10^7$ 样本训练一个 epoch 约 1 小时（量级估计）。

## 10. 代码

逐行对应官方 `models/tlob.py` 和 `models/mlplob.py`（已运行：把官方仓库的权重复制进来后输出逐元素一致；参数量见第 5 节）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiN(nn.Module):
    """Bilinear normalisation, x: (B, D, T). Same as on the BiN-CTABL page."""

    def __init__(self, D, T, eps_std=1e-4):
        super().__init__()
        self.gamma2, self.beta2 = nn.Parameter(torch.empty(D, 1)), nn.Parameter(torch.zeros(D, 1))
        self.gamma1, self.beta1 = nn.Parameter(torch.empty(T, 1)), nn.Parameter(torch.zeros(T, 1))
        self.lambda1, self.lambda2 = nn.Parameter(torch.tensor(0.5)), nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_normal_(self.gamma2); nn.init.xavier_normal_(self.gamma1)
        self.eps_std = eps_std

    def forward(self, x):
        mu2, sd2 = x.mean(2, keepdim=True), x.std(2, keepdim=True)
        sd2 = torch.where(sd2 < self.eps_std, torch.ones_like(sd2), sd2)
        mu1, sd1 = x.mean(1, keepdim=True), x.std(1, keepdim=True)
        sd1 = torch.where(sd1 < self.eps_std, torch.ones_like(sd1), sd1)
        X2 = self.gamma2 * (x - mu2) / sd2 + self.beta2
        X1 = self.gamma1.T * (x - mu1) / sd1 + self.beta1.T
        return self.lambda1 * X1 + self.lambda2 * X2


class MLPBlock(nn.Module):
    """gelu(LayerNorm(gelu(x W1) W2 + x)); the residual only when dims match."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1, self.fc2, self.norm = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out)

    def forward(self, x):                                  # (B, N, d_in) -> (B, N, d_out)
        y = self.fc2(F.gelu(self.fc1(x)))
        if y.shape[-1] == x.shape[-1]:
            y = y + x
        return F.gelu(self.norm(y))


class TLOBAttentionLayer(nn.Module):
    """Self-attention over the token axis of x: (B, N, d), then an MLP block d -> d_out."""

    def __init__(self, d, num_heads, d_out):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(d, d * num_heads) for _ in range(3))
        self.attn = nn.MultiheadAttention(d * num_heads, num_heads, batch_first=True)
        self.w0 = nn.Linear(d * num_heads, d)
        self.norm = nn.LayerNorm(d)
        self.mlp = MLPBlock(d, 4 * d, d_out)

    def forward(self, x):
        res = x
        a, att = self.attn(self.q(x), self.k(x), self.v(x), need_weights=True, average_attn_weights=False)
        x = self.mlp(self.norm(self.w0(a) + res))          # post-LN residual, then MLP
        if x.shape[-1] == res.shape[-1]:
            x = x + res                                    # second residual (official code)
        return x, att                                      # att: (B, heads, N, N)


def sinusoidal_pe(T, d, n=10000.0):
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.pow(n, 2 * torch.arange(d // 2) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / div), torch.cos(pos / div)
    return pe


def _head(n):
    layers = []
    while n > 128:
        layers += [nn.Linear(n, n // 4), nn.GELU()]
        n //= 4
    return nn.Sequential(*layers, nn.Linear(n, 3))


class TLOB(nn.Module):
    """x: (B, T, F) -> logits (B, 3)."""

    def __init__(self, num_features, seq_size=128, hidden_dim=40, num_layers=4, num_heads=1):
        super().__init__()
        T, H = seq_size, hidden_dim
        self.bin = BiN(num_features, T)
        self.emb = nn.Linear(num_features, H)
        self.register_buffer("pe", sinusoidal_pe(T, H))
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            last = i == num_layers - 1
            self.layers.append(TLOBAttentionLayer(H, num_heads, H // 4 if last else H))   # temporal
            self.layers.append(TLOBAttentionLayer(T, num_heads, T // 4 if last else T))   # feature
        self.head = _head((H // 4) * (T // 4))

    def forward(self, x):                                  # (B, T, F)
        x = self.bin(x.transpose(1, 2)).transpose(1, 2)
        x = self.emb(x) + self.pe                          # (B, T, H)
        for layer in self.layers:
            x, _ = layer(x)
            x = x.transpose(1, 2)                          # alternate (B,T,H) <-> (B,H,T)
        return self.head(x.transpose(1, 2).reshape(x.shape[0], -1))


class MLPLOB(nn.Module):
    def __init__(self, num_features, seq_size=384, hidden_dim=40, num_layers=3):
        super().__init__()
        T, H = seq_size, hidden_dim
        self.bin = BiN(num_features, T)
        self.emb = nn.Linear(num_features, H)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.blocks += [MLPBlock(H, 4 * H, H), MLPBlock(T, 4 * T, T)]
            else:
                self.blocks += [MLPBlock(H, 2 * H, H // 4), MLPBlock(T, 2 * T, T // 4)]
        self.head = _head((H // 4) * (T // 4))

    def forward(self, x):
        x = self.bin(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.emb(x))
        for blk in self.blocks:
            x = blk(x).transpose(1, 2)                     # feature mixing, then time mixing
        return self.head(x.reshape(x.shape[0], -1))


if __name__ == "__main__":
    t, m = TLOB(40, 128, 40, 4), MLPLOB(40, 384, 40, 3)
    print(sum(p.numel() for p in t.parameters()), sum(p.numel() for p in m.parameters()))  # 1135971 3012411
```

EMA 权重（训练时每步更新，评估时换上）：

```python
import copy
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.shadow = decay, copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
```

## 参考文献

- Berti, Kasneci (2025). TLOB: a novel transformer model with dual attention for price trend prediction with limit order book data. https://arxiv.org/abs/2502.15757 ；代码 https://github.com/LeonardoBerti00/TLOB （MIT）
- Vaswani 等 (2017). Attention is all you need. *NeurIPS 2017*.
- Tolstikhin 等 (2021). MLP-Mixer: an all-MLP architecture for vision. *NeurIPS 2021*.
- Prata 等 (2024). LOBCAST. *Artificial Intelligence Review* 57. https://arxiv.org/abs/2308.01915

未核实：论文推理时间的测量条件；TLOB 的第三方复现。
