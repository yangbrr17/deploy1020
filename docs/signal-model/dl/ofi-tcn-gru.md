# 推荐方案：OFI-TCN-GRU

我们自己的设计，综合了前面四个模型的结论。返回 [深度学习总览](../deep-learning.md)。

**一句话**：输入用逐档订单流（来自 [Deep OFI](deep-ofi.md)），先用因果膨胀卷积（TCN）抓最近几十个事件的局部模式，再用 GRU 承接更长的依赖；股票属性和慢变量在输出头之前拼接；多个 horizon 各有一个「首达」头和两个「条件幅度」头。**约 5 万参数，可以逐事件增量推理**：纯 numpy 单步约 160 µs，C++ 预计几十微秒。

| 设计选择 | 依据 |
|---|---|
| 逐档 OFI 当主输入 | Deep OFI 和独立复现：订单流输入显著好于原始盘口 |
| 小网络（5 万参数） | 真实数据上模型之间只差 1–5 个 F1 点；BiN-CTABL 用 1.1 万参数就是最稳的基线 |
| 因果 TCN + GRU，不用 Transformer | 有效 horizon 约 2 次价格变动，信息集中在最近几十个事件；GRU 可以逐事件更新，Transformer 不行；TLOB 显示短 horizon 上注意力没有优势 |
| 首达头为主，幅度头以「动了」为条件、涨跌分开 | 执行决策需要的就是这三个量（见 [`ml-pipeline.md`](../ml-pipeline.md) 2.8 节的成本公式） |
| 不加类权重、不用 focal、独立窗口温度校准 | 执行层直接用概率算成本，概率必须校准 |
| 所有股票共用一个模型 + 股票 embedding | 通用模型好于逐只模型（Sirignano 和 Cont 2019；Lucchese 等 2024） |

## 1. 输入

**序列输入** $x_t \in \mathbb{R}^{F}$，$F \approx 48$，每个决策相关事件一行，窗口长 $T = 100$（会做消融）：

| 组 | 维数 | 内容 |
|---|---|---|
| 逐档 OFI | 10 | $\mathrm{OFI}^{(l)}_t / \overline{\mathrm{Depth}}^{(l)}$，$l = 1..10$ |
| 最优 3 档的 bOF / aOF | 6 | 买卖两路分开，让网络学不对称 |
| 前 5 档相对中间价的价格和量 | 20 | $(P^a_l - m_t)/\delta$、$(m_t - P^b_l)/\delta$、$\log(1 + q^a_l)$、$\log(1 + q^b_l)$，按股票滚动标准化 |
| 事件元数据 | 8 | 事件类型 one-hot（加挂、撤单、成交 × 买卖）、$\log \Delta t$、价差 $S_{\text{tk}}$ |
| 成交 | 4 | 带符号成交量 $\varepsilon v$、成交是否吃穿一档、撤单比例、$\mathrm{OBI}_1$ |

**静态输入** $u \in \mathbb{R}^{16}$：时段（$\sin, \cos$）、日内已实现波动、近 5 分钟成交强度、指数期货 / ETF 的滞后 OFI、tick 类别、平均中间价变动间隔 $\bar\Delta$、$P(S_{\text{tk}} = 1)$。

**股票 id**：8 维 embedding。

所有特征只用决策时刻已经到达的信息；标准化用过去 5 个交易日的统计量（见 [`features-and-targets.md`](../features-and-targets.md) 第 5 节）。

## 2. 数学

### 2.1 输入投影

$$
z_t = \mathrm{LN}\big(W_{\text{in}} x_t + b_{\text{in}}\big) \in \mathbb{R}^{C},\qquad C = 32
$$

### 2.2 因果膨胀卷积

核大小 $k$、膨胀率 $d$ 的因果卷积只用当前和过去的输入：

$$
\big(\mathcal{C}_{k,d}\, z\big)_t = b + \sum_{j=0}^{k-1} W_j\, z_{t - j d},\qquad W_j \in \mathbb{R}^{C \times C}
$$

实现上在左边补 $(k-1)d$ 个零。一个残差块两层卷积：

$$
\mathrm{Block}_d(z)_t = z_t + \mathrm{Drop}\Big(\mathrm{GELU}\big(\mathcal{C}_{k,d}\ \mathrm{Drop}(\mathrm{GELU}(\mathcal{C}_{k,d}\, z))\big)_t\Big)
$$

三个块的膨胀率 $d \in \{1, 2, 4\}$，$k = 3$。**感受野**（输出 $t$ 能看到的最远输入）：

$$
R = 1 + \sum_{\text{块}} \sum_{\text{块内两层}} (k - 1)\,d = 1 + 2 \cdot 2 \cdot (1 + 2 + 4) = 29\ \text{个事件}
$$

TCN 负责最近 29 个事件的局部模式（比如队列快速变薄、一串同向的订单流）；更长的依赖交给 GRU。

$\mathrm{GELU}(x) = x\,\Phi(x)$，$\Phi$ 是标准正态分布函数。

### 2.3 GRU

对 TCN 的输出序列 $\tilde z_t$：

$$
\begin{aligned}
r_t &= \sigma\big(W_r \tilde z_t + b_{ir} + U_r h_{t-1} + b_{hr}\big)\\
u_t &= \sigma\big(W_u \tilde z_t + b_{iu} + U_u h_{t-1} + b_{hu}\big)\\
n_t &= \tanh\big(W_n \tilde z_t + b_{in} + r_t \odot (U_n h_{t-1} + b_{hn})\big)\\
h_t &= (1 - u_t) \odot n_t + u_t \odot h_{t-1}
\end{aligned}
\qquad h_t \in \mathbb{R}^{64}
$$

**GRU 的隐状态可以逐事件更新**：来一个新事件，只算一步。这是它在执行场景里胜过 Transformer 的关键。

### 2.4 共享主干和输出头

$$
s = W_s\,[u;\ e_{\text{stock}}] + b_s \in \mathbb{R}^{16},\qquad
g = \mathrm{Drop}\big(\mathrm{GELU}(W_g [h_T;\ s] + b_g)\big) \in \mathbb{R}^{64}
$$

每个 horizon $\tau_j$（如 1、5、30 秒）三个头：

$$
\hat\pi_j = \operatorname{softmax}\big(A_j g + a_j\big) = \big(\hat\pi^{\uparrow}_j,\ \hat\pi^{\downarrow}_j,\ \hat\pi^{0}_j\big),
\qquad
\hat\rho^{\uparrow}_j = \operatorname{softmax}\big(B^{\uparrow}_j g + b^{\uparrow}_j\big),
\qquad
\hat\rho^{\downarrow}_j = \operatorname{softmax}\big(B^{\downarrow}_j g + b^{\downarrow}_j\big)
$$

- $\hat\pi_j$：τ 内先涨 ≥ 1 tick / 先跌 ≥ 1 tick / 都没有。**主决策用这个。**
- $\hat\rho^{\uparrow}_j$：在「先涨」的条件下，τ 内最大上移 $\in \{1, 2, \ge 3\}$ tick 的概率；$\hat\rho^{\downarrow}_j$ 同理。

**交给执行层的量**（买单为例，对应成本公式里的 $p_\uparrow, p_\downarrow, p_0, \Delta_\uparrow, \Delta_\downarrow$）：

$$
p_\uparrow = \hat\pi^{\uparrow},\quad p_\downarrow = \hat\pi^{\downarrow},\quad p_0 = \hat\pi^{0},\qquad
\Delta_\uparrow = \delta \sum_{m=1}^{3} \bar m_m\,\hat\rho^{\uparrow}_m,\quad \bar m = (1, 2, \mathbb{E}[\text{幅度} \mid \ge 3])
$$

$\mathbb{E}[\text{幅度} \mid \ge 3]$ 用训练集的经验均值。

## 3. 损失

样本 $i$ 的唯一性权重 $w_i$（见 [`ml-pipeline.md`](../ml-pipeline.md) 2.4 节）：

$$
\mathcal{L} = \frac{1}{\sum_i w_i}\sum_i w_i \sum_{j} \Big[\underbrace{-\log \hat\pi_{j, y^{\text{fp}}_{ij}}(x_i)}_{\text{首达}} + \lambda_{\text{mag}}\Big(\mathbb{1}[y^{\text{fp}}_{ij} = \uparrow]\big(-\log \hat\rho^{\uparrow}_{j, y^{\text{mag}}_{ij}}\big) + \mathbb{1}[y^{\text{fp}}_{ij} = \downarrow]\big(-\log \hat\rho^{\downarrow}_{j, y^{\text{mag}}_{ij}}\big)\Big)\Big]
$$

- 幅度项只对「动了」的样本计算（条件分布），涨和跌用不同的头。
- $\lambda_{\text{mag}} = 0.3$ 起步；跑 1 个 epoch 后看两项的量级，调到幅度项约占总损失的 20%–30%。
- **不加类权重、不用 focal loss**：两者都会让 $\hat\pi$ 偏离真实概率，而成本公式直接用概率。

幅度也可以用有序分类（累积 logit）：$P(Y \le m \mid \text{涨}) = \sigma(\theta_m - f(g))$，$\theta_1 < \theta_2$，参数更少、保证单调。数据量大时两种差别不大。

## 4. 参数量

$C = 32$，$H_{\text{gru}} = 64$，3 个 horizon，500 只股票：

| 模块 | 计算 | 参数 |
|---|---|---|
| 输入投影 + LN | $48 \cdot 32 + 32 + 2 \cdot 32$ | 1,632 |
| TCN（3 块 × 2 层） | $6 \times (3 \cdot 32 \cdot 32 + 32)$ | 18,624 |
| GRU | $3 \times (32 \cdot 64 + 64 \cdot 64 + 2 \cdot 64)$ | 18,816 |
| 股票 embedding | $500 \times 8$ | 4,000 |
| 静态投影 | $(16 + 8) \cdot 16 + 16$ | 400 |
| 主干 | $(64 + 16) \cdot 64 + 64$ | 5,184 |
| 三组头 | $3 \times (64 \cdot 9 + 9)$ | 1,755 |
| **合计** | | **50,411** |

每个事件的乘加次数约 $4.5 \times 10^4$（流式推理时）。

## 5. 超参数怎么选

| 超参数 | 起点 | 搜索范围 | 怎么判断 |
|---|---|---|---|
| 窗口 $T$ | 100 | {50, 100, 200} | 用 OFI 输入时信息集中在最近几十个事件，$T = 50$ 很可能就够；TLOB 的结论是超过 128 进入平台期。**流式推理时 $T$ 只影响训练，不影响线上延迟** |
| TCN 膨胀率 | {1, 2, 4} | {1, 2}、{1, 2, 4}、{1, 2, 4, 8} | 感受野 $R$ 应覆盖「约 2 次平均价格变动」所对应的事件数；按股票池里这个事件数的中位数来选 |
| 通道 $C$ | 32 | {16, 32, 64} | 看验证 log loss 和种子间方差；数据少时取小 |
| GRU 隐藏维 | 64 | {32, 64, 128} | 同上 |
| Dropout | 0.1（TCN）/ 0.2（主干） | [0, 0.3] | 验证 loss 1–3 个 epoch 就回升时加大 |
| 学习率 | OneCycle，max 2e-3 | [5e-4, 5e-3] log | 只搜这一项和上面三项，20–40 个 trial |
| Batch | 512 | {256, 512, 1024} | CPU 上越大越快，学习率按比例放大 |
| Weight decay | 1e-4 | [0, 1e-3] | — |
| $\lambda_{\text{mag}}$ | 0.3 | [0.1, 1] | 按损失量级平衡 |
| EMA decay | 0.999（带预热） | — | 噪声大的数据上让验证曲线更平滑 |

**搜索方法**：Optuna TPE + 中位数剪枝；每个 trial 按日切分训练 / 早停验证 / 校准 / 测试；选模用 CPCV 路径上验证 log loss 技能分的中位数和 10% 分位数。至少 3 个种子，报告均值 ± 标准差（文献里种子间方差可达 5 个 F1 点）。

**必须做的消融**（判断每个设计是不是真有用）：

1. 输入：OFI 组 vs 原始价量组 vs 两者都有。
2. 结构：去掉 TCN（只用 GRU）；去掉 GRU（TCN + 最后时刻）；换成 BiN-CTABL。
3. 多任务：只有首达头 vs 加幅度头（看首达头的 log loss 有没有被拖累）。
4. 股票 embedding：有 vs 无；新股票上的表现（留一批股票不参与训练）。
5. 和 GBDT 比：同样的样本、标签、切分；再试「GBDT + 网络输出当特征」。

## 6. 训练流程

1. **切分**：按交易日，训练 60 天 → 空 1 天（purge）→ 早停验证 8 天 → 校准 2 天 → 测试 10 天；之后 walk-forward 滚动。有多日滚动特征时 purge ≥ 窗口天数。
2. **采样**：决策时点采样；样本高度重叠，每隔 5–10 个事件取一个窗口，并乘唯一性权重。
3. **优化**：AdamW + OneCycle（warmup 10%）；梯度裁剪 1.0；EMA 权重（带预热 $d_t = \min(0.999, \frac{1+t}{10+t})$）。
4. **早停**：盯 EMA 模型在验证集上的加权 log loss，patience 5，最多 30 个 epoch。
5. **校准**：在校准窗口上对每个 horizon 的首达头单独拟合温度 $T_j$（见 [`ml-pipeline.md`](../ml-pipeline.md) 2.7 节）。
6. **i9 上的训练时间**（量级估计，未实测）：5 万参数、$10^7$ 个样本、batch 512，前向加反向每个样本约 $3 \times 4.5 \times 10^4 \times T$ 次乘加，$T = 100$ 时约 $1.4 \times 10^7$；一个 epoch 约 $1.4 \times 10^{14}$ 次运算，i9 多线程实际吞吐按 $10^{11}$–$10^{12}$ 次/秒算，**一个 epoch 约 3–20 分钟，30 个 epoch 几小时以内**。用 `torch.set_num_threads(物理核数)`。

## 7. 怎么分析训练好的模型

- **分 horizon 的技能曲线**：$\text{Skill}_j = 1 - \mathrm{LL}_j / \mathrm{LL}_{\text{prior}, j}$ 对 $\tau_j / \bar\Delta$ 画图，找峰值位置，用来确定执行的决策间隔。
- **校准**：每个 horizon 的可靠性图和 classwise-ECE，分价差状态、时段、tick 类别看。
- **信号十分位图**：按 $\hat\pi^\uparrow - \hat\pi^\downarrow$ 分十组，看未来 $\Delta m / \delta$ 的均值，应当单调、两端超过半个价差。
- **按股票分组**：按 tick 类别或信息丰富度 $\mathrm{IR} = \log\frac{\#\text{更新}}{\#\text{价格变动}}$ 分组看技能分，决定在哪些股票上启用。
- **特征归因**：积分梯度（integrated gradients），对输入 $x$ 和基线 $x'$（如全零的标准化输入）：

$$
\mathrm{IG}_i(x) = (x_i - x'_i)\int_0^1 \frac{\partial F\big(x' + \alpha (x - x')\big)}{\partial x_i}\,d\alpha
$$

  $F$ 取 $\log\hat\pi^\uparrow - \log\hat\pi^\downarrow$。按特征组和时间位置汇总，看模型是不是主要在用最近几个事件的 OFI（符合预期），还是在用时段这类可能过拟合的变量。
- **种子间方差**：至少 3 个种子；技能分的标准差大于均值的一半，结论就不可靠。

## 8. 部署：逐事件流式推理

**原理**：因果卷积在时刻 $t$ 只需要每层最近 $(k - 1)d$ 个输入，GRU 只需要 $h_{t-1}$。所以给每层卷积维护一个长度 $(k-1)d$ 的环形缓冲，加上 GRU 隐状态，来一个新事件只算一步：

$$
\text{整窗计算：}\ O(T \cdot \text{每步代价}) \qquad\longrightarrow\qquad \text{流式：}\ O(\text{每步代价})
$$

**验证**：流式逐事件推进 100 步的输出和整窗前向**逐元素一致**（最大差 0.0）；纯 numpy 版本和 PyTorch 的差在 $10^{-7}$ 量级。

**实测延迟**（4 核 Xeon 2.8 GHz，单线程，batch 1；i9 单核更快）：

| 方式 | p50 | p99 |
|---|---|---|
| PyTorch 整窗前向（$T = 100$） | 5.7 ms | 11.6 ms |
| PyTorch 流式单步 | 1.2 ms | 2.9 ms |
| **numpy 流式单步** | **0.16 ms** | 0.53 ms |
| C++ 手写（估计：约 $4.5 \times 10^4$ 次乘加） | 约 10–30 µs | — |

PyTorch 单步慢在框架对几十个小算子的调用开销；上线用 numpy 或 C++ 手写前向，权重从训练好的模型导出。也可以导出 ONNX，用 ONNX Runtime 的动态 int8 量化，但要和 fp32 比 log loss，并在目标 CPU 上实测。

## 9. 代码

模型、多任务损失和流式推理（已运行：参数量 50,411，感受野 29，流式和整窗输出一致）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """y_t = b + sum_{j=0}^{k-1} W_j x_{t - j*d}: left-pad by (k-1)*d so no future step is used."""

    def __init__(self, c_in, c_out, k, dilation):
        super().__init__(c_in, c_out, k, dilation=dilation)
        self.left = (k - 1) * dilation

    def forward(self, x):                                   # (B, C, T)
        return super().forward(F.pad(x, (self.left, 0)))


class TCNBlock(nn.Module):
    """Two causal dilated convs with GELU + dropout and a residual connection."""

    def __init__(self, c, k, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(c, c, k, dilation)
        self.conv2 = CausalConv1d(c, c, k, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # (B, C, T)
        y = self.drop(F.gelu(self.conv1(x)))
        y = self.drop(F.gelu(self.conv2(y)))
        return x + y


class OFITCNGRU(nn.Module):
    """OFI sequence -> causal TCN -> GRU -> shared trunk (+ static features) -> per-horizon heads.

    seq:    (B, T, F_seq)  per-event inputs (multi-level OFI, prices rel. to mid in ticks, event meta)
    static: (B, F_static)  slow features (time of day, realised vol, lagged cross-asset OFI, tick class)
    stock:  (B,)           integer stock id for a small embedding
    returns dict of logits, each (B, H, 3):
      first_passage: [up first, down first, none]   (main decision head)
      mag_up / mag_down: ordinal bins {1, 2, >=3 ticks}, conditional on the move
    """

    def __init__(self, f_seq=48, f_static=16, n_stocks=500, n_horizons=3, c=32, hidden=64,
                 k=3, dilations=(1, 2, 4), dropout=0.1, emb_dim=8):
        super().__init__()
        self.inp = nn.Linear(f_seq, c)
        self.norm = nn.LayerNorm(c)
        self.tcn = nn.ModuleList([TCNBlock(c, k, d, dropout) for d in dilations])
        self.gru = nn.GRU(c, hidden, batch_first=True)
        self.stock_emb = nn.Embedding(n_stocks, emb_dim)
        self.static = nn.Linear(f_static + emb_dim, 16)
        self.trunk = nn.Sequential(nn.Linear(hidden + 16, 64), nn.GELU(), nn.Dropout(0.2))
        self.H = n_horizons
        self.head_fp = nn.Linear(64, 3 * n_horizons)
        self.head_up = nn.Linear(64, 3 * n_horizons)
        self.head_dn = nn.Linear(64, 3 * n_horizons)
        self.receptive_field = 1 + sum(2 * (k - 1) * d for d in dilations)

    def encode(self, seq):                                  # (B, T, F) -> (B, T, hidden)
        x = self.norm(self.inp(seq)).transpose(1, 2)        # (B, C, T)
        for blk in self.tcn:
            x = blk(x)
        out, _ = self.gru(x.transpose(1, 2))
        return out

    def heads(self, h_last, static, stock):
        s = self.static(torch.cat([static, self.stock_emb(stock)], -1))
        z = self.trunk(torch.cat([h_last, s], -1))
        B = z.shape[0]
        return {"first_passage": self.head_fp(z).view(B, self.H, 3),
                "mag_up": self.head_up(z).view(B, self.H, 3),
                "mag_down": self.head_dn(z).view(B, self.H, 3)}

    def forward(self, seq, static, stock):
        return self.heads(self.encode(seq)[:, -1], static, stock)


def multitask_loss(out, y_fp, y_mag, w, lam_mag=0.3):
    """Cross-entropy on the first-passage head + conditional ordinal-bin CE on the magnitude heads.
    y_fp: (B, H) in {0 up, 1 down, 2 none}; y_mag: (B, H) in {0, 1, 2} (ignored when y_fp == 2);
    w: (B,) uniqueness weights. No class weights, so probabilities stay calibrated."""
    B, H, _ = out["first_passage"].shape
    ce_fp = F.cross_entropy(out["first_passage"].reshape(B * H, 3), y_fp.reshape(-1), reduction="none").view(B, H)
    up, dn = (y_fp == 0).float(), (y_fp == 1).float()
    ce_up = F.cross_entropy(out["mag_up"].reshape(B * H, 3), y_mag.reshape(-1), reduction="none").view(B, H)
    ce_dn = F.cross_entropy(out["mag_down"].reshape(B * H, 3), y_mag.reshape(-1), reduction="none").view(B, H)
    per_sample = ce_fp.sum(1) + lam_mag * (up * ce_up + dn * ce_dn).sum(1)
    return (per_sample * w).sum() / w.sum()


class StreamingOFITCNGRU:
    """Event-by-event inference: each new event costs O(1) instead of re-running the whole window.
    Keeps, per causal conv, a ring buffer of its last (k-1)*d inputs, plus the GRU hidden state."""

    def __init__(self, model: OFITCNGRU):
        self.m = model.eval()
        self.convs = [conv for blk in self.m.tcn for conv in (blk.conv1, blk.conv2)]
        self.reset(1)

    def reset(self, batch=1):
        c = self.m.inp.out_features
        self.buf = [torch.zeros(batch, c, conv.left) for conv in self.convs]
        self.h = torch.zeros(1, batch, self.m.gru.hidden_size)

    def _conv_step(self, i, x):                             # x: (B, C) -> (B, C)
        conv, buf = self.convs[i], self.buf[i]
        window = torch.cat([buf, x.unsqueeze(-1)], -1)      # (B, C, left + 1)
        y = F.conv1d(window, conv.weight, conv.bias, dilation=conv.dilation)[..., -1]
        self.buf[i] = window[..., 1:]
        return y

    @torch.inference_mode()
    def step(self, x_event, static, stock):                 # x_event: (B, F_seq)
        x = self.m.norm(self.m.inp(x_event))                # (B, C)
        i = 0
        for _ in self.m.tcn:
            y = F.gelu(self._conv_step(i, x)); i += 1
            y = F.gelu(self._conv_step(i, y)); i += 1
            x = x + y
        out, self.h = self.m.gru(x.unsqueeze(1), self.h)
        return self.m.heads(out[:, -1], static, stock)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = OFITCNGRU()
    print("params", sum(p.numel() for p in m.parameters()), "receptive field", m.receptive_field)
    B, T = 4, 100
    seq, st, sid = torch.randn(B, T, 48), torch.randn(B, 16), torch.randint(0, 500, (B,))
    m.eval()
    with torch.inference_mode():
        full = m(seq, st, sid)["first_passage"]
    s = StreamingOFITCNGRU(m); s.reset(B)
    for t in range(T):
        last = s.step(seq[:, t], st, sid)["first_passage"]
    print("streaming == full window:", torch.allclose(full, last, atol=1e-5))
```

纯 numpy 的逐事件编码器（batch 1，上线用；输出 $h_t$ 后接主干和输出头，这两部分只是几个小矩阵乘）：

```python
import numpy as np
from scipy.special import erf

_SQRT2 = np.sqrt(2.0)


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / _SQRT2))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyStreamer:
    """Batch-1, event-by-event forward pass of a trained OFITCNGRU with plain numpy (no framework overhead).
    Ring buffers hold each causal conv's last (k-1)*d inputs; h is the GRU state."""

    def __init__(self, model):
        g = lambda t: t.detach().cpu().numpy().astype(np.float64)
        self.Wi, self.bi = g(model.inp.weight), g(model.inp.bias)
        self.ln_w, self.ln_b, self.ln_eps = g(model.norm.weight), g(model.norm.bias), model.norm.eps
        self.convs = []
        for blk in model.tcn:
            for conv in (blk.conv1, blk.conv2):
                W = g(conv.weight)                              # (out, in, k)
                self.convs.append((W, g(conv.bias), conv.dilation[0], W.reshape(W.shape[0], -1)))
        gru = model.gru
        self.Wih, self.Whh, self.bih, self.bhh = g(gru.weight_ih_l0), g(gru.weight_hh_l0), g(gru.bias_ih_l0), g(gru.bias_hh_l0)
        self.model = model
        self.reset()

    def reset(self):
        self.bufs = [np.zeros((W.shape[1], (W.shape[2] - 1) * d)) for W, _, d, _ in self.convs]
        self.h = np.zeros(self.Whh.shape[1])

    def _conv(self, i, x):
        W, b, d, Wflat = self.convs[i]
        win = np.concatenate([self.bufs[i], x[:, None]], 1)    # (C, (k-1)d + 1)
        taps = win[:, ::d]                                      # inputs at t-(k-1)d, ..., t-d, t
        y = Wflat @ taps.ravel() + b                            # (out, in*k) @ (in*k,)
        self.bufs[i] = win[:, 1:]
        return y

    def step(self, x_event):
        """Returns the GRU output h_t; feed it to the (cheap) trunk and heads."""
        x = self.Wi @ x_event + self.bi
        mu, var = x.mean(), x.var()
        x = (x - mu) / np.sqrt(var + self.ln_eps) * self.ln_w + self.ln_b
        for j in range(0, len(self.convs), 2):
            y = _gelu(self._conv(j, x))
            y = _gelu(self._conv(j + 1, y))
            x = x + y
        H = self.h.shape[0]
        gi, gh = self.Wih @ x + self.bih, self.Whh @ self.h + self.bhh
        r = _sigmoid(gi[:H] + gh[:H]); z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1 - z) * n + z * self.h
        return self.h
```

训练循环（唯一性权重、AdamW + OneCycle、梯度裁剪、带预热的 EMA、按验证 log loss 早停；在模拟数据上验证过能学到信号）：

```python
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate_logloss(model, loader):
    """Uniqueness-weighted log loss of the first-passage head, averaged over horizons."""
    model.eval(); num = den = 0.0
    with torch.inference_mode():
        for seq, st, sid, y_fp, y_mag, w in loader:
            lg = model(seq, st, sid)["first_passage"]                       # (B, H, 3)
            ce = F.cross_entropy(lg.reshape(-1, 3), y_fp.reshape(-1), reduction="none").view_as(y_fp).mean(1)
            num += float((ce * w).sum()); den += float(w.sum())
    return num / den


def train(model, train_ds, val_ds, loss_fn, epochs=30, batch=512, max_lr=2e-3, wd=1e-4,
          patience=5, ema_decay=0.999, threads=None):
    if threads:
        torch.set_num_threads(threads)                                    # e.g. physical cores of the i9
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=4 * batch)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs * len(tl), pct_start=0.1)
    ema = copy.deepcopy(model).eval()
    best, best_state, bad, step = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        for seq, st, sid, y_fp, y_mag, w in tl:
            loss = loss_fn(model(seq, st, sid), y_fp, y_mag, w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            d = min(ema_decay, (1 + step) / (10 + step))                     # EMA warm-up
            with torch.no_grad():                                           # EMA of the weights
                for e, p in zip(ema.parameters(), model.parameters()):
                    e.mul_(d).add_(p, alpha=1 - d)
        val = evaluate_logloss(ema, vl)
        print(f"epoch {ep}: val log loss {val:.4f}")
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(ema.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    ema.load_state_dict(best_state)
    return ema, best
```

## 参考文献

- Bai, Kolter, Koltun (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling (TCN). https://arxiv.org/abs/1803.01271
- van den Oord 等 (2016). WaveNet: a generative model for raw audio（因果膨胀卷积）. https://arxiv.org/abs/1609.03499
- Cho 等 (2014). Learning phrase representations using RNN encoder-decoder (GRU). *EMNLP 2014*.
- Hendrycks, Gimpel (2016). Gaussian error linear units (GELUs). https://arxiv.org/abs/1606.08415
- Smith, Topin (2019). Super-convergence: very fast training of neural networks using large learning rates (OneCycle).
- Loshchilov, Hutter (2019). Decoupled weight decay regularization (AdamW). *ICLR 2019*.
- Sundararajan, Taly, Yan (2017). Axiomatic attribution for deep networks (integrated gradients). *ICML 2017*.
- Kolm, Turiel, Westray (2023). Deep order flow imbalance. *Mathematical Finance* 33(4):1044–1081.
- Sirignano, Cont (2019). Universal features of price formation in financial markets. *Quantitative Finance* 19(9):1449–1459.
- Lucchese, Pakkanen, Veraart (2024). The short-term predictability of returns in order book markets. *International Journal of Forecasting* 40(4):1587–1621.
