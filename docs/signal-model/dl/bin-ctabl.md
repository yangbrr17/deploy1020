# BiN-CTABL：双线性网络 + 双向归一化

Tran、Iosifidis、Kanniainen、Gabbouj（TABL，IEEE TNNLS 2019）；Tran、Kanniainen、Gabbouj、Iosifidis（BiN，ICPR 2020）。返回 [深度学习总览](../deep-learning.md)。

**一句话**：把一个样本当成「特征 × 时间」的矩阵，用两侧各乘一个矩阵的双线性层代替全连接层，最后一层加时间注意力；输入先用样本自己的统计量沿两个方向归一化。**1.1 万个参数，是公开基准里最稳的模型**，适合当深度学习的基线。

## 1. 为什么选它

- **参数极少、推理快**：11,446 个参数；单样本 CPU 推理在 0.5 ms 以内（第 7 节）。
- **复现性最好**：LOBCAST 统一复现时，它在 FI-2010 上排第一（F1 82.6 ± 7.0），稳健性分 99.7，复现结果和原文基本一致。
- **结构可解释**：$W_1$ 是「同一时刻各档位怎么组合」，$W_2$ 是「同一特征怎么随时间组合」，注意力矩阵直接告诉你哪些时刻重要。
- **代价**：换到新数据后 F1 平均下降约 19.6%（LOBCAST），和所有模型一样；它擅长的是 FI-2010 这种已经预处理过的数据。

## 2. 主要创新点

1. **双线性层（BL）**：多元时间序列本身是二阶张量。把它拉平成向量喂给 MLP 会丢掉结构，参数也多。BL 只用两侧两个矩阵：参数量 $D D' + T T' + D' T'$，同尺寸的全连接层要 $(DT + 1) D' T'$。
2. **时间注意力（TABL）**：在 BL 中间插一个结构化的注意力，让**同一特征的不同时刻**互相竞争；再用一个可学习标量 $\lambda$ 在「用注意力」和「不用注意力」之间软混合。
3. **双向归一化（BiN）**：金融序列非平稳、价格和数量量级差很多，用训练期统计量做 z-score 会失效。BiN 用**单个样本自己的**统计量，同时沿时间维和特征维归一化，再用两个非负标量加权合成。只放在输入层，只多 102 个参数。

## 3. 输入和标签

- **张量**：$X \in \mathbb{R}^{D \times T}$，$D = 40$（10 档 × {卖价, 卖量, 买价, 买量}），$T = 10$（最近 10 个快照；FI-2010 每 10 个事件一个快照，所以覆盖约 100 个事件）。只用 40 维原始盘口，不用数据集里另外 104 维手工特征。
- **数据集里的批张量**是 `(B, T, D)`，进模型前转置成 `(B, D, T)`。
- **标签**：FI-2010 自带的平滑标签（见 [`features-and-targets.md`](../features-and-targets.md) 7.1 节 (C)），$\theta = 0.002$，$k \in \{1, 2, 3, 5, 10\}$ 对应 10–100 个事件。**我们不用这个标签**，改用首达三分类（见第 9 节）。

## 4. 数学

记号：$X \in \mathbb{R}^{D \times T}$，第 $t$ 列是时刻 $t$ 的盘口，第 $d$ 行是特征 $d$ 在过去 $T$ 步的轨迹。

### 4.1 双线性层 BL

$$
Y = \phi\big(W_1 X W_2 + B\big),\qquad
W_1 \in \mathbb{R}^{D' \times D},\quad W_2 \in \mathbb{R}^{T \times T'},\quad B \in \mathbb{R}^{D' \times T'}
$$

- $W_1 X = [W_1 x_1, \dots, W_1 x_T]$：对每个时刻的特征列做投影，建模**同一时刻**各档位、价量之间的关系。
- $X W_2$ 的第 $d$ 行是 $x^{(d)\top} W_2$：对每个特征的时间轨迹做投影，建模**同一特征**的时间演化。
- $\phi$ 在隐藏层用 ReLU，输出层用 softmax。

### 4.2 时间注意力双线性层 TABL

参数 $W_1 \in \mathbb{R}^{D' \times D}$，$W \in \mathbb{R}^{T \times T}$，$W_2 \in \mathbb{R}^{T \times T'}$，$B \in \mathbb{R}^{D' \times T'}$，标量 $\lambda \in [0, 1]$。

**第 1 步，特征投影**（保持时间顺序）：

$$
\bar X = W_1 X \in \mathbb{R}^{D' \times T}
$$

**第 2 步，时间打分**，$W$ 的对角线固定为 $1/T$：

$$
E = \bar X W,\qquad W_{tt} \equiv \frac{1}{T},\qquad
e_t = \frac{1}{T}\bar x_t + \sum_{k \ne t} W_{kt}\,\bar x_k
$$

也就是说，时刻 $t$ 的得分里自己的权重恒为 $1/T$，可学习的只有其他时刻对它的贡献。实现上用 $W_{\text{eff}} = W \odot (\mathbf{1}\mathbf{1}^\top - \mathbf{I}) + \mathbf{I}/T$。

**第 3 步，softmax 沿时间方向**（每一行内部归一化）：

$$
A_{ij} = \frac{\exp(E_{ij})}{\sum_{k=1}^{T}\exp(E_{ik})}
$$

竞争发生在**同一特征的不同时刻**之间，不同特征互不竞争。

**第 4 步，软注意力混合**：

$$
\tilde X = \lambda\,(\bar X \odot A) + (1 - \lambda)\,\bar X
$$

**第 5 步，时间投影**：

$$
Y = \phi\big(\tilde X W_2 + B\big) \in \mathbb{R}^{D' \times T'}
$$

**初始化**：$W_1, W_2$ 用 He 初始化；$W$ 全部元素为 $1/T$（开始时一视同仁）；$\lambda = 0.5$。

### 4.3 BiN

**沿时间维**：每个特征用它自己在 $T$ 步内的均值和标准差。

$$
\bar x_2 = \frac{1}{T}\sum_{t=1}^{T} x_t \in \mathbb{R}^{D},\qquad
\sigma_2 = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(x_t - \bar x_2)^{\odot 2}},\qquad
\tilde X_2 = \big(\gamma_2 \mathbf{1}_T^\top\big) \odot \frac{X - \bar x_2 \mathbf{1}_T^\top}{\sigma_2 \mathbf{1}_T^\top} + \beta_2 \mathbf{1}_T^\top
$$

**沿特征维**：每个时刻用它自己在 $D$ 个特征上的均值和标准差。

$$
\bar x_1 = \frac{1}{D}\sum_{d=1}^{D} x^{(d)} \in \mathbb{R}^{T},\qquad
\sigma_1 = \sqrt{\frac{1}{D}\sum_{d=1}^{D}(x^{(d)} - \bar x_1)^{\odot 2}},\qquad
\tilde X_1 = \big(\mathbf{1}_D \gamma_1^\top\big) \odot \frac{X - \mathbf{1}_D \bar x_1^\top}{\mathbf{1}_D \sigma_1^\top} + \mathbf{1}_D \beta_1^\top
$$

**合成**：

$$
\tilde X = \lambda_1 \tilde X_1 + \lambda_2 \tilde X_2,\qquad \lambda_1, \lambda_2 \ge 0
$$

每次更新后若 $\lambda$ 为负就置 0。可学习参数 $\gamma_2, \beta_2 \in \mathbb{R}^D$，$\gamma_1, \beta_1 \in \mathbb{R}^T$，$\lambda_1, \lambda_2$，共 $2D + 2T + 2$ 个（$D = 40, T = 10$ 时 102 个）。

**和 BatchNorm 的区别**：BiN 用的是每个样本自己的统计量，不依赖 batch，训练和推理行为一致，单样本推理也没问题。论文里输入层用 BatchNorm 反而有害（F1 从 77.63 降到 66.87）。

## 5. 结构和参数量

拓扑：$40 \times 10 \xrightarrow{\text{BiN}} 40 \times 10 \xrightarrow{\text{BL}} 60 \times 10 \xrightarrow{\text{BL}} 120 \times 5 \xrightarrow{\text{TABL}} 3 \times 1$

| # | 层 | 输入 | 输出 | 激活 | Dropout | 参数 |
|---|---|---|---|---|---|---|
| 1 | BiN$(40, 10)$ | $40 \times 10$ | $40 \times 10$ | — | — | $2 \cdot 40 + 2 \cdot 10 + 2 = 102$ |
| 2 | BL：$W_1 \in \mathbb{R}^{60 \times 40}$，$W_2 \in \mathbb{R}^{10 \times 10}$ | $40 \times 10$ | $60 \times 10$ | ReLU | 0.1 | $2400 + 100 + 600 = 3{,}100$ |
| 3 | BL：$W_1 \in \mathbb{R}^{120 \times 60}$，$W_2 \in \mathbb{R}^{10 \times 5}$ | $60 \times 10$ | $120 \times 5$ | ReLU | 0.1 | $7200 + 50 + 600 = 7{,}850$ |
| 4 | TABL：$W_1 \in \mathbb{R}^{3 \times 120}$，$W \in \mathbb{R}^{5 \times 5}$，$W_2 \in \mathbb{R}^{5 \times 1}$ | $120 \times 5$ | $3 \times 1$ | softmax | — | $360 + 25 + 5 + 3 + 1 = 394$ |
| | **合计** | | | | | **11,446** |

通用公式：

$$
\#\mathrm{BL} = D'D + TT' + D'T',\qquad
\#\mathrm{TABL} = D'D + T^2 + TT' + D'T' + 1,\qquad
\#\mathrm{BiN} = 2D + 2T + 2
$$

（不加 BiN 的 C(TABL) 是 11,344 个。）

## 6. 训练设置

| 项 | 原论文（BiN） | 我们建议 |
|---|---|---|
| 损失 | 加权交叉熵，权重 $\propto 1/N_c$ | 交叉熵，样本乘**唯一性权重**，**不加类权重**（要保校准，见总览） |
| 优化器 | Adam | AdamW，weight decay 0 |
| 学习率 | $10^{-3}$，第 11 个 epoch 降到 $10^{-4}$，第 71 个降到 $10^{-5}$ | 同左，或 OneCycle(max_lr $2 \times 10^{-3}$) |
| Epoch | 80 | 最多 80，早停盯验证集 log loss，patience 8 |
| Batch | TABL 原文 256；LOBCAST 用 32–128 | 256 |
| 正则 | Dropout 0.1；max-norm（每个神经元入权重的 $\ell_2$ 范数 ≤ 3–7）；不用 weight decay（论文说不适合双线性结构） | 同左 |

**max-norm 约束**：每次 `optimizer.step()` 之后把 $W_1$ 的每一行、$W_2$ 的每一列投影回半径 $c$ 的球：

$$
w \leftarrow w \cdot \frac{\min\big(\lVert w \rVert_2,\ c\big)}{\lVert w \rVert_2}
$$

## 7. 参数怎么选、结果怎么分析

**超参数和敏感性**：

- **深度**：A（只有 TABL）< B（BL + TABL）< C（BL + BL + TABL）。但**加了 BiN 以后 B 和 C 几乎一样**（H = 10：81.10 vs 81.04），说明 C 的第二个隐藏层是多余的。数据少的时候可以用 BiN-B(TABL)，参数只有 5,946 个。
- **注意力只放最后一层**：作者认为对高层表示做注意力更有益，其他位置没有验证。
- **BiN 只放输入层**：放在所有层没有额外收益。
- **max-norm 半径**在 {3, 5, 7} 里选。
- **对超参敏感**：LOBCAST 的超参搜索里约一半的运行发散（F1 ≤ 33%）；换数据集要重新搜 batch {16–256} 和学习率 {$10^{-2}$ – $10^{-5}$}。

**怎么分析训练好的模型**：

- **看注意力矩阵 $A$**：对一批样本取平均，看每个特征在哪些时刻权重高。如果 OBI 类特征的权重集中在最近 1–2 个时刻，说明信号是「瞬时」的，和文献「有效 horizon 约 2 次价格变动」一致。
- **看 $\lambda$**：$\lambda$ 接近 0 说明注意力几乎没用（模型退化成 C(BL)）。论文训练后 $\lambda \approx 0.95$，但官方代码的公式和论文是反的（第 8 节），所以要按你用的实现解读。
- **看 BiN 的 $\lambda_1, \lambda_2$**：哪个大，说明模型更依赖哪个方向的归一化。

**报告的结果**（F1，%）：

| 数据 | h = 10 | h = 20 | h = 50 | h = 100 | 来源 |
|---|---|---|---|---|---|
| FI-2010（原文） | 81.04 | 71.22 | 88.06 | — | BiN 论文 |
| FI-2010（复现） | 81.1 | 71.5 | 87.7 | 92.1 | LOBCAST |
| NASDAQ 2021 / 2022 | 57.0 / 58.4（k=1） | 62.4 / 62.0 | 63.5 / 60.4 | 59.1 / 53.2 | LOBCAST |
| TSLA / INTC / BTC | 58.7 / 72.7 / 73.4 | 48.8 / 66.6 / 61.3 | 42.2 / 54.0 / 47.1 | 38.8 / 41.1 / 40.6 | TLOB 论文 |

**推理时间**：原文在 i7-4790 上单样本前向 0.025 ms（C(TABL)）。我们在 4 核 Xeon 2.8 GHz 上用 PyTorch 2.5、单线程、batch 1 实测 BiN-CTABL **p50 0.42 ms，p99 0.59 ms**（大部分是 PyTorch 的调用开销）。i9 上会更快；手写 numpy 或 C++ 前向可以到几十微秒。

## 8. 已知问题

1. **$\lambda$ 的写法论文和代码相反**：论文是 $\lambda(\bar X \odot A) + (1 - \lambda)\bar X$，官方 Keras 和 PyTorch 移植都是 $\lambda \bar X + (1 - \lambda)(\bar X \odot A)$。初值 0.5、截断在 [0, 1] 时是同一个模型（$\lambda \to 1 - \lambda$），但对 $\lambda$ 的解读会反过来。下面的代码两种都支持。
2. **移植代码里有双重 softmax**：LOBCAST 和 TLOB 仓库的 `BiN_CTABL.forward` 最后做了 softmax，训练又用 `CrossEntropyLoss`（内部再做一次 log-softmax），梯度被压小。**下面的实现输出 logits**，softmax 只在损失里做一次。
3. **max-norm 实现不同**：论文是每个神经元的入权重，移植代码改成了整矩阵 Frobenius 范数 ≤ 10，宽松得多。
4. **BiN 的细节**：论文 Eq. (6c) 分母的 $\sigma_2$ 应为 $\sigma_1$（笔误）；代码用无偏标准差；代码只在时间维那一路给极小的标准差加了保护。
5. **泛化差**：从 FI-2010 换到 NASDAQ 数据，F1 平均下降约 19.6%。

## 9. 在我们的场景里怎么用

- **定位**：深度学习的**基线**。任何更复杂的网络都要先打赢它。
- **输入改造**：$D$ 换成我们的逐档 OFI（10 维）+ 相对中间价的价格和量（20 维）+ 事件元数据（约 6 维），$D \approx 36$；$T$ 取 10–20 个**决策时点**或固定事件数的快照。BiN 会自动处理量纲。
- **输出改造**：输出头换成首达三分类（先涨 / 先跌 / 没动），多个 τ 就并列几个 TABL 头。
- **损失**：交叉熵 × 唯一性权重，不加类权重；独立窗口上做温度缩放。
- **训练成本**：$10^7$ 个样本、30 个 epoch，在 i9 上用 PyTorch CPU 多线程大约几小时。

## 10. 代码

下面是按论文公式写的最小实现（已运行：参数量 11,446；把 TLOB 仓库的权重复制进来、用 `lambda_convention="code"` 并加 softmax 后，输出和仓库实现逐元素一致）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiN(nn.Module):
    """Bilinear normalisation (Tran et al. 2020, Eq. 5-7). x: (B, D, T) -> (B, D, T)."""

    def __init__(self, D: int, T: int, eps_std: float = 1e-4):
        super().__init__()
        self.gamma2 = nn.Parameter(torch.empty(D, 1))      # temporal-mode scale, one per feature
        self.beta2 = nn.Parameter(torch.zeros(D, 1))
        self.gamma1 = nn.Parameter(torch.empty(T, 1))      # feature-mode scale, one per time step
        self.beta1 = nn.Parameter(torch.zeros(T, 1))
        self.lambda1 = nn.Parameter(torch.tensor(0.5))
        self.lambda2 = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_normal_(self.gamma2)
        nn.init.xavier_normal_(self.gamma1)
        self.eps_std = eps_std

    def forward(self, x):                                  # (B, D, T)
        mu2, sd2 = x.mean(2, keepdim=True), x.std(2, keepdim=True)          # stats over T
        sd2 = torch.where(sd2 < self.eps_std, torch.ones_like(sd2), sd2)
        X2 = self.gamma2 * (x - mu2) / sd2 + self.beta2
        mu1, sd1 = x.mean(1, keepdim=True), x.std(1, keepdim=True)          # stats over D
        sd1 = torch.where(sd1 < self.eps_std, torch.ones_like(sd1), sd1)
        X1 = self.gamma1.T * (x - mu1) / sd1 + self.beta1.T
        return self.lambda1 * X1 + self.lambda2 * X2

    @torch.no_grad()
    def project_(self):
        self.lambda1.clamp_(min=0.0)
        self.lambda2.clamp_(min=0.0)


class BL(nn.Module):
    """Bilinear layer Y = relu(W1 X W2 + B): (B, D, T) -> (B, D', T')."""

    def __init__(self, D, T, D_out, T_out):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(D_out, D))
        self.W2 = nn.Parameter(torch.empty(T, T_out))
        self.B = nn.Parameter(torch.zeros(D_out, T_out))
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")

    def forward(self, X):
        return F.relu(self.W1 @ X @ self.W2 + self.B)


class TABL(nn.Module):
    """Temporal-attention bilinear layer: (B, D, T) -> (B, D', T') (no output activation).
    lambda_convention="paper": lam*(Xbar*A) + (1-lam)*Xbar ; "code": lam*Xbar + (1-lam)*(Xbar*A)."""

    def __init__(self, D, T, D_out, T_out, lambda_convention="paper"):
        super().__init__()
        self.T = T
        self.W1 = nn.Parameter(torch.empty(D_out, D))
        self.W = nn.Parameter(torch.full((T, T), 1.0 / T))
        self.W2 = nn.Parameter(torch.empty(T, T_out))
        self.B = nn.Parameter(torch.zeros(D_out, T_out))
        self.lam = nn.Parameter(torch.tensor(0.5))
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")
        self.register_buffer("eye", torch.eye(T))
        self.lambda_convention = lambda_convention

    def forward(self, X, return_attention=False):
        Xbar = self.W1 @ X                                 # (B, D', T)
        W = self.W * (1 - self.eye) + self.eye / self.T    # diagonal fixed to 1/T
        A = torch.softmax(Xbar @ W, dim=-1)                # softmax over time, per feature row
        if self.lambda_convention == "paper":
            Xt = self.lam * (Xbar * A) + (1 - self.lam) * Xbar
        else:
            Xt = self.lam * Xbar + (1 - self.lam) * (Xbar * A)
        Y = Xt @ self.W2 + self.B                          # (B, D', T')
        return (Y, A) if return_attention else Y

    @torch.no_grad()
    def project_(self):
        self.lam.clamp_(0.0, 1.0)


@torch.no_grad()
def _renorm_(W, max_val, dim=None):
    """Max-norm: dim=None whole-matrix Frobenius norm; dim=1 rows / dim=0 columns (per unit)."""
    n = torch.linalg.matrix_norm(W) if dim is None else W.norm(dim=dim, keepdim=True)
    W.mul_(torch.clamp(n, max=max_val) / (n + 1e-8))


class BiNCTABL(nn.Module):
    """40x10 -BiN-> 40x10 -BL-> 60x10 -BL-> 120x5 -TABL-> 3x1. Returns logits (B, 3)."""

    def __init__(self, D=40, T=10, n_classes=3, use_bin=True, dropout=0.1, lambda_convention="paper"):
        super().__init__()
        self.bin = BiN(D, T) if use_bin else nn.Identity()
        self.bl1 = BL(D, T, 60, 10)
        self.bl2 = BL(60, 10, 120, 5)
        self.tabl = TABL(120, 5, n_classes, 1, lambda_convention)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, T, D)
        x = self.bin(x.transpose(1, 2))                    # (B, D, T)
        x = self.drop(self.bl1(x))                         # (B, 60, 10)
        x = self.drop(self.bl2(x))                         # (B, 120, 5)
        return self.tabl(x).squeeze(-1)                    # (B, 3) logits

    @torch.no_grad()
    def constrain_(self, max_val=5.0):
        """Call after every optimizer.step(): per-unit max-norm (paper) and lambda projections."""
        for m in (self.bl1, self.bl2, self.tabl):
            _renorm_(m.W1, max_val, dim=1)
            _renorm_(m.W2, max_val, dim=0)
        if isinstance(self.bin, BiN):
            self.bin.project_()
        self.tabl.project_()


if __name__ == "__main__":
    m = BiNCTABL()
    print(m(torch.randn(8, 10, 40)).shape, sum(p.numel() for p in m.parameters()))   # (8, 3) 11446
```

训练循环的关键几行（这里用模拟数据代替真实的 `loader`；需要上面的 `BiNCTABL`）：

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# replace with real data: x (N, T, D) windows, y (N,) first-passage class, w (N,) uniqueness weights
N, T, D = 2048, 10, 36
loader = DataLoader(TensorDataset(torch.randn(N, T, D), torch.randint(0, 3, (N,)), torch.rand(N)),
                    batch_size=256, shuffle=True)

model = BiNCTABL(D=D, T=T)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y, w in loader:
    loss = (F.cross_entropy(model(x), y, reduction="none") * w).sum() / w.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    model.constrain_(max_val=5.0)              # per-unit max-norm + lambda projections
print("last batch loss", float(loss))
```

## 参考文献

- Tran, Iosifidis, Kanniainen, Gabbouj (2019). Temporal attention-augmented bilinear network for financial time-series data analysis. *IEEE TNNLS* 30(5):1407–1418. https://arxiv.org/abs/1712.00975 ；代码 https://github.com/viebboy/TABL （无 LICENSE 文件）
- Tran, Kanniainen, Gabbouj, Iosifidis (2020). Data normalization for bilinear structures in high-frequency financial time-series. *ICPR 2020*. https://arxiv.org/abs/2003.00598
- Prata 等 (2024). LOB-based deep learning models for stock price trend prediction: a benchmark study (LOBCAST). *Artificial Intelligence Review* 57. https://arxiv.org/abs/2308.01915
- Berti, Kasneci (2025). TLOB. https://arxiv.org/abs/2502.15757 ；仓库里有 BiN-CTABL 的 PyTorch 实现（MIT）：https://github.com/LeonardoBerti00/TLOB

未核实：TNNLS 正式版的数字是否和 arXiv v1 一致；BiN 论文里的初始化方式（上面用的是移植代码的初始化）。
