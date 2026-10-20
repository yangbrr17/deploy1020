# 事件到达模型：下一笔什么时候来

2026-10 调研整理。问题是：下一笔订单 / 成交 / 中间价变动在什么时候到达，应该假设什么分布。本页给出经济学和金融计量里成熟的几条路线（泊松、ACD、Hawkes、排队模型）的数学、估计方法、检验方法，以及怎么接进我们的 hazard 模型。总览见 [`modeling-guide.md`](modeling-guide.md)，hazard 模型本身见 [`ml-pipeline.md`](ml-pipeline.md) 第 1.5 节。**拿到数据后怎么逐项检验（要什么字段、每个假设对应哪个分布、用什么统计量、怎么判定）见 [`arrival-tests.md`](arrival-tests.md)，代码在 [`src/arrivals/`](../../src/arrivals/__init__.py)。** 调研的阅读顺序见第 13 节，本页全部公式的 LaTeX 源码见第 14 节。

## 结论

1. **泊松（间隔服从指数分布）只能当零假设，不能当工作模型。** 实证上间隔的变异系数大于 1（过度离散），相邻间隔正相关（聚集），日内有很强的季节性，计数有长记忆。用泊松会同时低估「短时间内连续来很多笔」和「很长时间一笔不来」这两种情形。
2. **成熟的参考有三条线，各管一件事：**
   - **计量经济学的 ACD 族**（Engle、Russell 1998）。它是 GARCH 在「间隔」上的对应物，属于加速失效时间（AFT）模型。适合回答「下一笔大概要等多久」。
   - **点过程的 Hawkes 自激过程**（Hawkes 1971）。它直接给出条件强度，也就是 hazard。适合刻画「一笔成交引来更多成交」的聚集。
   - **排队 / 订单簿模型**（Cont–Stoikov–Talreja 2010、Cont–de Larrard 2013、queue-reactive 2015）。强度是盘口状态的函数，中间价变动时间是「最优档队列被打光的首达时间」。
3. **「tick 级大部分时候价格不变」这件事，排队模型解释得最好。** 盘口事件很密（毫秒级），但绝大多数只是改变队列长度；中间价只在某一侧最优档被吃光或撤光时才动。所以「下一次中间价变动什么时候来」的头号决定因素是**最优档队列长度**和**两侧的净流入速度**。两侧订单流平衡时，这个等待时间是重尾的，期望甚至是无穷；不平衡时按指数速度衰减。
4. **和我们现有方案的关系：离散时间 hazard 就是条件强度在时间格点上的离散化。** Hawkes 强度、ACD 的期望间隔、日内季节因子、队列状态都可以直接当 hazard 的协变量。ACD 和 Hawkes 不用替代 hazard 模型，它们的作用是提供特征、基线和检验工具（时间重标度检验）。

## 1. 先分清三种「下一笔」

| 事件 | 典型间隔 | 这一步改不改中间价 | 合适的模型 |
|---|---|---|---|
| 盘口事件（新增、撤单、成交） | 毫秒级，活跃时亚毫秒 | 大多不改 | 多元 Hawkes、queue-reactive |
| 成交 | 毫秒到秒级 | 只有吃光最优档时才改 | ACD、Hawkes |
| 中间价变动 | 大 tick 股票可以是秒到分钟级 | 本身就是变化 | 队列耗尽首达时间、竞争风险 hazard |

三者的关系：中间价变动是盘口事件流的一个**稀疏子序列**。设每个盘口事件引起中间价变动的概率是 $p$，盘口事件强度是 $\lambda_{\text{evt}}$，那么中间价变动强度近似为 $p\,\lambda_{\text{evt}}$。但这里 $p$ 不是常数，而是强烈依赖队列长度：队列只剩 1–2 单时 $p$ 很大，队列很长时 $p$ 接近 0。所以对中间价，**「什么时候动」主要由状态决定，而不是由时间决定**，这就是第 6–7 节的排队模型。

## 2. 泊松过程：零假设

### 2.1 定义与性质

强度为 $\lambda$ 的齐次泊松过程 $N(t)$ 满足

\[
\Pr\{N(t+\Delta)-N(t)=k\}=e^{-\lambda\Delta}\frac{(\lambda\Delta)^k}{k!},\qquad
X_i=t_i-t_{i-1}\overset{\text{iid}}{\sim}\mathrm{Exp}(\lambda),\qquad \Pr(X>x)=e^{-\lambda x}.
\]

几条性质：

- **无记忆性**：$\Pr(X>s+x\mid X>s)=\Pr(X>x)$，即 hazard 恒为 $\lambda$。「已经等了多久」对「还要等多久」没有信息。
- **离散程度**：间隔的变异系数 $\mathrm{CV}=\sigma_X/\mathbb E X=1$；计数的 Fano 因子 $\mathrm{Var}\,N/\mathbb E N=1$。
- **叠加与拆分**：独立泊松流叠加后仍是泊松流，强度相加；按概率 $p$ 独立拆分也仍是泊松流。所以「下一个事件是哪一类」的概率是 $\lambda_k/\sum_l\lambda_l$，和「什么时候发生」相互独立。

### 2.2 零智能模型

Smith、Farmer、Gillemot、Krishnamurthy (2003) 把订单簿写成三条独立泊松流：市价单以每单位时间 $\mu$ 股到达；限价单以每单位价格、每单位时间 $\alpha$ 股在价格轴上均匀到达；每股挂单以常数速率 $\delta$ 被撤。由维度分析和平均场近似可以推出价差、深度和冲击，例如价差粗略近似为 $s\approx\mu/(2\alpha)$。

Cont、Stoikov、Talreja (2010) 做了更贴近数据的版本：距对手方最优价 $i$ 档处的限价单速率为 $\lambda(i)=K i^{-\beta}$（幂律），市价单速率为常数 $\mu$，撤单率与该档挂单数成正比（有 $x$ 单时为 $\theta(i)\,x$）。订单簿因此是一个连续时间马尔可夫链，「下一次中间价上涨的概率」「限价单在中间价变动前成交的概率」等都可以用生灭过程首达时间的拉普拉斯变换算出来。

**优点**：可解析，参数少，强度就是「事件数 ÷ 暴露时间」。**用途**：零模型、模拟器起点、检验的基准。

### 2.3 为什么实证上不成立

| 事实 | 内容 | 来源 |
|---|---|---|
| 过度离散 | 间隔的 $\sigma/\mu>1$ | Engle、Russell (1998)；Bauwens、Hautsch 综述 |
| 聚集 | 去掉确定性季节后，间隔仍有强自相关；ACD(1,1) 的典型估计 $\alpha+\beta\in(0.85,1)$，$\alpha\in(0.01,0.15)$ | Engle、Russell (1998)；Bauwens、Hautsch 综述 §3.1 |
| 日内季节性 | 活跃度大体呈 U 形：开盘、收盘附近间隔短，午间间隔长 | Engle、Russell (1998)；Ghysels、Jasiak (1998) |
| hazard 非单调 | 价格间隔的条件 hazard 呈驼峰形（先升后降） | Grammig、Maurer (2000)，因此提出 Burr-ACD |
| 计数长记忆 | 收益绝对值的长程相关主要来自成交笔数 $N_{\Delta t}$ 的长程相关 | Plerou 等 (2000) |
| 自激核呈幂律 | 中间价变动的 Hawkes 核积分约为 1，衰减呈幂律 | Hardiman、Bercot、Bouchaud (2013)；Bacry、Dayri、Muzy (2012) |

Cont、de Larrard (2012) 直接把一只美国大盘股某日的订单簿事件间隔分位数和同均值指数分布对比，结论是「far from being exponential」。

## 3. 非齐次泊松与 Cox 过程

**非齐次泊松（NHPP）**：强度是确定性函数 $\lambda(t)$，常取 $\lambda(t)=\lambda_0/s(t)$，$s(t)$ 是日内季节因子（该时段的平均间隔）。累积强度 $\Lambda(t)=\int_0^t\lambda(u)\,du$，计数 $N(a,b]\sim\mathrm{Poisson}\big(\Lambda(b)-\Lambda(a)\big)$，对数似然

\[
\ell=\sum_{i}\log\lambda(t_i)-\int_0^T\lambda(u)\,du .
\]

它能刻画 U 形季节性，但刻画不了聚集：去掉季节后各事件仍然独立。

**Cox 过程（双随机泊松）**：强度本身是随机过程，例如 $\lambda(t)=X(t)/s(t)$，$X$ 是一个正值潜变量（CIR 过程、马尔可夫切换等）。由全方差公式

\[
\mathrm{Var}\,N(0,t]=\mathbb E\,\Lambda(t)+\mathrm{Var}\,\Lambda(t)\;\ge\;\mathbb E\,N(0,t],
\]

所以 Cox 过程天然过度离散。它和 Hawkes 的区别：Cox 的强度由**外生**潜变量驱动，Hawkes 的强度由**自身历史**驱动。似然通常要滤波或模拟，比 ACD、Hawkes 难算，实务上用得少。

## 4. ACD：对「间隔」做 GARCH

### 4.1 模型

Engle、Russell (1998)。设季节调整后的间隔 $x_i=(t_i-t_{i-1})/s(t_{i-1})$，

\[
x_i=\psi_i\,\varepsilon_i,\qquad \varepsilon_i\overset{\text{iid}}{\sim}F_\varepsilon,\quad \mathbb E\varepsilon_i=1,\qquad
\psi_i=\mathbb E[x_i\mid\mathcal F_{i-1}]=\omega+\alpha x_{i-1}+\beta\psi_{i-1},
\]

其中 $\omega>0$，$\alpha,\beta\ge 0$。

- **平稳与矩**：$\alpha+\beta<1$ 时 $\mathbb E x=\omega/(1-\alpha-\beta)$；协方差平稳还要求 $(\alpha+\beta)^2-\alpha^2\sigma_\varepsilon^2<1$。自相关函数 $\rho_n=(\alpha+\beta)\rho_{n-1}$（$n\ge2$），形状和 ARMA(1,1) 一样，几何衰减。
- **和 hazard 的关系**：

\[
\lambda(t\mid\mathcal F_{t})=\frac{1}{\psi_{N(t)+1}}\;h_\varepsilon\!\left(\frac{t-t_{N(t)}}{\psi_{N(t)+1}}\right),
\]

$h_\varepsilon$ 是 $\varepsilon$ 的 hazard。这说明 **ACD 是 AFT 模型**：历史信息只把时间轴整体拉伸或压缩，hazard 的形状由 $\varepsilon$ 的分布决定。

### 4.2 误差分布与似然

| $\varepsilon$ 的分布 | hazard 形状 | 文献 |
|---|---|---|
| 指数（EACD） | 常数 | Engle、Russell (1998) |
| Weibull（WACD），形状参数 $k$ | $k<1$ 递减，$k>1$ 递增 | Engle、Russell (1998) |
| Burr | 可以是驼峰形 | Grammig、Maurer (2000) |
| 广义 gamma | 可以非单调，含指数、Weibull 为特例 | Lunde（工作论文） |

**EACD 的 QML 对数似然**：只要 $\psi_i$ 设对，即使误差分布设错也是一致的（和 GARCH 的 QML 结论同理）：

\[
\ell(\theta)=-\sum_{i=1}^n\left[\log\psi_i+\frac{x_i}{\psi_i}\right].
\]

**WACD 的对数似然**：令 $g=\Gamma(1+1/k)$，使 $\mathbb E\varepsilon=1$：

\[
\ell(\theta,k)=\sum_{i=1}^n\left[\log k-\log x_i+k\log\frac{g\,x_i}{\psi_i}-\left(\frac{g\,x_i}{\psi_i}\right)^{k}\right].
\]

它的积分 hazard（广义残差）$\hat e_i=(g x_i/\hat\psi_i)^{k}$ 在模型正确时应为 iid Exp(1)，第 9 节的检验就用它。

### 4.3 Log-ACD：方便加协变量

Bauwens、Giot (2000)：

\[
\log\psi_i=\omega+\beta\log\psi_{i-1}+\alpha\,g(\varepsilon_{i-1})+\gamma^\top z_{i-1},\qquad g(\varepsilon)=\log\varepsilon\ \ \text{或}\ \ \varepsilon .
\]

不需要符号约束，可以直接加入价差、成交量、盘口不平衡等协变量 $z_{i-1}$。平稳条件是 $|\beta|<1$。

### 4.4 季节调整一定要先做

标准两步法：

1. 估计 $s(t)=\mathbb E[t_i-t_{i-1}\mid\text{时段}]$：按半小时分箱求均值，再用三次样条平滑。
2. 用 $x_i=(t_i-t_{i-1})/s(t_{i-1})$ 拟合 ACD。

不做第 1 步会怎样：第 11 节的演示里，真值 $\alpha+\beta=0.9$ 的间隔序列乘上日内季节后直接拟合 WACD，得到 $\hat\alpha+\hat\beta\approx0.98$；做季节调整后回到约 0.91。持久性从 0.9 变成 0.98，冲击的半衰期就从约 7 笔（$\ln 0.5/\ln 0.9$）变成约 34 笔：季节性被误当成了持久性（和 GARCH 里的 IGARCH 假象同理）。另外 ACD 的似然面在 $\alpha+\beta\to1$ 附近有平坦的脊，单起点优化容易停在 $\hat\omega\approx0$、$\hat\beta\approx0.98$ 这类假解上，下面的代码因此用了多起点。

### 4.5 代码：WACD 模拟、估计、季节调整

```python
import numpy as np
from scipy import optimize, stats, special, integrate
from scipy.signal import lfilter
from scipy.interpolate import CubicSpline


def numerical_hessian(f, p, rel=1e-4):
    """Central-difference Hessian of a scalar function f at p (for MLE standard errors)."""
    p = np.asarray(p, float)
    k = len(p)
    H = np.empty((k, k))
    h = rel * np.maximum(np.abs(p), 1e-3)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(p + ei + ej) - f(p + ei - ej)
                                 - f(p - ei + ej) + f(p - ei - ej)) / (4 * h[i] * h[j])
    return H


def simulate_wacd(omega, a, b, k, n, rng, burn=1000):
    """x_i = psi_i * eps_i, psi_i = omega + a x_{i-1} + b psi_{i-1}, eps ~ Weibull(k)/Gamma(1+1/k)."""
    g = special.gamma(1.0 + 1.0 / k)
    eps = rng.weibull(k, size=n + burn) / g
    x = np.empty(n + burn)
    psi = omega / (1.0 - a - b)
    xprev = psi
    for i in range(n + burn):
        psi = omega + a * xprev + b * psi
        x[i] = psi * eps[i]
        xprev = x[i]
    return x[burn:]


def acd_psi(x, omega, a, b):
    """psi_i = omega + a x_{i-1} + b psi_{i-1}; psi_0 = sample mean. Vectorised with lfilter."""
    u = np.empty_like(x)
    u[0] = np.mean(x)
    u[1:] = omega + a * x[:-1]
    psi = lfilter([1.0], [1.0, -b], u[1:], zi=[b * u[0]])[0]
    return np.concatenate([[u[0]], psi])


def wacd_loglik(params, x):
    omega, a, b, k = params
    psi = acd_psi(x, omega, a, b)
    g = special.gamma(1.0 + 1.0 / k)
    z = g * x / psi
    return np.sum(np.log(k) - np.log(x) + k * np.log(z) - z ** k), psi


def fit_wacd(x):
    """MLE on log-parameters; rejects a + b >= 1 (non-stationary)."""
    def nll(th):
        omega, a, b, k = np.exp(th)
        if a + b >= 0.9999:
            return 1e12
        return -wacd_loglik((omega, a, b, k), x)[0]
    best = None
    for a0, b0 in [(0.1, 0.8), (0.05, 0.9), (0.2, 0.6)]:      # multi-start: the surface has flat ridges
        x0 = np.log([(1 - a0 - b0) * np.mean(x), a0, b0, 1.0])
        res = optimize.minimize(nll, x0, method="Nelder-Mead",
                                options=dict(xatol=1e-7, fatol=1e-7, maxiter=8000))
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x), -best.fun


def wacd_integrated_hazard(params, x):
    """Generalised residuals (g x_i / psi_i)^k, iid Exp(1) if the model is right."""
    omega, a, b, k = params
    psi = acd_psi(x, omega, a, b)
    g = special.gamma(1.0 + 1.0 / k)
    return (g * x / psi) ** k


def diurnal_adjust(x, tod, day_len, n_bins=26):
    """s(tod) = E[x | time of day]: bin means smoothed by a natural cubic spline; returns x / s."""
    edges = np.linspace(0, day_len, n_bins + 1)
    idx = np.clip(np.digitize(tod, edges) - 1, 0, n_bins - 1)
    means = np.array([x[idx == j].mean() for j in range(n_bins)])
    centres = 0.5 * (edges[1:] + edges[:-1])
    s = CubicSpline(centres, means, bc_type="natural")(tod)
    return x / s, s
```

## 5. Hawkes：自激过程

### 5.1 定义

Hawkes (1971)。单元 Hawkes 的条件强度

\[
\lambda(t)=\mu+\sum_{t_i<t}\phi(t-t_i),\qquad \phi\ge 0 .
\]

$\mu$ 是外生的基准强度，$\phi$ 是激发核：每来一个事件，之后一段时间的强度被抬高。常用两种核：

- **指数核** $\phi(t)=\alpha e^{-\beta t}$，可推广为多指数之和 $\sum_j\alpha_je^{-\beta_jt}$。计算最便宜，有 $O(N)$ 递推。
- **幂律核** $\phi(t)=K/(t+c)^{1+\gamma}$。更贴近实证（第 2.3 节），但没有递推，常用多个指数之和近似。

**分支比** $n=\|\phi\|_1=\int_0^\infty\phi(t)\,dt$，指数核下 $n=\alpha/\beta$。

- **平稳条件**：$n<1$，平均强度 $\bar\lambda=\mu/(1-n)$。
- **聚类表示**（Hawkes、Oakes 1974）：「移民」事件按强度 $\mu$ 的泊松流到达，每个事件生成的直接后代数服从均值为 $n$ 的泊松分布。所以 $n$ 可以理解为**内生事件的占比**：$n=0.7$ 表示约七成事件是被之前的事件「引来」的。

### 5.2 多元 Hawkes

\[
\lambda_m(t)=\mu_m+\sum_{l=1}^{M}\sum_{t^l_j<t}\phi_{ml}(t-t^l_j),\qquad m=1,\dots,M .
\]

平稳条件是核范数矩阵 $\|\Phi\|=(\|\phi_{ml}\|_1)$ 的谱半径小于 1，此时 $\bar{\boldsymbol\lambda}=(I-\|\Phi\|)^{-1}\boldsymbol\mu$。微观结构里常见的维度划分：

- 中间价上 / 下两维：上下跳互相激发可以复现短尺度的均值回复和 Epps 效应（Bacry、Delattre、Hoffmann、Muzy 2013）。
- 买 / 卖 × 市价单 / 限价单 / 撤单：Bacry、Jaisson、Muzy (2016) 对一档订单簿建了 8 维 Hawkes，核在 $10\,\mu s$ 到 $100\,s$ 上呈幂律。
- 状态依赖：Morariu-Patrichi、Pakkanen (2022) 让激发核依赖价差、队列不平衡等离散状态，发现盘口失衡时内生性更高。这一步把 Hawkes 和第 6 节的 queue-reactive 接了起来。

### 5.3 似然与指数核递推

一般点过程的对数似然是 $\ell=\sum_i\log\lambda(t_i)-\int_0^T\lambda(u)\,du$。指数核单元 Hawkes 用 Ozaki (1979) 的递推，$O(N)$：

\[
A_1=0,\qquad A_i=e^{-\beta(t_i-t_{i-1})}\,(1+A_{i-1}),\qquad \lambda(t_i)=\mu+\alpha A_i,
\]

\[
\ell(\mu,\alpha,\beta)=\sum_{i=1}^{N}\log\!\big(\mu+\alpha A_i\big)-\mu T-\frac{\alpha}{\beta}\sum_{i=1}^{N}\Big(1-e^{-\beta(T-t_i)}\Big).
\]

多元时对每对 $(m,l)$ 维护一个递推量 $A^{ml}$；若同一目标维的衰减率都相同（$\beta_{ml}=\beta_m$），每个目标维只需 $M$ 个递推量。

**相邻事件之间的补偿量**（第 9 节的检验要用）：

\[
\Lambda(t_{i-1},t_i)=\mu\,\Delta_i+\frac{\alpha}{\beta}\,(1+A_{i-1})\big(1-e^{-\beta\Delta_i}\big),\qquad \Delta_i=t_i-t_{i-1}.
\]

**模拟**用 Ogata (1981) 的 thinning：指数核下两次事件之间强度单调递减，所以「当前强度」是合法上界。按上界抽一个指数等待时间得到候选点，再以 $\lambda(t)/\bar\lambda$ 的概率接受。

### 5.4 分支比是否接近 1：一个未定的争论

- Filimonov、Sornette (2012)：美国股指期货 1998–2010，用指数核估计的内生度从约 0.3 升到 0.7 以上。
- Hardiman、Bercot、Bouchaud (2013)：同类数据用非参数幂律核，核积分在 1998–2011 年**都约等于 1**，即长期处在临界附近。
- Filimonov、Sornette (2015) 反驳：基准强度的体制变化、季节性、多段泊松混合，都会让估计出的 $n$ 虚高到接近 1。
- Lallouache、Challet (2016)：外汇数据，1 小时窗口用双指数核、全天窗口用 2–3 个指数核能通过检验，内生度约 0.8，高但亚临界；窗口再长就因非平稳而失败。

**对我们的含义**：$n$ 的估计对季节性、窗口长度、核的形式都很敏感。要用就在固定短窗口上滚动估计、先做季节调整，只当一个「当前有多拥挤」的状态特征，不要当结构性结论。

### 5.5 代码：Hawkes 模拟、估计、补偿量

```python
def simulate_hawkes_exp(mu, alpha, beta, T, rng):
    """Ogata (1981) thinning; between events the intensity only decays, so the current value bounds it."""
    t, events = 0.0, []
    excite = 0.0                         # sum_j alpha * exp(-beta (t - t_j))
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1.0 / lam_bar)
        if t + w > T:
            break
        excite *= np.exp(-beta * w)
        t += w
        if rng.uniform() * lam_bar <= mu + excite:
            events.append(t)
            excite += alpha
    return np.asarray(events)


def hawkes_exp_loglik(params, t, T):
    """Exact log-likelihood with the Ozaki (1979) O(N) recursion; also returns A_i."""
    mu, alpha, beta = params
    decay = np.exp(-beta * np.diff(t))
    A = np.zeros_like(t)
    a = 0.0
    for i in range(1, len(t)):
        a = decay[i - 1] * (1.0 + a)
        A[i] = a
    comp = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    return np.sum(np.log(mu + alpha * A)) - comp, A


def fit_hawkes_exp(t, T):
    """MLE over log-parameters; rejects alpha/beta >= 1 (non-stationary)."""
    def nll(theta):
        p = np.exp(theta)
        if p[1] / p[2] >= 0.999:
            return 1e12
        return -hawkes_exp_loglik(p, t, T)[0]
    x0 = np.log([0.5 * len(t) / T, 0.5, 1.0])
    res = optimize.minimize(nll, x0, method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
    return np.exp(res.x), -res.fun


def hawkes_exp_compensator_increments(params, t):
    """Lambda(t_{i-1}, t_i); iid Exp(1) under the true model (time-rescaling theorem)."""
    mu, alpha, beta = params
    _, A = hawkes_exp_loglik(params, t, t[-1])
    dt = np.diff(t)
    return mu * dt + (alpha / beta) * (1.0 + A[:-1]) * (1.0 - np.exp(-beta * dt))
```

## 6. Queue-reactive：状态依赖的泊松

Huang、Lehalle、Rosenbaum (2015)。

- **状态**：$X(t)=(q_{-K},\dots,q_{-1},q_1,\dots,q_K)$，单位是平均事件大小（AES），实证取 $K=3$。
- **动态**：在参考价不变的区间内，$X$ 是连续时间马尔可夫跳过程。第 $i$ 档以 $\lambda^L_i(q)$ 加 1（限价单），以 $\lambda^C_i(q)+\lambda^M_i(q)$ 减 1（撤单 + 市价单）。强度**只依赖当前状态**。
- **三种设定**：Model I 各队列独立、只依赖自身长度；Model II 允许相邻档相互依赖；Model III 加入参考价跳动机制，含外生重置概率 $\theta_{\text{reinit}}$。
- **估计**：每个状态下等待时间是指数分布，所以极大似然估计就是计数除以暴露时间：

\[
\hat\Lambda(n)=\big(\overline{\Delta t\mid q=n}\big)^{-1},\qquad
\hat\lambda^{L}(n)=\hat\Lambda(n)\cdot\frac{\#\{\text{新增},\,q=n\}}{\#\{q=n\}} ,
\]

$\hat\lambda^C,\hat\lambda^M$ 同理。

- **实证结果**（欧洲大 tick 股票）：
  - 一档撤单率随队列长度**凹增**，约 25 AES 后变平，和「撤单率正比于队列长度」的经典假设不同；
  - 一档市价单强度随一档挂单量大致**指数递减**（队列越长，越少有人来吃）；
  - 泊松基准会**高估**限价单的成交概率。
- **定位**：做模拟器订单流最实用的起点。比 Cont–Stoikov–Talreja 多了状态依赖，估计又只是计数除以时间。

## 7. 下一次中间价变动：队列耗尽的首达时间

这一节直接回答「价格大部分时候不动，下一次什么时候动」。

### 7.1 Cont–de Larrard 模型

Cont、de Larrard (2013)。只看最优买一和卖一两个队列，价差恒为 1 tick。每侧限价单速率 $\lambda$、市价单速率 $\mu$、撤单速率 $\theta$，都是独立泊松流，单位大小。某一侧队列被打光时价格移动一档，新的队列对从某个分布 $f$ 中重新抽取。

**耗尽时间是生灭过程的首达时间。** 卖一队列是出生率 $\lambda$、死亡率 $c=\mu+\theta$ 的生灭过程。记 $\sigma^a=\inf\{t:q^a_t=0\}$，初始队列为 $x$，其拉普拉斯变换

\[
\mathcal L(s,x)=\mathbb E\big[e^{-s\sigma^a}\mid q^a_0=x\big]=\left(\frac{(\lambda+c+s)-\sqrt{(\lambda+c+s)^2-4\lambda c}}{2\lambda}\right)^{x},
\]

反演得密度（$I_x$ 为第一类修正 Bessel 函数）：

\[
f_{\sigma^a}(t\mid x)=\frac{x}{t}\left(\frac{c}{\lambda}\right)^{x/2}I_x\!\big(2\sqrt{\lambda c}\,t\big)\,e^{-(\lambda+c)t}.
\]

**到下一次中间价变动的时间**是两侧中先耗尽的那个：$\tau=\sigma^a\wedge\sigma^b$。两侧独立，所以

\[
\Pr(\tau>t\mid q^a_0=x,\ q^b_0=y)=\Pr(\sigma^a>t\mid x)\cdot\Pr(\sigma^b>t\mid y).
\]

这就是「$t$ 秒内中间价不动」的概率，只依赖两侧队列长度和三个速率。

### 7.2 平衡情形：重尾，期望无穷

两侧订单流平衡（$\lambda=c$）时，原文给出

\[
\Pr(\sigma^a>t\mid x)\sim\frac{x}{\sqrt{\pi\lambda}}\,t^{-1/2},\qquad
\Pr(\tau>t\mid x,y)\sim\frac{xy}{\pi\lambda}\,t^{-1},\qquad \mathbb E[\tau\mid x,y]=\infty .
\]

直觉：平衡时队列是对称随机游走，回到 0 的时间本身就是重尾的。所以「价格长时间不动」并不罕见，正是这个模型的自然结果。我们的数值核对与之一致：$x=5$、$\lambda=12$ 时，$t=500$ 处 Bessel 精确值 0.0364，渐近式 0.0364，模拟 0.0368。

**先涨的概率**（平衡情形，记买一队列 $n$、卖一队列 $p$）：

\[
\phi(n,p)=\Pr(\sigma^a<\sigma^b)=\frac{1}{\pi}\int_0^{\pi}\Big(2-\cos t-\sqrt{(2-\cos t)^2-1}\Big)^{p}\,\frac{\sin(nt)\cos(t/2)}{\sin(t/2)}\,dt .
\]

它**和订单流速率无关**，只依赖两侧队列长度。例如 $\phi(5,2)=0.756$、$\phi(10,1)=0.936$，我们用模拟核对过（0.759、0.931）。这是 queue imbalance 能预测下一次价格方向的结构性解释。

**扩散极限的波动率**：$\sigma=\delta\sqrt{\pi\lambda/D(f)}$，$D(f)=\sum_{i,j}ij\,f(i,j)$，即「tick² × 订单到达率 ÷ 深度」。tick 越大、深度越深，价格越「黏」。

### 7.3 不平衡情形：指数截断

原文对 $\lambda<c$ 写的是 $\Pr(\sigma^a>t)\sim\frac{x(\lambda+c)}{2\lambda(c-\lambda)}\,t^{-1}$，即 $\tau$ 的尾指数为 2。**我们的核对表明这一点不对**：

- $\mathcal L(s,x)$ 在 $s=0$ 处导数有限，所以 $\mathbb E\sigma^a=x/(c-\lambda)$ 有限（标准生灭过程结论）。
- $\mathcal L(s,x)$ 的奇点在 $s^\ast=-(\sqrt c-\sqrt\lambda)^2<0$，所以尾部按**指数**衰减，大致为

\[
\Pr(\sigma^a>t)\propto t^{-3/2}\,e^{-(\sqrt c-\sqrt\lambda)^2\,t}.
\]

- 数值上，$x=5$、$\lambda=12$、$c=13$：模拟均值 4.90，理论 $x/(c-\lambda)=5$；$\Pr(\sigma>200)$ 的 Bessel 精确值 $1.19\times10^{-4}$、模拟 $1.0\times10^{-4}$，而 $1/t$ 渐近式给出 $2.6\times10^{-2}$。

**实务含义**：真实盘口通常接近平衡，截断率 $(\sqrt c-\sqrt\lambda)^2$ 很小，所以在很宽的时间区间里看起来像 $t^{-1/2}$ 幂律，之后才被指数截断。原文对一只美国大盘股的估计是 $\hat\lambda=2204$、$\hat c=2331$（每秒批次数），截断率约 $1.78\,\text{s}^{-1}$，即约 0.56 秒的时间尺度，相对约 0.45 ms 的事件间隔已经很长；平均耗尽时间约 $x/127$ 秒。**「$\tau$ 内不动」的概率主要由队列长度 $x,y$ 和净流 $c-\lambda$ 决定，这两个量必须是 hazard 模型的核心协变量。**

### 7.4 放宽假设：heavy-traffic 版本

Cont、de Larrard (2012) 放宽了「泊松、单位大小、两侧独立」，允许自激、ACD 类订单流。扩散极限下两侧队列是正象限内相关系数 $\rho$ 的二维布朗运动，平衡时先涨概率（$x$ 为买一、$y$ 为卖一，$v_a,v_b$ 为净流变化的标准差，$\lambda_a,\lambda_b$ 为事件率）：

\[
p_{\text{up}}(x,y)=\frac12-\frac{\arctan\!\Big(\sqrt{\tfrac{1+\rho}{1-\rho}}\;\dfrac{y/(\sqrt{\lambda_a}v_a)-x/(\sqrt{\lambda_b}v_b)}{y/(\sqrt{\lambda_a}v_a)+x/(\sqrt{\lambda_b}v_b)}\Big)}{2\arctan\!\Big(\sqrt{\tfrac{1+\rho}{1-\rho}}\Big)} .
\]

它可以直接当「先涨 / 先跌」的结构化基线特征。

### 7.5 代码：不动概率与先涨概率

```python
def depletion_density(t, x, lam, c):
    """Density of the depletion time of a birth-death queue (birth lam, death c), start x."""
    z = 2.0 * np.sqrt(lam * c) * t
    # ive(x, z) = I_x(z) * exp(-z), numerically stable
    return (x / t) * (c / lam) ** (x / 2.0) * special.ive(x, z) * np.exp(z - (lam + c) * t)


def depletion_survival(t, x, lam, c):
    """P(sigma > t | x). For lam > c the queue may never deplete, so this tends to 1 - (c/lam)^x."""
    val, _ = integrate.quad(depletion_density, t, np.inf, args=(x, lam, c), limit=400)
    return val


def prob_no_mid_change(t, q_ask, q_bid, lam_a, c_a, lam_b, c_b):
    """P(tau > t): neither best queue depleted within t (independent sides)."""
    return depletion_survival(t, q_ask, lam_a, c_a) * depletion_survival(t, q_bid, lam_b, c_b)


def up_probability(n_bid, p_ask):
    """Balanced flow: P(ask queue depletes first) = P(next mid move is up)."""
    def f(t):
        r = 2 - np.cos(t)
        return (r - np.sqrt(r * r - 1)) ** p_ask * np.sin(n_bid * t) * np.cos(t / 2) / np.sin(t / 2)
    val, _ = integrate.quad(f, 0, np.pi, limit=400)
    return val / np.pi


def simulate_depletion(x, lam, c, n_paths, rng, max_steps=200_000):
    """Monte Carlo check: embedded random walk (+1 w.p. lam/(lam+c)), holding times Exp(lam+c)."""
    p_up = lam / (lam + c)
    q = np.full(n_paths, x, dtype=np.int64)
    steps = np.zeros(n_paths, dtype=np.int64)
    alive = np.ones(n_paths, bool)
    block, done = 1000, 0
    while alive.any() and done < max_steps:
        idx = np.flatnonzero(alive)
        path = q[idx, None] + np.cumsum(np.where(rng.uniform(size=(len(idx), block)) < p_up, 1, -1), axis=1)
        hit = path <= 0
        first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, -1)
        ok = first > 0
        steps[idx[ok]] += first[ok]
        alive[idx[ok]] = False
        steps[idx[~ok]] += block
        q[idx[~ok]] = path[~ok, -1]
        done += block
    times = rng.gamma(shape=np.maximum(steps, 1), scale=1.0 / (lam + c))
    times[alive] = np.inf
    return times
```

## 8. 用什么时钟：物理时间、事件时间、业务时间

- **从属过程**：Clark (1973) 把价格写成 $P(t)=W(T(t))$，「信息时钟」$T(t)$ 用成交量作代理；Ané、Geman (2000) 发现用**累计成交笔数**当时钟更好，换钟后收益接近正态。
- **事件时间**：令 $T(t)=N(t)$。「什么时候」的随机性被吸收进时钟，方向类问题（下一次变动涨还是跌）在事件时间里更平稳、季节性更弱。
- **物理时间**：执行决策的期限 $\tau$ 是物理时间，所以最终需要 $N(t,t+\tau]$ 的分布，也就是本页讨论的到达模型。可以分解为

\[
\Pr(\text{先涨}\mid\tau)=\sum_{k\ge1}\Pr(\text{第 }k\text{ 个价格相关事件是首次变动且为涨})\cdot\Pr(\text{它在 }\tau\text{ 内到达}) .
\]

- **业务时间（推荐的折中）**：用补偿量当时钟，$\tilde t=\hat\Lambda(t)=\int_0^t\hat\lambda(u)\,du$，$\hat\lambda$ 取季节因子乘以 Hawkes 或 ACD 强度。由时间重标度定理，这个时钟下事件近似是单位速率泊松，hazard 的基线近似平坦，格点划分更均匀。代价是要把物理期限 $\tau$ 换算成 $\hat\Lambda(t+\tau)-\hat\Lambda(t)$，这需要强度的预测值。

我们的 τ 约定（见 [`modeling-guide.md`](modeling-guide.md)）仍用物理时间；业务时间只用于特征（例如「过去 1 个业务时间单位内的事件数」）和检验。

## 9. 模型检验：时间重标度

**定理**：若 $\lambda(t)$ 是 $N$ 关于自然滤子的条件强度，则

\[
e_i=\Lambda(t_{i-1},t_i)=\int_{t_{i-1}}^{t_i}\lambda(u)\,du\ \overset{\text{iid}}{\sim}\ \mathrm{Exp}(1).
\]

（Brown 等 2002；金融点过程里的用法见 Bowsher 2007。）多元模型对每一维分别重标度。要检验：

1. **分布**：KS 检验 $e_i$ 是否为 Exp(1)；QQ 图画 $e_{(i)}$ 对 $-\log\big(1-\tfrac{i-0.5}{n}\big)$。
2. **独立性**：Ljung–Box，$Q=n(n+2)\sum_{h=1}^{H}\hat\rho_h^2/(n-h)\sim\chi^2_H$。
3. **过度离散**：Engle–Russell 统计量 $\sqrt{n/8}\,(\hat\sigma_e^2-1)\overset{a}{\sim}N(0,1)$。分母的 8 来自 Exp(1) 下 $\mathrm{Var}\big[(e-1)^2\big]=8$。
4. **模型比较**：嵌套模型用似然比，非嵌套用 AIC/BIC。注意 $n\to1$ 在参数空间边界，常规渐近不一定适用。和执行更相关的做法是样本外比较预测对数得分：hazard 模型的分类对数损失和点过程的 $\sum\log\lambda-\int\lambda$ 可以直接对比。

**数据层面的注意事项**：

- 时间戳精度不够会出现大量 0 间隔。可以合并同一毫秒的事件、加随机抖动，或直接用离散时间模型（这也是离散 hazard 方案的天然优势）。
- 开盘、收盘、午休要截断或单独建模。
- 拟合 Hawkes 前先去掉日内季节性，否则 $n$ 会被高估。

在真实数据上的完整检验流程（H1–H9、判定阈值、结果表）见 [`arrival-tests.md`](arrival-tests.md)。

```python
def ljung_box(z, lags=20):
    z = np.asarray(z) - np.mean(z)
    n = len(z)
    acf = np.array([np.sum(z[h:] * z[:-h]) for h in range(1, lags + 1)]) / np.sum(z * z)
    Q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return Q, stats.chi2.sf(Q, lags)


def rescaling_report(name, e):
    """e: compensator increments / generalised residuals; should be iid Exp(1)."""
    ks = stats.kstest(e, "expon")
    Q, pq = ljung_box(e, 20)
    z_disp = np.sqrt(len(e) / 8.0) * (np.var(e) - 1.0)   # Engle-Russell excess dispersion
    print(f"  [{name}] var={e.var():.3f} | KS p={ks.pvalue:.3g} | LB(20) p={pq:.3g} | disp z={z_disp:.2f}")
    return dict(ks_p=ks.pvalue, lb_p=pq, disp_z=z_disp)
```

## 10. 怎么接进我们的 pipeline

### 10.1 统一视角：hazard 就是条件强度

把 τ 内的结局记为「先涨 / 先跌 / 不动」。用连续时间语言，就是两个 cause-specific 强度 $\lambda_\uparrow(t),\lambda_\downarrow(t)$ 的竞争风险：

\[
\Pr(\text{先涨且在 }\tau\text{ 内})=\int_0^\tau\lambda_\uparrow(s)\exp\!\Big(-\int_0^s\big(\lambda_\uparrow+\lambda_\downarrow\big)(u)\,du\Big)ds .
\]

若每个时间格 $[t_j,t_{j+1})$（宽 $\Delta_j$）内强度近似为常数，离散 hazard 为

\[
h_k(j)=\frac{\lambda_k}{\lambda_\uparrow+\lambda_\downarrow}\Big(1-e^{-(\lambda_\uparrow+\lambda_\downarrow)\Delta_j}\Big),\qquad k\in\{\uparrow,\downarrow\},
\]

\[
\Pr(\text{先涨于第 }j\text{ 格})=h_\uparrow(j)\prod_{i<j}\big(1-h_\uparrow(i)-h_\downarrow(i)\big).
\]

由此得到几条对应关系：

- 离散 hazard 用多项 logit 或 LightGBM 多分类建模，相当于对 $\lambda_k\Delta_j$ 学一个变换。
- hazard 里加「已等待时间 / 段号」的哑变量，就是非参数基线。Grammig–Maurer 发现的驼峰形 hazard 可以直接表达，比 WACD 的单调 hazard 灵活。
- 把 $\log\hat\psi_i$ 放进 hazard 当协变量，近似一个 AFT（ACD）结构。
- 把 Hawkes 强度放进 hazard 并在每格开始时更新，就得到「自激 + 状态依赖」的时变协变量强度。

### 10.2 作为特征

| 特征 | 计算 | 说明 |
|---|---|---|
| 季节因子 $\hat s(t)$ | 按半小时分箱求均值，样条平滑；分事件类型估计 | 所有强度的乘性项 |
| 多尺度指数衰减计数 $A^{(k)}_m(t)=\sum_{t^m_j<t}e^{-\beta_k(t-t^m_j)}$ | 每类事件 $m$（市价买 / 卖、一档撤单、一档新增、中间价上 / 下）取 2–4 个衰减率，如 $\beta=10,1,0.1\ \text{s}^{-1}$ | 「未加权的 Hawkes 状态」。让 hazard 自己学权重，就等价于多指数核 Hawkes，不必单独估计 Hawkes |
| 拟合后的 Hawkes 强度 $\hat\lambda_\uparrow,\hat\lambda_\downarrow$ | 滚动 1 小时窗口拟合 2 维指数核 Hawkes | $\hat n$ 可当「内生度」特征，但不稳定 |
| ACD 期望间隔 $\hat\psi_i$ | 中间价变动间隔、成交间隔分别季节调整后拟合 Log-ACD | 「下一次变动的预期等待」，取对数放入 hazard |
| 队列状态 | $q^b_1,q^a_1$（以 AES 为单位）、不平衡度 $I=(q^b-q^a)/(q^b+q^a)$、价差 | 第 7 节表明这是「不动多久」的头号决定因素 |
| 净流速度 | 每侧近期 $\hat\lambda-\hat c$（新增速率减撤单 + 市价单速率） | 决定耗尽时间是重尾还是指数截断 |
| 结构化基线概率 | $\phi(n,p)$ 或 $p_{\text{up}}(x,y)$；$\Pr(\tau>t\mid x,y)$ | 当先验特征或 offset |

多尺度衰减计数可以逐事件 $O(1)$ 更新，下面是在线版本：

```python
class DecayCounts:
    """Online multi-scale exponentially decayed event counts A_m^{(k)}(t), O(M*K) per event."""

    def __init__(self, n_types, betas):
        self.betas = np.asarray(betas, float)            # decay rates, 1/s
        self.state = np.zeros((n_types, len(self.betas)))
        self.t_last = None

    def _decay_to(self, t):
        if self.t_last is None:
            return self.state.copy()
        return self.state * np.exp(-self.betas * (t - self.t_last))

    def update(self, t, m):
        """Register an event of type m at time t (t non-decreasing)."""
        self.state = self._decay_to(t)
        self.state[m] += 1.0
        self.t_last = t

    def value(self, t):
        """Feature vector at time t (does not change the state); flatten for the model."""
        return self._decay_to(t).ravel()
```

### 10.3 作为「距下一次中间价变动时间」的基线

按复杂度从低到高：

1. **B0：分时段 NHPP**，$\lambda(t)=\hat\lambda_0/\hat s(t)$。只用来检验其他模型有没有增益。
2. **B1：队列状态依赖的泊松**（Cont–de Larrard / queue-reactive）。用 $(q^b,q^a)$ 的经验强度和生灭首达公式得到 $\Pr(\tau>t\mid q^b,q^a)$ 和先涨概率。可解释，不用历史信息。
3. **B2：2 维指数核 Hawkes，或 Log-ACD 加队列协变量**。把聚集效应加进来。
4. **主模型：离散时间竞争风险 hazard**。用 10.2 节的特征，并以 B1、B2 的输出作为 offset 或特征。

样本外用同一套评估比较：τ 内三分类的对数损失（见 [`ml-pipeline.md`](ml-pipeline.md) 第 3 节），以及时间重标度检验。

### 10.4 作为模拟器的订单流

强化学习和执行评估要用的模拟器（见 [`reinforcement-learning.md`](reinforcement-learning.md)）：

- **起点**：queue-reactive（Model I 或 III）。状态依赖的泊松，能复现队列分布，估计就是计数除以暴露时间。
- **加聚集**：强度上叠一个 Hawkes 激发项，$\lambda_m(t)=f_m(\text{state})+\sum_l\sum_j\phi_{ml}(t-t^l_j)$，或者直接用状态依赖 Hawkes。
- **季节性**：在业务时间里模拟，再用 $\hat s(t)$ 映射回物理时间。
- **验收**：模拟数据要复现第 2.3 节的事实（间隔 CV、自相关、中间价变动间隔分布、队列分布、成交概率），而且模拟器的强度在真实数据上能通过时间重标度检验。
- **注意**：纯外生的泊松或 Hawkes 流不会对我们的订单做出反应。至少要让强度依赖队列状态，这样我们的订单通过改变队列，间接影响别人的订单流。

## 11. 演示：在模拟数据上跑一遍

下面把前面的函数串起来，验证三件事：Hawkes 和 WACD 能恢复真值并通过检验，而同一份数据当泊松拟合会被检验拒绝；不做季节调整会得到假的高持久性；Cont–de Larrard 的公式和模拟一致。

```python
rng = np.random.default_rng(20261020)

print("(1) Hawkes, exponential kernel")
mu, alpha, beta, T = 0.5, 0.8, 1.2, 8000.0          # n = 0.667, mean rate 1.5
t = simulate_hawkes_exp(mu, alpha, beta, T, rng)
gaps = np.diff(t)
print(f"  N={len(t)}  CV of gaps={gaps.std() / gaps.mean():.2f} (Poisson: 1)")
p_hat, ll_h = fit_hawkes_exp(t, T)
se = np.sqrt(np.diag(np.linalg.inv(numerical_hessian(lambda p: -hawkes_exp_loglik(p, t, T)[0], p_hat))))
for nm, tv, ev, s in zip(["mu", "alpha", "beta"], [mu, alpha, beta], p_hat, se):
    print(f"  {nm:6s} true={tv:.3f}  mle={ev:.3f}  se={s:.3f}")
lam0 = len(t) / T
ll_p = len(t) * np.log(lam0) - lam0 * T
print(f"  AIC Hawkes={6 - 2 * ll_h:.0f}  AIC Poisson={2 - 2 * ll_p:.0f}")
rep_h = rescaling_report("Hawkes fit ", hawkes_exp_compensator_increments(p_hat, t))
rep_p = rescaling_report("Poisson fit", lam0 * gaps)
assert rep_h["ks_p"] > 0.01 and rep_p["ks_p"] < 1e-6

print("(2) Weibull-ACD(1,1) and diurnal adjustment")
omega, a, b, k = 0.1, 0.1, 0.8, 0.7
x = simulate_wacd(omega, a, b, k, 30000, rng)
p_w, ll_w = fit_wacd(x)
print("  mle (omega, a, b, k) =", np.round(p_w, 3))
rep_w = rescaling_report("WACD fit   ", wacd_integrated_hazard(p_w, x))
assert rep_w["ks_p"] > 0.01
rescaling_report("iid-Exp fit", x / x.mean())
day_len = 23400.0
s_true = lambda u: 0.6 + 1.6 * (1.0 - 4.0 * (u / day_len - 0.5) ** 2)   # long gaps at midday
x_raw, clock = np.empty_like(x), 0.0
for i in range(len(x)):
    x_raw[i] = x[i] * s_true(clock % day_len)
    clock += x_raw[i]
tod_prev = np.concatenate([[0.0], np.cumsum(x_raw)[:-1]]) % day_len
x_adj, _ = diurnal_adjust(x_raw, tod_prev, day_len)
p_raw, _ = fit_wacd(x_raw / x_raw.mean())
p_adj, _ = fit_wacd(x_adj)
print(f"  raw      : a={p_raw[1]:.3f} b={p_raw[2]:.3f}   (spurious persistence)")
print(f"  adjusted : a={p_adj[1]:.3f} b={p_adj[2]:.3f}")

print("(3) Cont-de Larrard")
for lam_, c_ in [(12.0, 13.0), (12.0, 12.0)]:
    sims = simulate_depletion(5, lam_, c_, 10000, rng)
    for tt in [5.0, 50.0]:
        print(f"  lam={lam_} c={c_}  P(sigma>{tt:>4}) exact={depletion_survival(tt, 5, lam_, c_):.4f}"
              f"  sim={np.mean(sims > tt):.4f}")
print(f"  P(no mid change in 1s | q_ask=3, q_bid=8, balanced, rate 12/s) = "
      f"{prob_no_mid_change(1.0, 3, 8, 12.0, 12.0, 12.0, 12.0):.3f}")
for n_, p_ in [(3, 3), (5, 2), (10, 1)]:
    print(f"  phi(bid={n_}, ask={p_}) = P(next move up) = {up_probability(n_, p_):.3f}")

print("(4) online decayed counts")
dc = DecayCounts(n_types=2, betas=[10.0, 1.0, 0.1])
for ti, m in [(0.00, 0), (0.05, 0), (0.30, 1), (1.20, 0)]:
    dc.update(ti, m)
print("  features at t=1.5:", np.round(dc.value(1.5), 3))
```

参考输出（numpy 2.x、scipy 1.1x，单核约 15 秒）：上面的 `assert` 通过；Hawkes 三个参数都在真值的两个标准误以内，泊松拟合的 KS 检验 $p\approx0$、过度离散 $z$ 在 70 以上；WACD 估计 $(\hat\omega,\hat\alpha,\hat\beta,\hat k)\approx(0.093,0.101,0.811,0.697)$ 并通过检验；不做季节调整时 $\hat\alpha+\hat\beta\approx0.98$，调整后约 0.91；$\phi(5,2)\approx0.756$，$\phi(10,1)\approx0.936$。完整版本（更大样本）的数字：Hawkes 估计 $\hat\mu=0.500\,(0.010)$、$\hat\alpha=0.792\,(0.016)$、$\hat\beta=1.202\,(0.026)$，KS $p=0.35$；同一数据当泊松拟合 $\mathrm{Var}(e)=2.87$、过度离散 $z=113$。

## 12. 在 i9 CPU 上的成本

| 计算 | 规模 | 量级 |
|---|---|---|
| 指数核 Hawkes 单元 MLE | 3 万事件，Nelder–Mead | 纯 Python 递推几秒；用 numba 编译递推可降到亚秒 |
| WACD MLE | 3 万间隔 | `lfilter` 向量化，约 1 秒 |
| 多元 Hawkes（8 维，指数核） | 一天全量事件 | 每个目标维 $M$ 个递推量，几分钟；或用 `tick` 库 |
| 衰减计数特征 | 每个事件 | $O(M K)$ 次乘加，微秒级 |
| $\Pr(\tau>t\mid x,y)$、$\phi(n,p)$ | 每次查询 | 数值积分约毫秒级；线上应在 $(x,y)$ 网格上**预先算表**再查表 |

全部不需要 GPU。12 代 i9（例如 12900K：8 个性能核 + 8 个能效核，24 线程）上，这些计算基本都是单线程的；按「股票 × 日」拆成独立任务，开 8–16 个进程并行（每个进程设 `OMP_NUM_THREADS=1`）最划算。现成工具：Python 的 `tick`（BSD-3，Hawkes 估计与模拟）、R 的 `ACDm`（GPL，各类 ACD）。Python 里没有维护中的主流 ACD 包，用上面的代码即可。

## 13. 调研路线：读什么、按什么顺序、每一步带走什么

按下面的顺序走，每一步都有「读」「带走」「在我们数据上验证」三项。验证项对应 [`arrival-tests.md`](arrival-tests.md) 的假设编号，所以读完就能动手。

| 步骤 | 读 | 带走什么 | 在我们数据上验证 |
|---|---|---|---|
| 1. 建立事实（半天） | Bouchaud、Bonart、Donier、Gould (2018) *Trades, Quotes and Prices* 中订单流与 tick 大小的章节；Hautsch (2012) *Econometrics of Financial High-Frequency Data* 中描述间隔特征的章节 | 间隔的过度离散、聚集、日内 U 形；大 tick 与小 tick 的区别 | H1、H2：CV、Fano 曲线、季节曲线 |
| 2. ACD（1 天） | Engle、Russell (1998)；Bauwens、Giot (2000)；Grammig、Maurer (2000)；Bauwens、Hautsch (2009) 的综述 | 第 4 节的模型和似然；季节调整的做法；误差分布怎么选 | H3、H4：WACD 持久性、残差检验、误差分布 |
| 3. Hawkes（1 天） | Bacry、Mastromatteo、Muzy (2015) 的综述；Filimonov、Sornette (2012, 2015)；Hardiman 等 (2013)；Lallouache、Challet (2016) | 第 5 节的似然和递推；分支比争议的来龙去脉；什么情况会把 $n$ 估高 | H5、H9：分支比、核形状、跨日稳定性 |
| 4. 排队模型（1 天） | Cont、Stoikov、Talreja (2010)；Cont、de Larrard (2013)；Huang、Lehalle、Rosenbaum (2015)；Gould、Bonart (2016) | 第 6–7 节的首达时间、$\phi(n,p)$、状态依赖强度 | H6、H7、H8 |
| 5. 跑检验（半天） | [`arrival-tests.md`](arrival-tests.md) 第 14 节 | 结果表和汇总 | 全部 |
| 6. 写结论（半天） | — | 按 [`arrival-tests.md`](arrival-tests.md) 第 13.3 节的决策表，决定 hazard 模型的到达类特征 | 把结论写回本页第 10.2 节 |

每读一篇，记下四件事：**模型的原假设是什么**（哪个分布、哪些独立性假设）、**用什么数据**（市场、tick 大小、时间戳精度、样本期）、**怎么检验**（统计量、是否做了季节调整）、**结论在什么条件下成立**。四项里任何一项和我们的数据不一致，那篇的结论就只能当参考。

## 14. 附录：本页公式的 LaTeX 源码

页面上每个公式块右上角都有「复制 LaTeX」按钮。下面是本页主要公式的完整源码，可以整段粘贴到 Overleaf 编译（需要 `amsmath`、`amssymb`）；检验统计量的源码在 [`arrival-tests.md`](arrival-tests.md) 第 16 节。

```latex
\documentclass{article}
\usepackage{amsmath,amssymb}
\begin{document}
\section*{Event-arrival models}

\paragraph{Poisson.}
\[
\Pr\{N(t+\Delta)-N(t)=k\}=e^{-\lambda\Delta}\frac{(\lambda\Delta)^k}{k!},\qquad
X_i\overset{\text{iid}}{\sim}\mathrm{Exp}(\lambda),\qquad \mathrm{CV}=1,\qquad \frac{\mathrm{Var}\,N}{\mathbb{E}N}=1 .
\]

\paragraph{NHPP and Cox.}
\[
\ell=\sum_i\log\lambda(t_i)-\int_0^T\lambda(u)\,du,\qquad
\mathrm{Var}\,N(0,t]=\mathbb{E}\,\Lambda(t)+\mathrm{Var}\,\Lambda(t)\ \ge\ \mathbb{E}\,N(0,t].
\]

\paragraph{ACD (Engle--Russell 1998).}
\[
x_i=\psi_i\varepsilon_i,\quad \mathbb{E}\varepsilon_i=1,\quad
\psi_i=\omega+\alpha x_{i-1}+\beta\psi_{i-1},\quad
\lambda(t\mid\mathcal{F}_t)=\frac{1}{\psi_{N(t)+1}}\,h_\varepsilon\!\Big(\frac{t-t_{N(t)}}{\psi_{N(t)+1}}\Big),
\]
\[
\ell_{\text{EACD}}=-\sum_i\Big[\log\psi_i+\frac{x_i}{\psi_i}\Big],\qquad
\ell_{\text{WACD}}=\sum_i\Big[\log k-\log x_i+k\log\frac{g x_i}{\psi_i}-\Big(\frac{g x_i}{\psi_i}\Big)^k\Big],\quad g=\Gamma(1+1/k),
\]
\[
\log\psi_i=\omega+\beta\log\psi_{i-1}+\alpha\,g(\varepsilon_{i-1})+\gamma^\top z_{i-1}\quad\text{(Log-ACD)}.
\]

\paragraph{Hawkes.}
\[
\lambda(t)=\mu+\sum_{t_i<t}\phi(t-t_i),\qquad \phi(t)=\alpha e^{-\beta t},\qquad n=\|\phi\|_1=\frac{\alpha}{\beta}<1,\qquad \bar\lambda=\frac{\mu}{1-n},
\]
\[
A_i=e^{-\beta(t_i-t_{i-1})}(1+A_{i-1}),\qquad
\ell=\sum_{i=1}^N\log(\mu+\alpha A_i)-\mu T-\frac{\alpha}{\beta}\sum_{i=1}^N\big(1-e^{-\beta(T-t_i)}\big),
\]
\[
\Lambda(t_{i-1},t_i)=\mu\Delta_i+\frac{\alpha}{\beta}(1+A_{i-1})\big(1-e^{-\beta\Delta_i}\big),\qquad
\lambda_m(t)=\mu_m+\sum_{l=1}^M\sum_{t^l_j<t}\phi_{ml}(t-t^l_j).
\]

\paragraph{Queue-reactive.}
\[
\hat\Lambda(n)=\big(\overline{\Delta t\mid q=n}\big)^{-1},\qquad
\hat\lambda^{L}(n)=\hat\Lambda(n)\cdot\frac{\#\{\text{add},\,q=n\}}{\#\{q=n\}} .
\]

\paragraph{Cont--de Larrard.}
\[
\mathbb{E}\big[e^{-s\sigma}\mid q_0=x\big]=\Big(\frac{(\lambda+c+s)-\sqrt{(\lambda+c+s)^2-4\lambda c}}{2\lambda}\Big)^x,\qquad
f_\sigma(t\mid x)=\frac{x}{t}\Big(\frac{c}{\lambda}\Big)^{x/2}I_x\big(2\sqrt{\lambda c}\,t\big)e^{-(\lambda+c)t},
\]
\[
\Pr(\tau>t\mid x,y)=\Pr(\sigma^a>t\mid x)\Pr(\sigma^b>t\mid y),\qquad
\lambda=c:\ \Pr(\tau>t\mid x,y)\sim\frac{xy}{\pi\lambda}\,t^{-1},
\]
\[
\lambda<c:\ \Pr(\sigma>t)\propto t^{-3/2}e^{-(\sqrt c-\sqrt\lambda)^2t},\qquad \mathbb{E}\sigma=\frac{x}{c-\lambda},
\]
\[
\phi(n,p)=\frac1\pi\int_0^\pi\Big(2-\cos t-\sqrt{(2-\cos t)^2-1}\Big)^p\frac{\sin(nt)\cos(t/2)}{\sin(t/2)}\,dt .
\]

\paragraph{Time rescaling.}
\[
e_i=\int_{t_{i-1}}^{t_i}\lambda(u)\,du\overset{\text{iid}}{\sim}\mathrm{Exp}(1),\qquad
Q=n(n+2)\sum_{h=1}^H\frac{\hat\rho_h^2}{n-h},\qquad z=\sqrt{n/8}\,(\hat\sigma_e^2-1).
\]

\paragraph{Discrete-time competing-risk hazard.}
\[
h_k(j)=\frac{\lambda_k}{\lambda_\uparrow+\lambda_\downarrow}\Big(1-e^{-(\lambda_\uparrow+\lambda_\downarrow)\Delta_j}\Big),\qquad
\Pr(\text{up in bin } j)=h_\uparrow(j)\prod_{i<j}\big(1-h_\uparrow(i)-h_\downarrow(i)\big).
\]
\end{document}
```

## 15. 参考文献

- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1):83–90.
- Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. *Econometrica* 41(1):135–155.
- Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Ann. Inst. Statist. Math.* 31:145–155.
- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Trans. Inf. Theory* 27(1):23–31.
- Engle, R. F., Russell, J. R. (1998). Autoregressive conditional duration: a new model for irregularly spaced transaction data. *Econometrica* 66(5):1127–1162.
- Ghysels, E., Jasiak, J. (1998). GARCH for irregularly spaced financial data: the ACD-GARCH model. *Studies in Nonlinear Dynamics & Econometrics* 2(4).
- Bauwens, L., Giot, P. (2000). The logarithmic ACD model. *Annales d'Économie et de Statistique* 60:117–149.
- Grammig, J., Maurer, K.-O. (2000). Non-monotonic hazard functions and the autoregressive conditional duration model. *Econometrics Journal* 3(1):16–38.
- Ané, T., Geman, H. (2000). Order flow, transaction clock, and normality of asset returns. *J. Finance* 55(5):2259–2284.
- Plerou, V. et al. (2000). Economic fluctuations and anomalous diffusion. *Phys. Rev. E* 62:R3023.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., Frank, L. M. (2002). The time-rescaling theorem and its application to neural spike train data analysis. *Neural Computation* 14(2):325–346.
- Smith, E., Farmer, J. D., Gillemot, L., Krishnamurthy, S. (2003). Statistical theory of the continuous double auction. *Quantitative Finance* 3(6):481–514.
- Bowsher, C. G. (2007). Modelling security market events in continuous time: intensity based, multivariate point process models. *J. Econometrics* 141(2):876–912.
- Bauwens, L., Hautsch, N. (2009). Modelling financial high frequency data using point processes. In *Handbook of Financial Time Series*, Springer.
- Cont, R., Stoikov, S., Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research* 58(3):549–563.
- Bacry, E., Dayri, K., Muzy, J.-F. (2012). Non-parametric kernel estimation for symmetric Hawkes processes. *Eur. Phys. J. B* 85:157.
- Filimonov, V., Sornette, D. (2012). Quantifying reflexivity in financial markets. *Phys. Rev. E* 85:056108.
- Cont, R., de Larrard, A. (2012). Order book dynamics in liquid markets: limit theorems and diffusion approximations. arXiv:1202.6412.
- Hautsch, N. (2012). *Econometrics of Financial High-Frequency Data*. Springer.
- Cont, R., de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Math.* 4(1):1–25. arXiv:1104.4596.
- Bacry, E., Delattre, S., Hoffmann, M., Muzy, J.-F. (2013). Modelling microstructure noise with mutually exciting point processes. *Quantitative Finance* 13(1):65–77.
- Hardiman, S. J., Bercot, N., Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. *Eur. Phys. J. B* 86:442.
- Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1). arXiv:1502.04592.
- Huang, W., Lehalle, C.-A., Rosenbaum, M. (2015). Simulating and analyzing order book data: the queue-reactive model. *JASA* 110(509):107–122.
- Filimonov, V., Sornette, D. (2015). Apparent criticality and calibration issues in the Hawkes self-excited point process model. *Quantitative Finance* 15(8):1293–1314.
- Bacry, E., Jaisson, T., Muzy, J.-F. (2016). Estimation of slowly decreasing Hawkes kernels: application to high-frequency order book dynamics. *Quantitative Finance* 16(8):1179–1201.
- Lallouache, M., Challet, D. (2016). The limits of statistical significance of Hawkes processes fitted to financial data. *Quantitative Finance* 16(1):1–11.
- Gould, M. D., Bonart, J. (2016). Queue imbalance as a one-tick-ahead price predictor in a limit order book. *Market Microstructure and Liquidity* 2(2):1650006.
- Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. (2018). *Trades, Quotes and Prices: Financial Markets Under the Microscope*. Cambridge University Press.
- Morariu-Patrichi, M., Pakkanen, M. S. (2022). State-dependent Hawkes processes and their application to limit order book modelling. *Quantitative Finance* 22(3):563–583.
