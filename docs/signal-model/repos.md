# 开源仓库清单

2026-10-20 打开 GitHub 页面核对的。star 数会变；「最后提交」是默认分支最近一次提交的日期。许可证一栏要特别留意：**GPL、AGPL、LGPL 和 CC BY-NC-ND 的库，以及没有 LICENSE 文件的仓库（默认保留所有权利），在涉及分发、对外服务或商用前都要先过法务**。线上关键路径优先用 MIT、BSD、Apache。

## 1. 数据、订单簿重建、回测

| 仓库 | 用途 | 许可证 | Star | 最后提交 | 评价 |
|---|---|---|---|---|---|
| [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) | tick 级回测：L2 概率队列模型、L3 队列、实测延迟；Rust + Python（numba） | MIT | 4.9k | 2025-12 | **强烈推荐**，做第一层模拟器。主要面向加密货币数据格式，要自己写转换器；盘口不对自身订单反应 |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | Rust 原生交易引擎，纳秒级回测 | LGPL-3.0 | 30.4k | 2026-10 | 功能全，注意许可证 |
| [vgreg/MeatPy](https://github.com/vgreg/MeatPy) | ITCH 2.0–5.0、IEX DEEP 解析和订单簿重建 | BSD-3 | 36 | 2026-05 | 重建参考 |
| [bmoscon/orderbook](https://github.com/bmoscon/orderbook) | C 实现的 Python L2 / L3 订单簿数据结构 | GPL-3.0 | 320 | 2026-10 | 注意许可证 |
| [databento/databento-python](https://github.com/databento/databento-python) | 行情客户端，MBO / MBP / BBO | Apache-2.0 | 306 | 2026-09 | — |

## 2. 金融机器学习方法（标签、样本权重、purged CV）

| 仓库 | 用途 | 许可证 | Star | 最后提交 | 评价 |
|---|---|---|---|---|---|
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | AFML 方法库 | **专有协议**，现在付费订阅 | 4.9k | 2021-12 | **不建议引入**，包括它的 fork |
| [baobach/mlfinpy](https://github.com/baobach/mlfinpy) | 从零重写的 AFML 实现，含 sequential bootstrap | MIT | 89 | 2026-10 | 可参考 |
| [RiskLabAI/RiskLabAI.py](https://github.com/RiskLabAI/RiskLabAI.py) | 三重障碍、meta-labeling、唯一性权重、purged / combinatorial CV | BSD-3 | 8 | 2026-10 | 可参考 |
| [sam31415/timeseriescv](https://github.com/sam31415/timeseriescv) | `PurgedWalkForwardCV`、`CombPurgedKFoldCV`，按标签结束时间做 purge | MIT | 290 | 2022-02 | 不再维护，复制到自己的代码库再用 |
| [skfolio/skfolio](https://github.com/skfolio/skfolio) | `CombinatorialPurgedCV`、`WalkForward` | BSD-3 | 2.5k | 2026-10 | **默认不做 purge 和 embargo**，要自己设 |
| [esvhd/pypbo](https://github.com/esvhd/pypbo) | PBO、Deflated Sharpe | **AGPL-3.0** | 144 | 2026-07 | 注意许可证 |
| [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) | 书籍配套代码，GBDT + Optuna 示例 | MIT（代码） | 21.5k | 2026-10 | 学习用 |

唯一性权重和 purged CV 的逻辑只有几十行，建议照着 López de Prado（2018）自己实现。

## 3. 建模、校准、解释、监控、部署

| 仓库 | 用途 | 许可证 | Star | 最后提交 | 评价 |
|---|---|---|---|---|---|
| [microsoft/LightGBM](https://github.com/microsoft/LightGBM) | GBDT 主力 | MIT | 18.9k | 2026-10 | — |
| [dmlc/xgboost](https://github.com/dmlc/xgboost) | GBDT，内置 `survival:aft`、`survival:cox` | Apache-2.0 | 28.9k | 2026-10 | 生存模型用 |
| [catboost/catboost](https://github.com/catboost/catboost) | 对称树，MultiClass、MultiQuantile | Apache-2.0 | 9.2k | 2026-10 | 单行推理延迟友好 |
| [stanfordmlgroup/ngboost](https://github.com/stanfordmlgroup/ngboost) | 概率 boosting，`NGBSurvival` | Apache-2.0 | 1.9k | 2026-09 | 研究用 |
| [optuna/optuna](https://github.com/optuna/optuna) | 超参搜索 | MIT | 14.9k | 2026-10 | — |
| [sebp/scikit-survival](https://github.com/sebp/scikit-survival) | 梯度提升生存、RSF、C-index、IBS、竞争风险 CIF | **GPL-3.0** | 1.3k | 2026-10 | 研究对照 |
| [CamDavidsonPilon/lifelines](https://github.com/CamDavidsonPilon/lifelines) | 经典生存模型 | MIT | 2.6k | 2026-03 | 快速基线 |
| [EFS-OpenSource/calibration-framework](https://github.com/EFS-OpenSource/calibration-framework) | ECE、可靠性图、temperature、isotonic | Apache-2.0 | 380 | 2026-04 | — |
| [ip200/venn-abers](https://github.com/ip200/venn-abers) | Venn-Abers 校准 | MIT | 208 | 2026-09 | — |
| [shap/shap](https://github.com/shap/shap) | SHAP / TreeSHAP | MIT | 25.8k | 2026-10 | — |
| [evidentlyai/evidently](https://github.com/evidentlyai/evidently) | 漂移报告（PSI 等） | Apache-2.0 | 8.0k | 2026-09 | 监控 |
| [siboehm/lleaves](https://github.com/siboehm/lleaves) | LLVM 编译 LightGBM 推理 | MIT | 484 | 2025-10 | 低延迟 |
| [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | 树模型转 ONNX | Apache-2.0 | 1.2k | 2026-10 | — |
| [dmlc/tl2cgen](https://github.com/dmlc/tl2cgen) | 树模型转 C | Apache-2.0 | 55 | — | **2026-10-19 已归档**，新项目不要依赖 |
| [BayesWitnesses/m2cgen](https://github.com/BayesWitnesses/m2cgen) | 模型转多语言代码 | MIT | 3.0k | 2022-10 | 停止维护 |

## 4. 深度学习（LOB）

| 仓库 | 对应论文 | 框架 | 许可证 | Star | 最后提交 | 评价 |
|---|---|---|---|---|---|---|
| [LeonardoBerti00/TLOB](https://github.com/LeonardoBerti00/TLOB) | TLOB / MLPLOB，附 DeepLOB、BiN-CTABL | PyTorch + Lightning + Hydra | MIT | 182 | 2026-02 | **起步参考首选**：最活跃，支持 LOBSTER 格式，最容易改成自有数据 |
| [zcakhaa/DeepLOB-…](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) | DeepLOB | TF / PyTorch notebook | **无 LICENSE** | 618 | 2021-07 | 只读，自己重写 |
| [zcakhaa/Multi-Horizon-Forecasting-for-Limit-Order-Books](https://github.com/zcakhaa/Multi-Horizon-Forecasting-for-Limit-Order-Books) | DeepLOB-Seq2Seq / Attention | notebook | MIT | 216 | 2023-03 | 停更 |
| [jwallbridge/translob](https://github.com/jwallbridge/translob) | TransLOB | Keras / TF | **无 LICENSE** | 125 | 2020-08 | 只读 |
| [viebboy/TABL](https://github.com/viebboy/TABL) | TABL | Keras 2.1 / Python 2.7 | **无 LICENSE** | 36 | 2019-04 | 已过时 |
| [matteoprata/LOBCAST](https://github.com/matteoprata/LOBCAST) | LOBCAST 基准 | PyTorch | **无 LICENSE** | 125 | 2024-05 | 思路可以照着重写 |
| [FinancialComputingUCL/LOBFrame](https://github.com/FinancialComputingUCL/LOBFrame) | Briola 等 2025 + HLOB | PyTorch Lightning | **CC BY-NC-ND 4.0（禁止商用和演绎）** | 259 | 2024-05 | 思路可以照着重写，不要拷代码 |

## 5. 强化学习、模拟器

### 模拟器和环境

| 仓库 | 用途 | 许可证 | Star | 最后提交 | 评价 |
|---|---|---|---|---|---|
| [SaadSouilmi/Queue-Reactive](https://github.com/SaadSouilmi/Queue-Reactive) | 扩展 queue-reactive 模型的 C++20 模拟引擎 + Python 校准（Noble 等 2026） | MIT | 14 | 2026-03 | **推荐做第二层交互式模拟器的参考**，CPU 即可 |
| [peernagy/lob_bench](https://github.com/peernagy/lob_bench) | 生成式 LOB 评测：分布距离、条件统计、冲击响应曲线 | MIT | 46 | 2026-01 | **推荐**：借它的指标验证自己的模拟器 |
| [JJJerome/mbt_gym](https://github.com/JJJerome/mbt_gym) | 基于 SDE 的 LOB 交易 Gym 环境（做市、执行），向量化 | BSD-3 | 181 | 2024-01 | 检查 RL 管线能否复现解析最优解 |
| [microsoft/qlib](https://github.com/microsoft/qlib) | 量化平台；`examples/rl_order_execution` 有 PPO（Lin 和 Beling 2020）、OPDS（Fang 等 2021）的执行 RL | MIT | ~49.5k | 2026-10 | **开源执行 RL 的最佳工程参考**（偏排程层） |
| [X-DataInitiative/tick](https://github.com/X-DataInitiative/tick) | Hawkes 过程模拟和估计 | BSD-3 | 558 | 2026-05 | Hawkes 订单流成分 |
| [jpmorganchase/abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public) | ABIDES 多主体模拟 + ABIDES-Gym | BSD-3 | 175 | 2023-12 | **2025-06 已归档**；压力测试可用，要自己修依赖，慢 |
| [abides-sim/abides](https://github.com/abides-sim/abides) | ABIDES 原版 | BSD-3 | 586 | 2020-11 | 停更 |
| [KangOxford/AlphaTrade](https://github.com/KangOxford/AlphaTrade) | JAX-LOB | 未核实 | 146 | 2023-11 | 需要 GPU，不适合 |
| [microsoft/MarS](https://github.com/microsoft/MarS) | 订单级生成式市场模拟 | MIT | 1.9k | 2026-09 | 需要 GPU，研究跟踪 |
| [LeonardoBerti00/DeepMarket](https://github.com/LeonardoBerti00/DeepMarket) | TRADES（扩散）+ CGAN LOB 生成 | MIT | 113 | 2026-01 | 需要 GPU，研究跟踪 |
| [peernagy/LOBS5](https://github.com/peernagy/LOBS5) | LOBS5 自回归 S5 生成模型 | 未核实 | 32 | 2024-04 | 需要 GPU |

### RL、离线 RL、bandit 库

| 仓库 | 用途 | 许可证 | Star | 最后提交 | 评价 |
|---|---|---|---|---|---|
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | PPO、DQN、SAC 等；SB3-Contrib 有 **Maskable PPO** | MIT | 13.9k | 2026-10 | **推荐**：CPU 跑小 MLP 足够，直接支持动作屏蔽 |
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | 单文件 RL 实现 | MIT | 10.6k | 2026-04 | 要改算法细节时用 |
| [takuseno/d3rlpy](https://github.com/takuseno/d3rlpy) | 离线 RL：BC、CQL、BCQ（离散和连续）、IQL、TD3+BC | MIT | 1.7k | 2025-09 | **推荐**：DiscreteBC / DiscreteCQL / DiscreteBCQ |
| [st-tech/zr-obp](https://github.com/st-tech/zr-obp) | Open Bandit Pipeline：bandit 和离线策略评估（IPW、DR 等） | Apache-2.0 | 714 | 2022-11 | 停更但估计器可靠，固定依赖版本后用 |
| [hakuhodo-technologies/scope-rl](https://github.com/hakuhodo-technologies/scope-rl) | 离线 RL + OPE + 策略选择，含 CVaR 评估 | Apache-2.0 | 143 | 2023-12 | 低活跃 |
| [VowpalWabbit/vowpal_wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) | 工业级 contextual bandit | 未核实 | 8.7k | 2026-09 | 可选 |

### 参考价值有限

| 仓库 | 原因 |
|---|---|
| [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)（MIT） | 没有 LOB 和撮合，以日频、分钟级组合问题为主 |
| [AI4Finance-Foundation/FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)（MIT） | 没有订单簿建模 |
| [TradeMaster-NTU/TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster)（Apache-2.0） | 有订单执行任务，学术原型，低活跃 |
| [AminHP/gym-anytrading](https://github.com/AminHP/gym-anytrading)（MIT） | 没有订单簿，只有买卖持仓信号；停更 |

### 论文官方代码

| 论文 | 代码 |
|---|---|
| Ning、Lin、Jaimungal（DDQN 执行） | 没找到作者官方仓库 |
| Lin、Beling 2020（PPO 执行） | 没有官方代码；qlib 有复现配置 |
| Fang 等 2021（OPDS） | qlib `examples/rl_order_execution` |
| Kolm 等 2023（Deep OFI） | 没找到 |
| Fabre、Ragel 2023 | 没找到 |
| LOBS5 / LOB-Bench / TRADES / MarS / Queue-Reactive 2026 | 见上表 |
