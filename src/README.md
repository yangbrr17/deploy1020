# 代码包说明

两个独立的 Python 包，都只依赖 numpy、scipy、pandas；`execrl` 另外用 LightGBM、scikit-learn，IQL 和 PPO 用 PyTorch（CPU 版即可）。全部在没有 GPU 的机器上开发和测试。

| 包 | 做什么 | 对应文档 |
|---|---|---|
| [`arrivals/`](arrivals/__init__.py) | 事件到达分布的假设检验 H1–H9：数据清洗、检验统计量、ACD、Hawkes、Cont–de Larrard，输出一张结果表 | [`arrival-tests.md`](../docs/signal-model/arrival-tests.md)、[`event-arrivals.md`](../docs/signal-model/event-arrivals.md) |
| [`execrl/`](execrl/__init__.py) | 执行切片的决策：宏动作 bandit + DR、约束 FQI、FQE、IQL、PPO，统一的配置、数据、策略接口和训练函数 | [`reinforcement-learning.md`](../docs/signal-model/reinforcement-learning.md) 第 13 节 |

## 安装和自检

在仓库根目录：

```bash
pip install numpy scipy pandas scikit-learn lightgbm pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu      # only for IQL / PPO
PYTHONPATH=src python -m pytest -q src/tests                            # 16 tests, about 30 s
```

## arrivals：文件一览

| 文件 | 内容 |
|---|---|
| [`arrivals/schema.py`](arrivals/schema.py) | 输入表结构（trades、quotes、可选 level2）、校验、5 条清洗规则、4 条事件流、L1 流量近似 |
| [`arrivals/stats.py`](arrivals/stats.py) | 色散检验、KS（bootstrap）、Anderson–Darling、Ljung–Box、计数离散指数、似然比（含边界修正）、Vuong、Hill、Cochran Q、校准检验、BH |
| [`arrivals/distributions.py`](arrivals/distributions.py) | 指数 / Weibull / 伽马 / 对数正态 / 广义伽马 / Burr XII 的 MLE 和比较 |
| [`arrivals/models.py`](arrivals/models.py) | 日内季节曲线、业务时间、Weibull-ACD、指数核 Hawkes、时间重标度残差 |
| [`arrivals/queue.py`](arrivals/queue.py) | Cont–de Larrard 耗尽时间、生存函数、方向概率 φ、PIT、队列状态依赖检验 |
| [`arrivals/synth.py`](arrivals/synth.py) | 同一表结构的合成数据（dry run 和测试用） |
| [`arrivals/report.py`](arrivals/report.py) | 按顺序跑 H1–H9、判定规则、命令行 |

```bash
PYTHONPATH=src python -m arrivals.report --synthetic --days 3 --out synthetic.csv
PYTHONPATH=src python -m arrivals.report --trades trades.csv --quotes quotes.csv \
    --tz America/New_York --open 09:30 --close 16:00 --resolution-ns 1000000 --out results.csv
```

## execrl：文件一览

| 文件 | 内容 |
|---|---|
| [`execrl/config.py`](execrl/config.py) | 全部超参数（dataclass），`ExperimentConfig` 存取 JSON |
| [`execrl/compute.py`](execrl/compute.py) | CPU 检测（12 代 i9 的性能核 / 能效核）、线程设置、LightGBM 线程基准 |
| [`execrl/actions.py`](execrl/actions.py) | 原子动作、宏动作、进度带屏蔽 |
| [`execrl/reward.py`](execrl/reward.py) | 切片成本、逐步奖励、势函数塑形、宏动作奖励 |
| [`execrl/data.py`](execrl/data.py) | `BanditLog`、`Transitions`、按母单 / 按天切分、交叉拟合折 |
| [`execrl/simulator.py`](execrl/simulator.py) | 向量化玩具环境 `SliceEnv` 和日志数据生成 |
| [`execrl/logging_policy.py`](execrl/logging_policy.py) | 带倾向得分的随机化日志策略 |
| [`execrl/baselines.py`](execrl/baselines.py) | TWAP、Almgren-Chriss、规则策略、一步成本规则 |
| [`execrl/bandit/reward_model.py`](execrl/bandit/reward_model.py) | 奖励模型与交叉拟合 |
| [`execrl/bandit/ope.py`](execrl/bandit/ope.py) | IPW / SNIPW / DR、聚类标准误、上线门槛 |
| [`execrl/bandit/policy_tree.py`](execrl/bandit/policy_tree.py) | 深度 1–2 的精确策略树 |
| [`execrl/bandit/thompson.py`](execrl/bandit/thompson.py) | 线性 Thompson 采样 |
| [`execrl/offline/behaviour.py`](execrl/offline/behaviour.py) | 行为策略模型 |
| [`execrl/offline/fqi.py`](execrl/offline/fqi.py) | 带支撑约束的拟合 Q 迭代 |
| [`execrl/offline/fqe.py`](execrl/offline/fqe.py) | 拟合 Q 评估 |
| [`execrl/offline/iql.py`](execrl/offline/iql.py) | 离散 IQL |
| [`execrl/online/ppo.py`](execrl/online/ppo.py) | 带屏蔽的 PPO 和蒸馏 |
| [`execrl/evaluate.py`](execrl/evaluate.py) | 样本量、CUPED、聚类 bootstrap、模拟器真值 |
| [`execrl/train.py`](execrl/train.py) | `train_bandit`、`train_fqi`、`train_iql`、`train_ppo_run` 和命令行 |

```bash
PYTHONPATH=src python -m execrl.compute --bench
PYTHONPATH=src python -m execrl.train bandit --out runs/bandit
PYTHONPATH=src python -m execrl.train all --out runs/all --threads 8
```

## 测试

| 文件 | 测什么 |
|---|---|
| [`tests/test_arrivals.py`](tests/test_arrivals.py) | 零假设下不拒绝、明显违背时拒绝；Hawkes 能恢复参数；精确 Cont–de Larrard 数据上 PIT 均匀、φ 校准；报告端到端 |
| [`tests/test_execrl.py`](tests/test_execrl.py) | 屏蔽规则；逐步奖励之和等于负的切片成本（含塑形）；倾向得分一致；DR 无偏；策略树找到正确切分；各训练函数冒烟；配置读写 |

## 约定

- 单位：价格是货币单位，数量是手，成本和奖励是基点（奖励 = −成本），时间是秒（`arrivals`）或决策步（`execrl`）。
- 每个文件开头的 docstring 写契约：输入输出的形状、单位、对应的文档小节。改函数时同步改 docstring。
- 随机数一律通过 `numpy.random.default_rng(seed)` 传入，不用全局随机状态。
- 新功能先在合成数据上加测试，再接真实数据。
- 仓库是公开的：代码、注释、测试数据里不出现任何机构、产品、系统名称，也不放真实数据。
