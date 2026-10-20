# 给 Copilot（及其他代码助手）的说明

本仓库是**公开的**纯技术笔记：订单簿短期信号建模、事件到达分布检验、执行决策的强化学习，以及一个 Agent 平台的部署方案。文档是中文，代码和代码注释是英文。

## 硬性规则

1. **不写任何业务内容**：不出现机构名、人名、部门、客户、内部系统名、库表名、字段口径、真实数据。例子一律用合成数据（`src/arrivals/synth.py`、`src/execrl/simulator.py`）。提交前在改动中搜一遍专有名词。
2. 不引入需要 GPU 的依赖。目标机器是 12 代 Intel i9（例如 8 个性能核 + 8 个能效核，24 线程），没有 GPU。
3. 改了算法就要同步改：函数的 docstring、对应文档小节、测试。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/signal-model/*.md` | 建模文档。入口是 `modeling-guide.md`；到达分布在 `event-arrivals.md` 和 `arrival-tests.md`；强化学习在 `reinforcement-learning.md`（第 13 节是代码结构） |
| `docs/signal-model/dl/*.md` | 每个深度学习模型一页 |
| `src/arrivals/` | 到达分布检验包（H1–H9），入口 `report.py` |
| `src/execrl/` | 执行决策包（bandit、离线 RL、PPO），入口 `train.py` |
| `src/tests/` | pytest 测试 |
| `site/build.py` | 把 Markdown 和 `src/**/*.py` 生成静态站点；GitHub Actions 推送后自动发布到 Pages |

## 常用命令（仓库根目录）

```bash
pip install numpy scipy pandas scikit-learn lightgbm pytest markdown==3.7 pymdown-extensions==10.12
pip install torch --index-url https://download.pytorch.org/whl/cpu
PYTHONPATH=src python -m pytest -q src/tests                 # all tests, about 30 s
PYTHONPATH=src python -m arrivals.report --synthetic --days 3 --out synthetic.csv
PYTHONPATH=src python -m execrl.train all --out runs/all
python site/build.py                                         # builds _site/
```

## 写法约定

**文档（中文）**
- 短句、陈述句，一段一个结论；先结论后细节。
- 公式用 LaTeX：行内 `$...$`，独立公式 `$$` 单独成行。站点用 KaTeX 渲染，只用 KaTeX 支持的命令；表格单元格里的绝对值写 `\lvert x\rvert`，不要写 `|x|`。
- 每个符号第一次出现时定义；数字要么来自代码实测（写明条件），要么标「估计，未实测」。
- 链接到代码用相对路径，例如 `[report.py](../../src/arrivals/report.py)`，站点会改写成源码页。
- 文档里语言标为 `python` 的代码块按页拼接后要能直接运行（只用合成数据）；不能运行的片段标为 `text`。

**代码（英文注释）**
- 每个模块开头的 docstring 写契约：输入输出形状、单位、对应文档小节、已知局限。
- 单位：价格是货币，数量是手，成本和奖励是基点（奖励 = −成本）。
- 随机数通过 `np.random.default_rng(seed)` 传入，不用全局状态。
- `execrl` 的策略签名统一为 `policy(x, mask, rng) -> (actions, propensities)`；数据用 `BanditLog` / `Transitions`；超参数只放在 `config.py`。
- `arrivals` 的检验函数返回 dict（`stat`、`p_value`、`effect`、`effect_name`、`n`），在 `report.run_day` 里用 `_row(...)` 加到结果表。

## 待办（按优先级）

**到达分布检验（`src/arrivals/`）**
1. 用真实数据跑 `arrival-tests.md` 第 14 节的步骤，把结论写回 `event-arrivals.md` 第 10.2 节。
2. 支持 L2 逐笔表：在 `schema.py` 加计数函数，替换 `queue.estimate_rates` 和 `queue.queue_reactive_lr` 里的 L1 近似。
3. `models._hawkes_A` 加 `numba.njit`；新增双指数核和幂律核 Hawkes，H5 同时报告各核的 AIC。
4. Log-ACD（带协变量）和 Burr-ACD。
5. 按「股票 × 日」用进程池并行（`OMP_NUM_THREADS=1`）。

**执行决策（`src/execrl/`）**
1. 用 queue-reactive + Hawkes 订单流 + 传播子冲击写一个真实感的模拟器，接口与 `SliceEnv` 相同（`reset`、`step`、`info["fill_v"]`、`info["fill_vp"]`），替换玩具环境。
2. 真实决策日志的适配：`BanditLog.from_frame`、`split_by_time`；特征按 `reinforcement-learning.md` 第 2.2 节。
3. 离散 CQL（`offline/cql.py`，文档 7.3 节），和 IQL 用同一个训练函数模板。
4. 慢模拟器的进程池版本（每个 worker 一个环境），对接 `online/ppo.collect`。
5. 风险敏感评估：在 `evaluate.py` 加 CVaR 的估计和 bootstrap 置信区间。

**机器学习和深度学习**
1. 把 `ml-pipeline.md` 和 `dl/*.md` 里的代码整理成包（例如 `src/sigml/`、`src/lobdl/`），目录风格照 `execrl`：`config.py`、`data.py`、模型模块、`train.py`、测试。
2. 统一评估：三分类对数损失、校准、经济指标，放在一个模块里供 ML、DL、RL 共用。
