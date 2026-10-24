# deploy1020

两套技术方案的参考笔记：

1. **执行信号模型**：用订单簿数据（OBI、OFI、microprice 等）做 tick 级短期信号。输入、输出、机器学习和深度学习的改进路线、pipeline、训练指标、网络设计，以及强化学习放在哪一层。
2. **内部 Agent 平台**：架构选型、skill 链的后台设计、单机部署方案（组件、模型路线、资源和费用）。

只写通用的技术思路，不涉及任何机构、业务和数据。

网页版由 GitHub Pages 发布（`site/build.py` 把这些 Markdown 转成网页，推送后自动部署）。公式用 LaTeX 书写（`$...$` 和 `$$...$$`），网页上用 KaTeX 渲染；每个代码块和公式都有复制按钮。文档里的 Python 代码都在本地跑通过；`src/` 下的代码包有测试，网页上每个 `.py` 文件都有源码页。

## 目录

**信号建模**（从总览开始读）

| 文件 | 内容 |
|---|---|
| [`docs/signal-model/modeling-guide.md`](docs/signal-model/modeling-guide.md) | **总览**：问题定义、截面还是时序、输入和输出、ML / DL / RL 三条路线、pipeline、训练指标、先做什么 |
| [`docs/signal-model/features-and-targets.md`](docs/signal-model/features-and-targets.md) | 输入变量全表（公式、直觉、经济含义）、经济学概念、跨股票标准化、预测目标对比 |
| [`docs/signal-model/event-arrivals.md`](docs/signal-model/event-arrivals.md) | 订单 / 成交 / 中间价变动的到达分布：泊松、ACD、Hawkes、queue-reactive、队列耗尽时间；检验方法；怎么当特征、基线和模拟器；调研路线；全部公式的 LaTeX 源码 |
| [`docs/signal-model/arrival-tests.md`](docs/signal-model/arrival-tests.md) | **可直接执行的检验方案**：需要的数据字段、H1–H9 每个假设对应的分布、检验统计量、判定阈值、结果表、执行步骤（代码在 [`src/arrivals/`](src/README.md)） |
| [`docs/signal-model/ml-pipeline.md`](docs/signal-model/ml-pipeline.md) | 机器学习路线：逻辑回归 → GBDT → 生存模型；完整 pipeline；指标；LightGBM 超参起点 |
| [`docs/signal-model/deep-learning.md`](docs/signal-model/deep-learning.md) | 深度学习总览：精选模型对比、通用设计、训练指标、和 GBDT 比较、i9 CPU 上的训练和推理 |
| [`docs/signal-model/dl/`](docs/signal-model/dl/ofi-tcn-gru.md) | 每个模型一页：[DeepLOB](docs/signal-model/dl/deeplob.md)、[BiN-CTABL](docs/signal-model/dl/bin-ctabl.md)、[Deep OFI](docs/signal-model/dl/deep-ofi.md)、[TLOB](docs/signal-model/dl/tlob.md)、[推荐的 OFI-TCN-GRU](docs/signal-model/dl/ofi-tcn-gru.md)：数学、逐层结构、参数怎么选、怎么分析、可运行代码 |
| [`docs/signal-model/reinforcement-learning.md`](docs/signal-model/reinforcement-learning.md) | 强化学习：MDP、奖励和价值函数、十类方法比较、推荐方案（bandit + DR）的完整算法、模拟器、样本量；第 13 节是模块化代码（[`src/execrl/`](src/README.md)）的结构、训练函数和玩具模拟器上的结果 |
| [`docs/signal-model/execution-signal-plan.md`](docs/signal-model/execution-signal-plan.md) | 信号在执行里怎么用：阈值、进度带、评估、分步计划 |

**交付报告主题**（[`docs/report-theme/README.md`](docs/report-theme/README.md)）

内部报告的统一格式：单个 HTML 文件，结论先行，指标卡、编号章节、交互图表，能打印成 A4。网页上有[示例报告](report/example.html)和空白模板；代码在 [`src/reportkit/`](src/reportkit/__init__.py)。

**代码**（[`src/README.md`](src/README.md)）

| 包 | 内容 |
|---|---|
| [`src/arrivals/`](src/arrivals/__init__.py) | 到达分布检验 H1–H9：清洗、统计量、ACD、Hawkes、Cont–de Larrard、结果表和命令行 |
| [`src/execrl/`](src/execrl/__init__.py) | 执行决策：配置、动作和屏蔽、奖励、数据容器、玩具模拟器、bandit + DR、约束 FQI、FQE、IQL、PPO、训练函数和命令行 |
| [`src/reportkit/`](src/reportkit/__init__.py) | 交付报告主题：`Report`、图表 spec、主题 CSS / JS、空白模板 |
| [`src/tests/`](src/tests/test_execrl.py) | 25 个 pytest 测试：`PYTHONPATH=src python -m pytest -q src/tests` |

「笔记（需密码）」栏目里的页面在浏览器里用密码解密，仓库里只有密文（`site/encrypted/`，生成方法见 `site/encrypt_page.py`）。

接手修改前先读 [`.github/copilot-instructions.md`](.github/copilot-instructions.md)：写法约定和按优先级排的待办。

**Agent 平台**

| 文件 | 内容 |
|---|---|
| [`docs/agent-platform/architecture.md`](docs/agent-platform/architecture.md) | 选型结论、框架盘点与许可证、架构、技术栈、风险、MVP 范围 |
| [`docs/agent-platform/skill-chain-backend.md`](docs/agent-platform/skill-chain-backend.md) | skill 链的后台：共享工作台、输入清单、事件驱动、版本与过期 |
| [`docs/agent-platform/deployment.md`](docs/agent-platform/deployment.md) | 部署到一台云服务器：组件、两条模型路线、资源与费用、分阶段计划、上线前要确认的事 |

## 内容边界

这个仓库是公开的。往里加东西之前先检查：

- 不写机构名、人名、部门、办公地点。
- 不写内部系统名、库名、表名、字段名、指标口径。
- 不放任何真实数据、客户信息、订单或成交记录；例子一律用模拟数据。
- 价格、版本、许可证都注明查询日期，用之前重新核对。
