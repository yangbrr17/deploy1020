# deploy1020

两套技术方案的参考笔记：

1. **执行信号模型**：用订单簿数据（OBI、OFI、microprice 等）做 tick 级短期信号。输入、输出、机器学习和深度学习的改进路线、pipeline、训练指标、网络设计，以及强化学习放在哪一层。
2. **内部 Agent 平台**：架构选型、skill 链的后台设计、单机部署方案（组件、模型路线、资源和费用）。

只写通用的技术思路，不涉及任何机构、业务和数据。

网页版由 GitHub Pages 发布（`site/build.py` 把这些 Markdown 转成网页，推送后自动部署）。

## 目录

**信号建模**（从总览开始读）

| 文件 | 内容 |
|---|---|
| [`docs/signal-model/modeling-guide.md`](docs/signal-model/modeling-guide.md) | **总览**：问题定义、截面还是时序、输入和输出、ML / DL / RL 三条路线、pipeline、训练指标、先做什么 |
| [`docs/signal-model/features-and-targets.md`](docs/signal-model/features-and-targets.md) | 输入变量全表（公式、直觉、经济含义）、经济学概念、跨股票标准化、预测目标对比 |
| [`docs/signal-model/ml-pipeline.md`](docs/signal-model/ml-pipeline.md) | 机器学习路线：逻辑回归 → GBDT → 生存模型；完整 pipeline；指标；LightGBM 超参起点 |
| [`docs/signal-model/deep-learning.md`](docs/signal-model/deep-learning.md) | 深度学习路线：模型谱系、基准的问题、第一版网络结构、损失、训练超参、CPU 推理 |
| [`docs/signal-model/reinforcement-learning.md`](docs/signal-model/reinforcement-learning.md) | 强化学习：放在哪一层、MDP、算法选择、模拟器、评估与样本量 |
| [`docs/signal-model/repos.md`](docs/signal-model/repos.md) | GitHub 开源仓库清单：用途、许可证、维护状态 |
| [`docs/signal-model/execution-signal-plan.md`](docs/signal-model/execution-signal-plan.md) | 信号在执行里怎么用：阈值、进度带、评估、分步计划 |
| [`docs/signal-model/pooled-model.md`](docs/signal-model/pooled-model.md) | 从逐只股票建模改成所有股票共用一个模型 |

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
