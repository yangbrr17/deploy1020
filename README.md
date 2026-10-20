# deploy1020

两套技术方案的参考笔记：

1. **内部 Agent 平台**：架构选型、skill 链的后台设计、单机部署方案（组件、模型路线、资源和费用）。
2. **执行信号模型**：用 OBI / OFI 做 tick 级信号，怎么定标签、阈值、模型和评估，以及强化学习放在哪一层。

只写通用的技术思路，不涉及任何机构、业务和数据。

## 目录

| 文件 | 内容 |
|---|---|
| [`docs/agent-platform/architecture.md`](docs/agent-platform/architecture.md) | 选型结论、框架盘点与许可证、架构、技术栈、风险、MVP 范围 |
| [`docs/agent-platform/skill-chain-backend.md`](docs/agent-platform/skill-chain-backend.md) | skill 链的后台：共享工作台、输入清单、事件驱动、版本与过期 |
| [`docs/agent-platform/deployment.md`](docs/agent-platform/deployment.md) | 部署到一台云服务器：组件、两条模型路线、资源与费用、分阶段计划、上线前要确认的事 |
| [`docs/signal-model/obi-ofi-basics.md`](docs/signal-model/obi-ofi-basics.md) | OBI、OFI、微观价格的定义，特征清单 |
| [`docs/signal-model/execution-signal-plan.md`](docs/signal-model/execution-signal-plan.md) | 信号在执行里怎么用：标签、幅度、时长、阈值、进度带、模型、评估、强化学习、分步计划 |
| [`docs/signal-model/pooled-model.md`](docs/signal-model/pooled-model.md) | 从逐只股票建模改成所有股票共用一个模型 |

## 内容边界

这个仓库是公开的。往里加东西之前先检查：

- 不写机构名、人名、部门、办公地点。
- 不写内部系统名、库名、表名、字段名、指标口径。
- 不放任何真实数据、客户信息、订单或成交记录；例子一律用模拟数据。
- 价格、版本、许可证都注明查询日期，用之前重新核对。
