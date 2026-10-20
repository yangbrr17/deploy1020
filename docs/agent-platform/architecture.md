# 内部 Agent 平台：选型与架构

2026-10 整理。目标是一个内部使用的 AI Agent：按 Skills 的方式组织能力，能跑固定的 SQL / Python 取数，和用户交互，有权限管控，并把用户身份带进整个流程。

## 需求

- 取数走固定的 SQL / Python 调用。不让大模型直接做全量分析。
- 能和人交互，并通过内部接口访问数据库和其他系统。
- 有权限与安全管控：不同部门按各自权限拿到不同数据，用户不能随意调用所有权限。
- 能拿到用户信息，并带入流程。

## 结论

**不上重型多 agent 框架。** 用一个自己写的薄运行时：Pydantic 定义每个 skill 的输入输出，用 PydanticAI 处理模型调用这一小块；skill 的先后顺序由代码里登记好的「剧本」决定，模型只负责挑剧本、填参数、写解读文字。

「多个 agent」做成多份配置（各自能用哪些 skill 和剧本），不做成互相对话的多个模型。只有出现长时间挂起、等人审批这类需求时，再引入 LangGraph。

模型只做两件事：

1. 把自然语言指令解析成结构化的请求。
2. 按占位符写解读文字。文字里的数字都由代码从数据里填入。

## 正方和反方

| | 支持用框架 | 反对 | 取舍 |
|---|---|---|---|
| 多 agent | LangGraph、MAF 都能给每个 agent 配不同工具 | 查询→加工→渲染是一条直线，拆成多个 agent 只会多出交接点；Anthropic 的数据：多 agent 的 token 消耗约为普通对话的 15 倍，单 agent 约 4 倍 | 「agent」= 一份配置，共用一个运行时 |
| 谁决定下一步 | 框架支持模型路由 | 顺序固定时让模型选下一步是负担 | 模型挑剧本、填参数，代码按剧本执行 |
| 框架依赖 | LangGraph、PydanticAI 成熟、MIT | 框架变动快：LangGraph 1.0 改了接口，AutoGen 进入维护期，有团队用一年后拆掉 LangChain | 只用 PydanticAI 这种薄的、接近自己会写的那层 |
| Skills 和 MCP | — | MCP 不是「上一代」：2025 年底进了 Linux 基金会，规范仍在更新，还新增了 Skills 扩展 | 第一版不需要单独的 MCP 服务器；等别的系统也要调这些 skill 时，再包一层 MCP |

Skills 和 MCP 不必二选一：Skills 组织「什么场景用什么流程」，内部系统的接口仍然需要一层统一的调用方式。

## 框架盘点（2026-10 查）

**18 个 Python 框架**，外加 11 个低代码平台的许可证。

- **可以用（MIT 或 Apache-2.0，仍在维护）：** LangGraph、PydanticAI、Microsoft Agent Framework、LlamaIndex Workflows、Haystack、Apache Burr、CrewAI、Google ADK、Strands Agents、Agno
- **短名单：**
  - **PydanticAI**：类型化输出，`deps` 可以把用户身份类型安全地注入每次 skill 调用，不绑模型厂商。首选。
  - **LangGraph**：需要持久化状态、等人审批、断点续跑时再加。只用 MIT 开源部分；LangSmith 是收费产品。
  - **Microsoft Agent Framework**：全面用 Azure 时可以考虑。
  - **Apache Burr**：轻量状态机，自带界面，还在孵化。
- **不建议做主干：**
  - **Claude Agent SDK**：只能用 Claude；打包进去的命令行工具是专有许可；它的 agent 循环不适合固定流程。
  - **OpenAI Agents SDK**：以 OpenAI 为中心，追踪数据默认发给 OpenAI。
  - **smolagents**：让模型自己写 Python 再执行，在受监管环境里很难过审。
  - **CrewAI**：可用，但匿名遥测默认开着，必须手动关。
- **已经停更或被取代：** AutoGen（维护期）、Semantic Kernel（被 MAF 取代）、ControlFlow（已归档）、CAMEL（偏研究）

**低代码平台的许可证坑：**

| 平台 | 限制 |
|---|---|
| n8n | 只限内部业务用途，不能付费分发 |
| Dify、FastGPT | 改过的 Apache：不能做多租户，前端 logo 不能去掉 |
| Open WebUI | 2025 年改了许可证，用户超过 50 人必须保留品牌 |
| LobeChat | 衍生作品分发要商业许可 |
| Flowise、Onyx、Langfuse | 主体开源，企业功能目录是商业许可 |
| Langflow、LibreChat | MIT，没有坑 |

## 架构

设想：多个 agent，每个能用几个 skill；skill 一个接一个，各有输入输出；查询 skill 跑 SQL 或 Python 拿结构化数据，渲染 skill 负责展示；输入有的来自数据库、有的来自用户，分门别类后合成一份。

| 概念 | 落地方式 |
|---|---|
| Skill | 一个 Python 函数 + 输入输出的 Pydantic 模型 + 一份 YAML 清单。SKILL.md 保留，当说明文档和提示词用 |
| Agent | `agents.yaml` 里的一条：系统提示、能用哪些 skill、能跑哪些剧本。权限由注册表强制，不写在提示词里 |
| Skill 串联 | 剧本：有顺序的 skill 列表加参数映射。注册时就检查上一步的输出能不能接下一步的输入 |
| 数据库和用户输入分开 | 统一的数据信封：每条数据标来源（数据库、用户、推算），带查询 id、时间、用户 id、置信度 |
| 合成一份 | 渲染步骤收齐所有信封按键合并。用户说的数字**不覆盖**数据库的，冲突并排显示 |
| 渲染 | 模型只写「报告规格」：有哪些段、哪个数据集放进哪张表、哪张图，正文用占位符引用数字；代码负责出图出表 |

### 一次请求怎么走

1. 用户在前端提问，通过公司 SSO 登录。
2. 运行时根据用户身份找到对应的 agent 配置。
3. 模型把问题解析成一个类型化的请求：选哪个剧本、参数是什么。Pydantic 校验，不合格就让模型重填；把握不大的地方回头问用户。
4. 代码按剧本依次执行 skill，每一步以**当前用户的身份**调用数据。
5. 结果装进数据信封，渲染 skill 按报告规格出页面、PDF 或 PPT。
6. 模型只写解读文字，文字里的每个数字都要能在数据里找到。
7. 全过程写进只能追加的审计表，可以原样重放。

### Skill 清单示例

```yaml
name: query_trades
version: 1.2.0
owner: data-team
kind: query                 # query / transform / render
description: 按账簿和日期区间取已成交记录。用于明细、下钻。
entrypoint: skills.query_trades:run
input_model: QueryTradesInput     # book_id, start, end, limit<=5000
output_model: TradeRows
side_effects: read
required_permission: TRADES_READ
data_sources: [warehouse.trades_view]
timeout_s: 30
audit_log: true
```

### 取数与权限的原则

- 取数走预定义的、带参数的查询和脚本，模型只负责选工具、填参数，不自己写任意 SQL。这让权限好管，也满足「不让大模型做全量分析」。
- 权限在工具层和数据层强制执行，不能只写在提示词里。用户身份从登录态带进来，每次调工具都以这个用户的身份去查，数据库侧再加行级权限兜底。
- 每一次工具调用都留审计日志：谁、什么时候、用什么参数、拿到了什么。

## 技术栈

| 层 | 推荐 | 备选 / 说明 |
|---|---|---|
| 运行时 | 自写薄层：skill 注册表 + 剧本执行器 + 审计，约 300–500 行 | — |
| 契约 | Pydantic v2 | 导出 JSON Schema 给模型的结构化输出用 |
| 模型调用 | PydanticAI（MIT） | 批准哪家模型就用哪家，框架不绑定 |
| 持久化流程 | 暂不需要 | 要等人审批、长时间挂起时加 LangGraph 或 Temporal |
| 身份传递 | 公司 SSO + OAuth 令牌交换（RFC 8693；Entra 叫 On-Behalf-Of） | 每个下游服务拿短期、限定范围的令牌，不用共享服务账号 |
| 权限策略 | Cedar 或 OPA（都是 Apache-2.0） | 部门级的角色和属性控制足够；按单个对象共享才需要 OpenFGA |
| 数据库兜底 | Postgres / SQL Server 用原生行级安全 | **MySQL 没有原生行级安全**：用按部门的视图（注意 DEFINER 和 INVOKER 的区别），或在应用层用 sqlglot 改写查询 |
| SQL 执行 | 固定的参数化模板 | 将来开放模型写 SQL 时：只读账号、超时、sqlglot 语法树检查、表白名单 |
| Python 执行 | 独立沙箱 Pod（gVisor） | 别在 API 进程里跑；E2B 的多节点离线部署还不成熟 |
| 渲染 | Great Tables 出表，Altair / Vega-Lite 出图，Jinja2 出 HTML，WeasyPrint 出 PDF，python-pptx 出 PPT | 全部可离线；Plotly 导图要装 Chrome |
| 前端 | MVP 用 Streamlit，登录放在网关（oauth2-proxy 或公司网关） | Chainlit 2025 年起改由社区维护，有风险；正式版用 React + assistant-ui 或 CopilotKit；Teams 用 Adaptive Cards |
| 审计与追踪 | Postgres 里只追加的审计表 + OpenTelemetry | 追踪界面：Langfuse（核心 MIT，部分企业功能收费）、Arize Phoenix（ELv2，内部用没问题）、MLflow |
| 打包部署 | 一个镜像两个入口（FastAPI + 后台任务），Postgres、Redis/Valkey、沙箱 Pod | 密钥走密钥管理服务；离线环境要镜像 PyPI、关掉所有遥测 |

单机部署的具体方案见 [`deployment.md`](deployment.md)。

## 安全与合规的设计要求

这些是金融类内部部署常见的要求，做成系统约束，不靠提示词：

1. **按组隔离，不只是「能不能看到数据」。** 还要防止信息通过 agent 从一个组流到另一个组。工作台、审计日志、缓存、检索库都按组分开，一次会话不能跨组。
2. **生成的文字只陈述事实。** 数字只能用占位符；不出现买卖建议和对价格的判断。后置检查拦明显的词，最终措辞由合规确认。
3. **检索内容只接已发布的。** 以后让 agent 引用文档时，只接已批准发布的库。
4. **受限名单。** 涉及具体标的的输出，可以在输出前查一次受限名单。
5. **数据最小化。** 第一版只接必要的那一类数据。
6. **提示注入。** 用户输入和数据库里的文本字段都当成不可信数据，不当指令。
7. **数据外流。** 「私有数据 + 不可信文本 + 对外输出」三者同时存在时风险最高。对外的产出先进草稿，人审批后才发出。
8. **模型风险管理。** 受监管机构一般有模型风险管理要求（例如 SR 11-7、PRA SS1/23），换模型、改提示都要能重放测试。
9. **开源审批。** 依赖越少越好过审。
10. **遥测。** CrewAI、MLflow、Langfuse、Phoenix、Streamlit 都要显式关掉遥测。

## MVP 范围

- 一个组、只读、两三个剧本、五个以内 skill、一种输出格式（HTML）。
- 先建一组标准问题做回归评测：剧本选得对不对、参数对不对、越权请求有没有被拒、解读里的数字是否都来自数据。

## 落地前要确认的事

1. 现有的 skill 仓库长什么样，skill 怎么声明输入输出。
2. 批准用哪家模型、走哪个渠道，数据能不能出内网。
3. 数据库是哪几种（MySQL 没有原生行级安全，影响权限方案）；有没有只读账号和现成的权限系统。
4. 部署到哪个平台：Kubernetes、OpenShift，还是别的；能不能跑 gVisor 这类沙箱。
5. 开源依赖的审批流程多久。
6. 前端放在哪里：Teams、网页，还是别的入口。
7. 只读，还是也要支持写操作；写操作要不要人工确认。
8. 产出是直接对外，还是先给内部的人审核。

## 主要来源

- Anthropic《Building effective agents》https://www.anthropic.com/engineering/building-effective-agents
- Anthropic 多 agent 研究系统 https://www.anthropic.com/engineering/multi-agent-research-system
- Cognition《Don't build multi-agents》https://cognition.com/blog/dont-build-multi-agents
- MAST 失败模式研究 https://arxiv.org/abs/2503.13657
- LangChain v1 迁移 https://docs.langchain.com/oss/python/migrate/langchain-v1
- Skill 与 MCP：https://agentskills.io/specification ；https://modelcontextprotocol.io/specification/latest ；https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/
- 框架许可证：各项目 GitHub 的 LICENSE 文件和 PyPI 元数据；Claude Code 许可 https://github.com/anthropics/claude-code/blob/main/LICENSE.md ；LangSmith 自托管 https://docs.langchain.com/langsmith/self-hosted
- 身份与权限：RFC 8693 https://www.rfc-editor.org/rfc/rfc8693 ；Entra OBO https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow ；MySQL 行级安全 https://aws.amazon.com/blogs/database/implement-row-level-security-in-amazon-aurora-mysql-and-amazon-rds-for-mysql ；sqlglot https://github.com/tobymao/sqlglot
- 许可证坑：n8n https://github.com/n8n-io/n8n/blob/master/LICENSE.md ；Dify https://github.com/langgenius/dify/blob/main/LICENSE ；Open WebUI https://github.com/open-webui/open-webui/blob/main/LICENSE
- 渲染与前端：A2UI https://github.com/a2ui-project/a2ui ；Streamlit 登录 https://docs.streamlit.io/develop/concepts/connections/authentication ；Chainlit 维护状态 https://github.com/Chainlit/chainlit

未核实：OWASP LLM06 原文、PRA SS1/23 对生成式 AI 的具体要求、个别遥测开关的变量名、Kata 和 nsjail 的许可证、Agent Skills 是否归 Linux 基金会管理。
