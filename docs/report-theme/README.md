# 交付报告主题：规范与用法

内部报告统一用这套主题交付。产出是**一个 HTML 文件**：不依赖网络就能打开，浏览器里「打印 → 存储为 PDF」得到 A4 版本，浅色和深色都能看，图表有悬停提示和「表格」视图。

- **看效果**：[示例报告](../../report/example.html)（合成数据，用到了全部组件）· [空白模板](../../report/template.html)
- **代码**：[`src/reportkit/`](../../src/reportkit/__init__.py)，只依赖 Python 标准库（ROC、校准图和示例报告要 numpy）
- **单独下载**：[theme.css](../../report/theme.css) · [theme.js](../../report/theme.js) · [template.html](../../report/template.html)（打开后「另存为」）

## 1. 两种用法

### 1.1 Python 生成（推荐）

把仓库里的 `src/reportkit/` 整个复制到项目里（或者 `PYTHONPATH=src`），然后：

```python
from reportkit import Report, charts, badge

r = Report(
    "方向信号模型评估：线性模型够用，下一步该换标签",          # 标题写结论
    subtitle="同一组特征下三种模型的样本外表现",               # 回答什么问题、用了什么数据
    kind="技术评估报告", status="review",                    # draft / review / final
    meta={"日期": "2026-10-24", "作者": "（作者）", "版本": "v1.0", "数据": "40 个交易日"},
)
r.summary([
    "**6 个特征的线性模型比单特征高 0.024 AUC。** 按天 bootstrap 的 95% 区间是 [0.022, 0.027]。",
    "**非线性项几乎没有增益。** AUC 只多 0.0007，低于 0.005 的门槛。",
], decision="是否把下一阶段的工作转到标签定义上。")
r.kpis([
    {"label": "AUC · 测试集", "value": 0.784, "fmt": ".3f", "delta": 0.024, "delta_fmt": "+.3f",
     "better": "up", "ref": "对比单特征", "spark": daily_auc},
    {"label": "对数损失", "value": 0.5606, "fmt": ".4f", "delta": -0.0221, "better": "down", "ref": "对比单特征"},
])

r.section("主要结果")
t = r.table(rows, title="线性模型明显好于单特征", fmt={"AUC": ".4f"}, best={"AUC": "max"})
r.p(f"{t} 里三个模型的差别集中在第一步。")             # 编号由 reportkit 生成，正文直接引用
with r.cols(2):
    r.chart(charts.roc(y, {"单特征": s1, "线性 6 特征": s2}), title="三条 ROC 曲线几乎重合在两条上")
    r.chart(charts.calibration(y, {"线性 6 特征": p2}), title="预测概率校准良好")
r.callout("balanced accuracy 依赖阈值，AUC 不依赖。", kind="warn", title="不要用 balanced accuracy 比较模型")

r.section("下一步")
r.actions([{"事项": "换成下一次变动方向的标签", "负责人": "（待定）", "时间": "第 1 周", "状态": "进行中"}])
r.appendix("指标定义")
r.math(r"\mathrm{BA}_{\max} = \Phi\left(\Phi^{-1}(\mathrm{AUC})/\sqrt{2}\right)")
r.changelog([("v1.0", "2026-10-24", "首次发布")])
r.save("report.html")
```

示例报告的完整代码在 [`example.py`](../../src/reportkit/example.py)，可以照着改。命令行：

```bash
PYTHONPATH=src python -m reportkit example report.html     # 生成示例报告
PYTHONPATH=src python -m reportkit template blank.html     # 导出空白模板（样式已内联）
PYTHONPATH=src python -m reportkit inline draft.html final.html   # 手写报告内联成单文件
PYTHONPATH=src python -m reportkit assets ./theme          # 复制 theme.css、theme.js、template.html
```

### 1.2 手写 HTML（不用 Python）

1. 复制 [template.html](../../report/template.html)，和 `theme.css`、`theme.js` 放在同一个目录（或者运行上面的 `assets` 命令）。
2. 替换「」里的占位内容，删掉用不到的组件。每个组件上面都有注释说明 class 的用法。
3. 图表写成 JSON（第 6 节），放在 `<div class="rk-chart"><script type="application/json">…</script></div>` 里。
4. 交付前运行 `python -m reportkit inline draft.html final.html`，把样式和脚本内联进去，只发 `final.html`。

## 2. 报告结构

顺序固定。读者只看前两块也能知道结论和要他做什么。

| 部分 | 内容 | 要求 |
|---|---|---|
| 抬头 | 报告类型、状态（草稿 / 评审中 / 定稿）、标题、副标题、日期 / 作者 / 版本 / 数据范围 | 标题写结论，不写主题：「线性模型够用」而不是「模型评估」 |
| 结论 | 3–5 条发现，可选一条「需要决策」 | 每条先一句加粗的结论，再给数字和条件 |
| 关键指标 | 2–4 个指标卡：数值、相对基线的变化、可选的走势小图 | 变化的颜色由「方向 × 越大越好还是越小越好」决定 |
| 正文章节 | 背景与问题 → 数据与口径 → 主要结果 → 诊断 / 检验 → 风险与局限 | 章节、小节、图、表自动编号，正文引用编号 |
| 下一步 | 事项、负责人、时间、状态 | 状态只用：完成、进行中、未开始、阻塞、有风险、待定 |
| 附录 | 公式、方法、复现命令 | 用 A、B 编号 |
| 版本记录 | 版本、日期、改了什么 | 最新的在最上面 |

## 3. 写作规则

- **先结论后细节。** 每节第一段说这一节的结论；每张图、每张表的标题写它说明了什么。
- **数字带口径。** 写清样本、时间范围、条件和单位；区间写成 [下限, 上限]；比较写清「对比什么」。
- **有效数字一致。** 同一列、同一指标用同样的小数位；AUC 3–4 位，百分比 1 位。
- **说「显著」要有检验。** 没有做检验就写「高 0.02」，不写「显著提高」。
- **不贴截图。** 图用 `charts` 画（随主题变色、有提示和表格视图）；matplotlib 图用 `r.figure(fig)` 嵌成 SVG。
- **术语第一次出现时定义。** 放在「数据与口径」的定义表里。
- **局限单独写。** 「风险与局限」一节说清结论成立的前提，而不是在每句话后面加限定。

## 4. 图表怎么选

先决定读者要做什么，再选图。颜色最后考虑。

| 读者要做的事 | 用 | Python |
|---|---|---|
| 看一个数和它的变化 | 指标卡（不是只有一根柱的柱状图） | `r.kpis([...])` |
| 看趋势（时间、阈值、参数） | 折线 | `charts.line(x, {名字: 值})` |
| 比较几个对象的大小 | 柱状图；类别名长时自动变横向 | `charts.bar(类别, 值, sort="desc")` |
| 看一个量的分布 | 直方图；时间间隔用对数刻度 | `charts.hist(值, log_x=True)` |
| 看矩阵（相关系数、混淆矩阵） | 热力图；以 0 为中心用双色，否则单色 | `charts.heatmap(矩阵, 行, 列, domain=(-1, 1))` |
| 看两个量的关系 | 散点，最多 3 组 | `charts.scatter({名字: (x, y)})` |
| 看分类器 | ROC、校准曲线 | `charts.roc(y, {名字: 分数})`、`charts.calibration(y, {名字: 概率})` |
| 强调一条，其余作背景 | 折线加 `emphasis="名字"`，柱状图加 `highlight="类别"` | 其余自动变灰 |

主题里已经固定、不需要再调的规则：

- 只有一个 y 轴。两个量级不同的量画成两张图，或者都换算成以起点为 100 的指数。
- 颜色跟着系列名字走：同一个模型在整份报告的每张图里都是同一个颜色。显示名和颜色身份不同时用 `key`（ROC 图的图例带 AUC，就是这样做的）。
- 两条及以上的系列自动出图例；不超过 4 条且线尾不打架时，名字直接标在线尾。
- 每张图右上角有「表格」按钮，显示图背后的数字；悬停或用方向键看具体值。
- 线宽 2px，柱子最粗 24px、数据端圆角，网格是实线细线，参考线是虚线。
- 不用饼图、环形图、三维图、双 y 轴、彩虹色。

## 5. 组件一览

| 组件 | Python | HTML |
|---|---|---|
| 抬头 | `Report(title, subtitle=, kind=, doc_id=, status=, meta={})` | `header.rk-head`、`.rk-status.rk-st-review` |
| 结论 | `r.summary([...], decision=)` | `section.rk-summary` > `ol.rk-findings` + `.rk-decision` |
| 指标卡 | `r.kpis([{label, value, fmt, unit, delta, delta_fmt, better, ref, note, spark}])` | `.rk-kpis` > `.rk-kpi`；变化 `.rk-good-t` / `.rk-bad-t` |
| 章节 | `r.section(title, lead=)`、`r.subsection(title)`、`r.appendix(title)` | `section.rk-section` > `h2#s1` > `span.rk-n` |
| 正文 | `r.p(text)`、`r.lead(text)`、`r.bullets([...])`、`r.defs({术语: 定义})` | `p`、`ul`、`dl.rk-defs` |
| 提示框 | `r.callout(body, title=, kind=)`，kind：note、warn、risk、good、decision | `.rk-callout.rk-warn` |
| 表格 | `r.table(data, fmt=, best=, delta=, bars=, highlight=)` | `.rk-tbl` > `table.rk-table`；数字列 `.rk-r` |
| 状态标签 | `badge("成立", "good")`，tone：good、warn、serious、bad、info、neutral | `span.rk-badge.rk-good` |
| 图表 | `r.chart(spec, title=, subtitle=, source=)` | `figure.rk-fig` > `.rk-chart` > JSON |
| 静态图 | `r.figure(matplotlib_fig)`、`r.image(path)` | `.rk-plate` |
| 并排 | `with r.cols(2): ...` | `.rk-cols` |
| 下一步 | `r.actions([{事项, 负责人, 时间, 状态}])` | `.rk-actions` |
| 公式、代码 | `r.math(tex)`、`r.code(text, lang=)` | `.rk-math[data-tex]`、`.rk-code` |
| 版本记录 | `r.changelog([(版本, 日期, 说明)])` | `footer.rk-foot` |

正文字符串里可以用 `**加粗**`、`` `代码` ``、`[文字](链接)` 和行内公式（两个美元符号包住 LaTeX）；字面的美元符号写成 `\$`。其余内容全部转义，标题里写 `<` 也不会破坏页面。

## 6. 图表 spec（手写 HTML 时用）

`charts.*` 生成的就是这些字典；手写时直接写 JSON。所有格式字符串用同一套写法：`.2f`、`,.0f`（千分位）、`+.1f`（带正号）、`.1%`、`.2s`（1.2K、3.4M）、`int`。

| type | 必填 | 常用选填 |
|---|---|---|
| `line` | `x`（数字或字符串）、`series: [{name, values}]` | `xLabel`、`yLabel`、`yFormat`、`xFormat`、`yMin`、`yMax`、`zero`、`xLog`、`markers`、`area`、`emphasis`、`refs: [{y, label}]`、`diagonal`、`labels: [{x, y, text}]`、`endLabels: false`、`height` |
| `bar` | `categories`、`series: [{name, values}]` | `horizontal`（默认自动）、`stacked`、`valueFormat`、`yLabel`、`labels`、`highlight`、`height` |
| `hist` | `edges`（n+1 个）、`series: [{name, counts}]` | `xLog`、`xLabel`、`yLabel`、`refs: [{x, label}]` |
| `heatmap` | `rows`、`cols`、`values`（二维） | `scale`：`diverging` 或 `sequential`、`domain`、`valueFormat`、`labels: false` |
| `scatter` | `series: [{name, x, y}]`（最多 3 组） | `xLabel`、`yLabel`、`diagonal`、`refs` |

系列还可以带 `color`（`s1`–`s8` 或 `gray`，固定颜色）、`key`（颜色身份）、`dash: true`（虚线）。缺失值写 `null`，折线在那里断开。

## 7. 设计规范

**颜色。** 页面只有一个强调色（深蓝），其余是中性灰。状态色只表示状态，永远和图标、文字一起出现。

| 角色 | 浅色 | 深色 | 用在哪里 |
|---|---|---|---|
| 纸面 | `#ffffff` | `#1a1a19` | 报告正文和图表底色 |
| 页面背景 | `#f2f1ee` | `#0f0f0e` | 纸面外面 |
| 正文 | `#1b1b1d` | `#f0efec` | 标题、正文、数值 |
| 次要文字 | `#4f4e4a` | `#c9c8be` | 副标题、表头、图例 |
| 弱化文字 | `#6b6a65` | `#a3a197` | 注释、来源、标签名 |
| 强调色 | `#1c5cab` | `#6da7ec` | 链接、章节号、结论标签 |
| 好 / 注意 / 差 | `#0ca30c` / `#fab219` / `#d03b3b` | 同左 | 状态标签、提示框、指标变化 |

**图表配色**（按顺序分配，不循环）：`#2a78d6` 蓝、`#eb6834` 橙、`#1baf7a` 青绿、`#eda100` 黄、`#e87ba4` 品红、`#008300` 绿、`#6250d6` 紫、`#e34948` 红；深色模式用同色相的深色版本。这个顺序用色觉缺陷模拟校验过：相邻两色在红色盲、绿色盲下的色差都在 8 以上（OKLab ×100）。散点图只用前 3 色。浅色下青绿、黄、品红对白底的对比度低于 3:1，所以每张图都有「表格」视图兜底。

**字体。** 系统无衬线字体（苹方、微软雅黑、思源黑体），不加载网络字体；代码用等宽字体。正文 15.5px，行高 1.72；表格和坐标轴的数字用等宽数字，大号指标用默认数字。

**打印。** 永远按浅色打印，A4，页边距 16 / 14 mm；隐藏目录、按钮和提示；图、表、提示框不跨页断开。

## 8. 交付前检查

- 标题、每张图表的标题都是结论，不是主题
- 结论区每条都能在正文里找到对应的数字和图表
- 正文引用的「图 n」「表 n」和实际编号一致（Python 路线自动保证）
- 每个数字有口径；同一指标小数位一致
- 有「风险与局限」和「下一步」
- 浏览器打开没有「图表渲染失败」；切换深浅色、缩到手机宽度都正常
- 打印预览里没有被切断的图表
- 文件是单个 HTML（搜索不到 `href="theme.css"`）

## 9. 给 Copilot 的提示词

在 Copilot Chat 里让它生成报告时，把下面这段放在前面：

```text
用 reportkit 生成交付报告（src/reportkit，说明在 docs/report-theme/README.md）。要求：
1. 用 Python 路线：Report(...) -> summary -> kpis -> section ... -> actions -> appendix -> changelog -> save。
   顺序不变，不手写 HTML/CSS，不引入其他前端库。
2. 标题、图表标题都写结论；结论区 3-5 条，每条以一句 **加粗** 的结论开头，再给数字和条件。
3. 所有数字从代码里的变量格式化得到，不手填；每个数字写清样本、区间、条件；同一指标小数位一致。
4. 图表只用 reportkit.charts（line / bar / hist / heatmap / scatter / roc / calibration），按第 4 节选图；
   不用饼图、双 y 轴；同一个模型在所有图里用同一个系列名。
5. 表格用 r.table，数字列给 fmt，最好值用 best，变化列用 delta；状态用 badge(...)。
6. 引用图表时用 r.chart / r.table 的返回值（"图 3"），不要手写编号。
7. 必须有「风险与局限」一节和 r.actions 的下一步。
8. 生成后运行脚本，确认文件能打开、没有「图表渲染失败」。
```
