# 交付报告主题：规范与用法

内部报告统一用这套主题交付。版式参考财经终端研报（Bloomberg 一类）：黑色页头，结论写进标题，紧接一条关键指标，正文是一行一行对齐的面板。产出是**一个 HTML 文件**，有以下特点：

- 不联网也能打开，字体已经内嵌。
- 浏览器里「打印 → 存储为 PDF」就是排好的 A4 版。
- 浅色是纸面风格，深色是终端风格（黑底、琥珀色标签）。
- 每张图都能悬停看数值，点「数据」看图背后的数字。

几个入口：

- **看效果**：示例报告 [中文](../../report/example.html) · [English](../../report/example-en.html)（合成数据，用到了全部图型和面板；页头右上角「中文 | EN」可以互相切换）· 空白模板 [中文](../../report/template.html) · [English](../../report/template-en.html)
- **代码**：[`src/reportkit/`](../../src/reportkit/__init__.py)，只依赖 Python 标准库；ROC、校准图和示例报告需要 numpy
- **单独下载**：[theme.css](../../report/theme.css) · [theme.js](../../report/theme.js) · [template.html](../../report/template.html) · [template-en.html](../../report/template-en.html)（打开后「另存为」，字体在 `report/fonts/`）

## 1. 版式：一行 12 栏，行内对齐

整份报告从上到下是这几块（对照[示例报告](../../report/example.html)看）：

| 位置 | 部分 | 样子 |
|---|---|---|
| 1 | 页头 | 黑底一条：左边产品线和编号，右边日期、版本、状态和按钮 |
| 2 | 标题区 | kicker（领域 / 主题）、粗体大标题（就是结论）、导语、署名行 |
| 3 | 指标条 | 上方一条粗线，3–6 格，格与格之间细竖线；大号数值、带箭头的变化、小走势 |
| 4 | 第一行 | 要点（8 栏）+ 本报告（4 栏，灰底，含「需要决策」） |
| 5 | 章节导航 | 一行章节链接，滚动时吸顶，当前章节加下划线 |
| 6 | 章节 | 粗线 + 编号 + 标题 + 一句话导语；下面是若干行面板 |
| 7 | 面板 | 顶部粗线；标签（图 n / 表 n）、标题、副标题和图例；正文；底部细线和来源行，右下角「数据」按钮 |
| 8 | 页脚 | 粗线；版本记录、生成信息 |

版式规则如下：

- **一行 12 栏，面板宽度加起来必须等于 12。** 常用组合：`12`（宽表）、`8 + 4`（主图加窄图或文字）、`7 + 5`（表加图）、`6 + 6`、`4 + 4 + 4`（三张同类图）。Python 路线下加起来不等于 12 会直接报错。
- **同一行的面板三段对齐。** 每个面板分三段：头（标签、标题、副标题、图例）、正文、来源。同一行里，标题从同一高度开始，图的上沿对齐，来源线对齐（CSS subgrid）。图会长到和同行最高的表一样高。
- **一行放同一层级的东西。** 两张互相印证的图并排，表和解释它的图并排。不要把无关的图塞进同一行凑满 12 栏。
- **行高**用 `r.row(height=…)` 统一设定：一行三张图约 220–240px，主图约 250–280px。
- **文字也进网格。** 解释性文字用 `r.text()` 或 `r.callout()` 放进行里，和图一起对齐。只有很短的段落才用全宽的 `r.p()`。
- 屏幕宽度在 900px 以下时，所有面板变成一栏；5 栏以上的表可以左右滑动，不会把列挤扁。

## 2. 报告结构与顺序

先结论，后证据，再细节。读者只看标题、指标条和要点，也能知道结论和要他做什么。

| 顺序 | 部分 | 内容 | 写法 |
|---|---|---|---|
| 1 | 标题区 | kicker、标题、导语、署名行 | 标题就是全文结论，带最关键的数字 |
| 2 | 指标条 | 3–6 个指标：数值、相对基线的变化、可选小走势 | 只放结论里提到的数 |
| 3 | 要点 + 本报告 | 3–5 条要点（8 栏）；数据、标签、方法、检验和「需要决策」（4 栏） | 每条要点先一句加粗的结论，再给数字和条件 |
| 4 | 主要结果 | 结论对应的表和图 | 章节导语写这一节的结论 |
| 5 | 稳定性 | 按天、按时段、按子样本看结论是否成立 | 先看分布，再看极端情况 |
| 6 | 驱动因素 | 特征、相关性、贡献分解 | 说明「为什么」 |
| 7 | 数据 | 清洗步骤、分布、口径 | 方法细节放这里，不放开头 |
| 8 | 检验与风险 | 假设检验表 + 结论成立的前提 | 状态用标签：成立、不成立、待定 |
| 9 | 下一步 | 事项、负责人、时间、状态 | 状态只用：完成、进行中、未开始、阻塞、有风险、待定 |
| 10 | 附录 | 公式、复现命令 | 编号 A、B |
| 11 | 版本记录 | 版本、日期、说明 | 最新的在最上面 |

## 3. 写作规则

- **标题写结论。** 报告标题、章节导语、每张图表的标题都写「说明了什么」，比如「线性模型在 15 / 15 个测试日都更好」，不写「单日 AUC」。
- **单位进副标题。** 坐标轴上不写单位，副标题写「单日 AUC，测试期」「样本数，40 个交易日合计」。
- **数字从代码来。** 标题和正文里的数字都用变量格式化，不手填；同一指标小数位一致（AUC 3–4 位，百分比 1 位）。
- **负号和正号。** 变化量带符号（`+0.025`、`−0.024`），负号用真正的减号（`fmt_num` 和图表会自动处理）。
- **颜色只表达一件事。** 同一个模型在所有图里同一个颜色；绿和红只表示变好和变差，而且总配 ▲▼ 或正负号。
- **说「显著」要有检验。** 没做检验就写「高 0.02」。
- **来源行写条件。** 数据范围、样本、方法和异常处理写在面板底部的来源行。

## 4. 图表怎么选

先确定读者要做什么，再选图。颜色最后考虑。

| 读者要做的事 | 图型 | Python |
|---|---|---|
| 看一个数和它的变化 | 指标条 | `r.kpis([...])` |
| 看随时间（或有序 x）的走势 | 折线；右侧标签是最后的值 | `charts.line(x, {名字: 值})` |
| 比较、排序 | 柱状；类别名长时自动横向 | `charts.bar(类别, 值, sort="desc")` |
| 看每期的正负变化 | 柱状，正绿负红 | `charts.bar(期, 值, tone="sign")` |
| 从 A 一步步走到 B（归因） | 瀑布图 | `charts.waterfall([("起点", v, "total"), ("一步", dv), …])` |
| 每项有前后两个值 | 哑铃图 | `charts.dumbbell(类别, {"前": a}, {"后": b})` |
| 看一个量的分布 | 直方图；时间间隔用对数刻度 | `charts.hist(值, log_x=True)` |
| 比较几组的分布 | 箱线图（带原始点） | `charts.box({组: 值})` |
| 看矩阵（相关系数、时段 × 日期） | 热力图；以 0 为中心用双色，否则单色 | `charts.heatmap(矩阵, 行, 列, domain=(-1, 1))` |
| 看两个量的关系 | 散点，可加回归线和对角线 | `charts.scatter({名字: (x, y)}, fit=True)` |
| 同一指标看很多对象 | 小多图（共用纵轴） | `charts.multiples(x, {对象: 值})` |
| 看分类器 | ROC、校准曲线 | `charts.roc(y, {名字: 分数})`、`charts.calibration(y, {名字: 概率})` |
| 强调一条，其余作背景 | 折线 `emphasis=`，柱状 `highlight=` | 其余自动变灰 |

## 5. 图的统一风格（主题已经固定，不用再调）

- **纵轴在右侧**，网格是实的细横线，零线和底线是黑色。
- **折线**线宽 2px；每条线最后的值放在右侧的彩色标签里，标签不会重叠，挡住的刻度会自动让开。
- **柱子**最粗 30px，几乎是方角；横向柱的数值写在柱尾，不画坐标轴。
- **瀑布图**：合计是深灰色，增加是绿色，减少是红色，虚线连接；纵轴不从 0 开始时，柱底画断裂符号。
- **图例在面板头里**，两条及以上的系列才有图例；参考线（随机、完美校准、阈值）是虚线，在图例里说明。
- **颜色跟着系列名字走**：同一个模型在整份报告里同一个颜色。显示名和颜色身份不同时用 `key`，例如 ROC 图例里带 AUC。
- 每张图都能悬停（或用方向键）看数值，右下角「数据」按钮显示图背后的表。
- 不用饼图、环形图、三维图、双 y 轴、渐变色和阴影。

## 6. 组件一览

| 组件 | Python | HTML |
|---|---|---|
| 页头和标题区 | `Report(title, dek=, kicker=, kind=, doc_id=, status=, date=, version=, meta={})` | `.rk-mast`、`header.rk-hero` |
| 指标条 | `r.kpis([{label, value, fmt, unit, delta, delta_fmt, better, ref, note, spark}])` | `.rk-strip` > `.rk-kpi` |
| 要点 | `r.takeaways([...], span=8)` | `.rk-panel.rk-text` > `ol.rk-points` |
| 本报告 | `r.info({…}, decision=…, span=4)` | `.rk-panel.rk-info` > `dl.rk-defs` + `.rk-decide` |
| 章节 | `r.section(title, lead=)`、`r.appendix(title, lead=)` | `section.rk-sec` > `.rk-sec-head` |
| 行 | `with r.row(height=250):` | `.rk-row`（`style="--h:250px"`） |
| 图 | `r.chart(spec, span=, title=, subtitle=, source=)` | `figure.rk-panel.rk-fig` > `.rk-chart` > JSON |
| 表 | `r.table(data, span=, fmt=, best=, delta=, bars=, heat=, spark=, highlight=, total=)` | `figure.rk-panel.rk-tbl` > `table.rk-table` |
| 文字 | `r.text(body, title=, span=, label=)` | `figure.rk-panel.rk-text` |
| 提示 | `r.callout(body, title=, kind=, span=)`，kind：note、warn、risk、good、decision | `.rk-panel.rk-call-warn` 等 |
| 状态标签 | `badge("成立", "good")`，tone：good、warn、serious、bad、info、neutral | `span.rk-badge.rk-good` |
| 下一步 | `r.actions([{事项, 负责人, 时间, 状态}], title=)` | `table.rk-table.rk-actions` |
| 静态图 | `r.figure(matplotlib_fig)`、`r.image(path)` | `.rk-plate` |
| 公式、代码 | `r.math(tex)`、`r.code(text, lang=)` | `.rk-math[data-tex]`、`.rk-code` |
| 全宽段落 | `r.p(text)`、`r.bullets([...])` | `.rk-prose` |
| 版本记录 | `r.changelog([(版本, 日期, 说明)])` | `footer.rk-foot` |

表格参数：

| 参数 | 作用 |
|---|---|
| `fmt` | 每列数字格式，如 `".4f"`、`",d"`、`"+.1%"` |
| `best` | 标出每列最好的值（`"max"` 或 `"min"`），加粗并带小方块 |
| `delta` | 正负值上色：变好绿、变差红 |
| `bars` | 单元格里的横条 |
| `heat` | 单元格底色：`"diverging"` 正绿负红，`"sequential"` 单色深浅 |
| `spark` | 单元格里放一个数列，画成小走势线 |
| `highlight` / `total` | 重点行（左侧色条、加粗）和合计行（上方黑线） |

字符串参数里可以用 `**加粗**`、`` `代码` ``、`[文字](链接)` 和行内公式（两个美元符号包住 LaTeX）；字面的美元符号写成 `\$`。其余内容全部转义。

## 7. 图表 spec（手写 HTML 时用）

`charts.*` 生成的就是这些字典；手写时直接写 JSON，放进 `<div class="rk-chart"><script type="application/json">…</script></div>`。格式字符串统一为：`.2f`、`,.0f`（千分位）、`+.1f`（带正号）、`.1%`、`.2s`（1.2K、3.4M）、`int`。

| type | 必填 | 常用选填 |
|---|---|---|
| `line` | `x`、`series: [{name, values}]` | `yFormat`、`xFormat`、`yMin`、`yMax`、`zero`、`xLog`、`markers`、`area`、`emphasis`、`tags: false`、`refs: [{y, label}]`、`diagonal`、`labels: [{x, y, text}]`、`xLabel`、`xName` |
| `bar` | `categories`、`series: [{name, values}]` | `horizontal`、`stacked`、`tone: "sign"`、`valueFormat`、`labels`、`highlight`、`axis`、`refs: [{x, label}]` |
| `waterfall` | `steps: [{label, value, kind: "total"/"delta"}]` | `valueFormat`、`better: "up"/"down"`、`yMin`（纵轴下限，大于 0 时画断裂符号） |
| `dumbbell` | `categories`、`a: {name, values}`、`b: {name, values}` | `valueFormat`、`min`、`max` |
| `hist` | `edges`（n+1 个）、`series: [{name, counts}]` | `xLog`、`xLabel`、`xFormat`、`refs: [{x, label}]` |
| `box` | `groups: [{name, n, min, q1, median, q3, max, points}]` | `valueFormat`、`yMin`、`yMax` |
| `heatmap` | `rows`、`cols`、`values`（二维） | `scale: "diverging"/"sequential"`、`domain`、`valueFormat`、`labels: false`、`scaleLabel` |
| `scatter` | `series: [{name, x, y, labels}]`（最多 3 组） | `fit`、`diagonal`、`xLabel`、`yLabel`、`xFormat`、`yFormat` |
| `multiples` | `x`、`panels: [{name, values}]` | `cols`、`yFormat`、`shareY: false`、`refs: [{y}]` |

系列还可以带 `color`（`s1`–`s8`、`gray`、`ink`）、`key`（颜色身份）和 `dash: true`。缺失值写 `null`，折线在那里断开。

## 8. 设计规范

**颜色。** 界面只有一个强调色（浅色是蓝，深色是琥珀）；结构线是黑色（深色模式是琥珀色）。绿和红只表示变好和变差。

| 角色 | 浅色（纸面） | 深色（终端） | 用在哪里 |
|---|---|---|---|
| 背景 | `#ffffff` | `#000000` | 页面和图表底色 |
| 正文 | `#111111` | `#ececea` | 标题、正文、数值 |
| 次要文字 | `#454545` | `#c2c2bd` | 导语、表头、图例 |
| 弱化文字 | `#6b6b6b` | `#9a9a94` | 来源、标签、注释 |
| 结构线 | `#111111` | `#ffa028` | 指标条、章节和面板顶部的粗线 |
| 强调色 | `#2154d8` | `#ffa028` | kicker、章节号、链接 |
| 变好 / 变差 | `#0a8a3a` / `#d1242f` | `#3ccf6e` / `#ff5c5c` | 指标变化、增减、表格 delta |
| 页头 | 黑底白字，琥珀色品牌字 | 同左 | 页头 |

**图表配色**按顺序分配，不循环使用：

- 浅色：`#2154d8` 蓝、`#e34480` 品红、`#eb9a04` 琥珀、`#37a4d5` 天蓝、`#e06623` 橙、`#069f8e` 青绿、`#834cd4` 紫、`#3d8f35` 绿。
- 深色：同样的色相换成适合黑底的明度。

这套顺序用色觉缺陷模拟校验过：

- 相邻两色在红色盲和绿色盲下的色差都不低于 12，正常视觉下不低于 20（OKLab ×100）。
- 前 3 色两两之间也通过，所以散点图最多 3 组。
- 浅色下琥珀和天蓝对白底的对比度低于 3:1，所以每张图都配了「数据」视图。

**字体。** 拉丁字母和数字用 Inter（OFL 许可，来自 visual-repo-202610 的 `kit/fonts`，已内嵌），中文用系统字体（苹方、微软雅黑、思源黑体）。英文报告的标题、导语和章节标题用 Source Serif 4（同样来自 visual-repo-202610，OFL），只在英文报告里内嵌。

- 正文 14px，标题 34px 粗体。
- 表格和坐标轴用等宽数字，大号指标用默认数字。

**打印。** 打印永远是浅色、A4 纵向，页边距 12 / 10 mm。打印前版面按 A4 宽度重排，图按新宽度重画，字号不会被缩小。导航和按钮会隐藏，面板不跨页断开。

## 9. 英文报告

同一套版式和图表，加 `lang="en"` 就是英文版：

```python
r = Report("The linear model has peaked at 0.792 AUC; nonlinear terms add only 0.0007",
           dek="How three models built on the same six features perform out of sample ...",
           kicker=["Quant research", "Model evaluation"], status="review", date="24 Oct 2026",
           lang="en", translations={"zh": "report-zh.html"})       # translations 会在页头加「中文 | EN」切换
r.chart(charts.roc(y, scores, lang="en"), span=4, title="Three ROC curves collapse onto two")
```

- 自己写标签的图表函数（`roc`、`calibration`、`hist`、`bar`）也传 `lang="en"`，坐标轴和图例会用英文。
- 示例的中英文两版共用一套版式，只有文字不同（`src/reportkit/example.py` 里的 `TEXT`）。命令行：`python -m reportkit example report-en.html --lang en`、`python -m reportkit template blank-en.html --lang en`。
- 一份报告出两种语言时，各生成一个文件，用 `translations` 互相链接；页头的「中文 | EN」就是切换按钮。

**英文版的样式差异**（只在 `<html lang="en">` 下生效，网格、面板和图表与中文版完全一样）：

| 元素 | 中文版 | 英文版 |
|---|---|---|
| 标题、导语、章节标题 | 无衬线粗体 | Source Serif 4 衬线体（OFL，内嵌） |
| 标签（kicker、图号、表头、指标名、署名） | 原样 | 全大写、加字距（small caps 风格） |
| 正文 | 14px，行高 1.62 | 14.5px，行高 1.56，自动断字，行宽更窄 |
| 界面文字 | 图 n、表 n、要点、本报告、数据 | Figure n、Table n、Key points、About this note、Data |
| 状态和提示 | 草稿 / 评审中 / 定稿；注意、风险 | Draft / In review / Final；Caution、Risk |

**英文写作规则：**

- 标题用 sentence case，写结论而不是主题：「The linear model has peaked at 0.792 AUC」，不写「Model Evaluation Results」。
- 数字：千分位用逗号（45,000）；区间用 en dash，两边不留空格（0.717–0.844）；负号用减号（−0.024）；变化量带符号（+0.025）。
- 比较写 `vs`（不加句点）；日期写 `24 Oct 2026`；单位写在副标题，例如「Daily AUC, test period」。
- 一份报告里拼写统一（英式或美式选一种，示例用英式：standardised、neighbouring）。
- 缩写第一次出现时写全称，例如 AUC (area under the ROC curve)、CI (confidence interval)；没有做检验不写 significant。

## 10. 交付前检查

- 报告标题、章节导语、每个图表标题都是结论，不是主题。
- 指标条和要点里的每个数，在正文里都有对应的图或表。
- 每一行的面板宽度加起来是 12；同一行放的是同一层级、互相关联的内容。
- 每个面板都有来源行；单位在副标题里。
- 同一个对象在所有图里名字相同（颜色才会一致）。
- 浏览器打开没有「图表渲染失败」；切换深浅色、缩到手机宽度都正常。
- 打印预览里没有被切断的面板。
- 文件是单个 HTML（搜索不到 `href="theme.css"`）。

## 11. 给 Copilot 的提示词

在 Copilot Chat 里让它生成报告时，把下面这段放在前面：

```text
用 reportkit 生成交付报告（src/reportkit，规范在 docs/report-theme/README.md，示例在 src/reportkit/example.py）。要求：
1. Python 路线：Report(...) -> kpis -> row(takeaways span=8 + info span=4) -> section(..., lead=...) + row(...) ...
   -> actions -> appendix -> changelog -> save。不手写 HTML/CSS，不引入其他前端或图表库。
2. 每个 section 由若干 `with r.row(height=...)` 组成，行内面板的 span 加起来等于 12；同一行只放互相关联的内容。
3. 报告标题、章节导语、每个图表标题都写结论；单位写在 subtitle；条件写在 source。
4. 所有数字从变量格式化得到，不手填；同一指标小数位一致；变化量带正负号。
5. 图表只用 reportkit.charts，按规范第 4 节选图；不用饼图、双 y 轴；同一个对象在所有图里用同一个系列名。
6. 表格用 r.table：数字列给 fmt，最好值用 best，变化列用 delta；状态用 badge(...)。
7. 引用图表编号用 r.chart / r.table 的返回值，不手写。
8. 必须有「检验与风险」（含结论成立的前提）和 r.actions 的下一步。
9. 生成后运行脚本，确认文件能打开、没有「图表渲染失败」。
10. 英文报告用 Report(..., lang="en")，roc / calibration / hist / bar 也传 lang="en"；标题用 sentence case，区间用 en dash（0.717–0.844）。
```

## 12. 参考与来源

- **版式风格**：参考财经终端研报的版式，包括黑色页头、粗结构线、结论式标题、指标条、右侧纵轴、线尾数值标签和来源行。只借鉴版式，不使用任何机构的标志、专有字体或品牌色。
- **[visual-repo-202610](https://github.com/yangbrr17/visual-repo-202610) 里用到的部分**：
  - Inter 和 Source Serif 4 字体文件（都是 OFL）；
  - 设计规范里的结论式标题、单位行、来源行、「文字只用文字色、颜色只给图形」等规则；
  - 组件目录里的图型（瀑布图、哑铃图、箱线图、小多图）；
  - 截图自检的思路：检查文字重叠和行内对齐。
- **没有用到的部分**：
  - 那个仓库的图表引擎是 ECharts（约 1 MB），这里保留了自写的 SVG 引擎，单文件约 230 KB。
  - `third_party/lieflat-charts` 是非商业许可，没有使用其中的任何代码或模板。
