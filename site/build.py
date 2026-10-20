"""Render README.md and docs/**/*.md into a static site under _site/.

python3 site/build.py            # needs: pip install markdown==3.7 pymdown-extensions==10.12
"""
import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"

# Sidebar order. Pages not listed here are still built, just not in the nav.
NAV = [
    ("", [("README.md", "首页")]),
    ("信号建模", [
        ("docs/signal-model/modeling-guide.md", "建模总览：输入、输出、改进路线"),
        ("docs/signal-model/features-and-targets.md", "变量与预测目标"),
        ("docs/signal-model/ml-pipeline.md", "机器学习路线与 pipeline"),
        ("docs/signal-model/deep-learning.md", "深度学习：网络设计"),
        ("docs/signal-model/reinforcement-learning.md", "强化学习与模拟器"),
        ("docs/signal-model/repos.md", "开源仓库清单"),
        ("docs/signal-model/execution-signal-plan.md", "执行信号方案"),
        ("docs/signal-model/obi-ofi-basics.md", "OBI / OFI 基础"),
        ("docs/signal-model/pooled-model.md", "通用模型"),
    ]),
    ("Agent 平台", [
        ("docs/agent-platform/architecture.md", "选型与架构"),
        ("docs/agent-platform/skill-chain-backend.md", "Skill 链后台"),
        ("docs/agent-platform/deployment.md", "部署方案"),
    ]),
]

CSS = """
:root{--bg:#fbfbfa;--panel:#f2f1ee;--ink:#1d1d1b;--muted:#5f5e5a;--line:#dddbd5;--accent:#2f5d8a;--code:#efede8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17181a;--panel:#1f2023;--ink:#e8e6e1;--muted:#a3a19b;--line:#33353a;--accent:#8db4dc;--code:#24262a}}
:root[data-theme="dark"]{--bg:#17181a;--panel:#1f2023;--ink:#e8e6e1;--muted:#a3a19b;--line:#33353a;--accent:#8db4dc;--code:#24262a}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Microsoft YaHei",sans-serif}
a{color:var(--accent)}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:100vh}
nav.side{background:var(--panel);border-right:1px solid var(--line);padding:20px 16px;position:sticky;top:0;height:100vh;overflow:auto;font-size:14.5px}
nav.side .brand{font-weight:700;font-size:16px;margin-bottom:12px;display:block;color:var(--ink);text-decoration:none}
nav.side h4{margin:18px 0 6px;font-size:12px;letter-spacing:.06em;color:var(--muted);text-transform:uppercase}
nav.side a{display:block;padding:4px 8px;border-radius:6px;text-decoration:none;color:var(--ink)}
nav.side a.on{background:var(--bg);color:var(--accent);font-weight:600}
main{padding:32px 48px 80px;max-width:980px;width:100%}
h1{font-size:1.75rem;line-height:1.3;margin:.2rem 0 1rem}
h2{font-size:1.3rem;margin:2.2rem 0 .8rem;padding-top:.6rem;border-top:1px solid var(--line)}
h3{font-size:1.08rem;margin:1.6rem 0 .5rem}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:1rem 0}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--panel)}
code{background:var(--code);padding:.1em .35em;border-radius:4px;font-size:.9em;overflow-wrap:anywhere}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto;font-size:13.5px;line-height:1.55}
pre code{background:none;padding:0;overflow-wrap:normal}
pre.mermaid{background:var(--panel);text-align:center}
blockquote{margin:1rem 0;padding:.4rem 1rem;border-left:3px solid var(--accent);background:var(--panel);color:var(--ink)}
.toggle{position:fixed;top:12px;right:14px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:4px 10px;font-size:13px;cursor:pointer}
footer{margin-top:3rem;color:var(--muted);font-size:13px}
@media (max-width:860px){
  .layout{display:block}
  nav.side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
  main{padding:20px 16px 60px}
  h1{font-size:1.45rem}
}
"""

JS = """
(function(){var k='theme',r=document.documentElement;try{var t=localStorage.getItem(k);if(t)r.setAttribute('data-theme',t)}catch(e){}
document.addEventListener('DOMContentLoaded',function(){var b=document.querySelector('.toggle');if(!b)return;b.onclick=function(){
var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);
var n=dark?'light':'dark';r.setAttribute('data-theme',n);try{localStorage.setItem(k,n)}catch(e){}location.reload()}})})();
"""

MERMAID = """<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
const r=document.documentElement,dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);
mermaid.initialize({startOnLoad:true,theme:dark?'dark':'neutral'});
</script>"""


def out_path(src: str) -> str:
    return "index.html" if src == "README.md" else src[:-3] + ".html"


def rel(from_out: str, to_out: str) -> str:
    depth = from_out.count("/")
    return "../" * depth + to_out


def render(src: str) -> tuple[str, str]:
    text = (ROOT / src).read_text(encoding="utf8")
    md = markdown.Markdown(extensions=[
        "tables", "toc", "sane_lists",
        "pymdownx.superfences",
    ], extension_configs={"pymdownx.superfences": {"custom_fences": [
        {"name": "mermaid", "class": "mermaid", "format": lambda s, *a, **k: f'<pre class="mermaid">{html.escape(s)}</pre>'},
    ]}})
    body = md.convert(text)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub("<[^>]+>", "", m.group(1)) if m else src
    # links between markdown files -> html pages, relative to the source file's folder
    here = Path(src).parent

    def fix(mo):
        href = mo.group(1)
        if re.match(r"^[a-z]+:", href) or href.startswith("#"):
            return mo.group(0)
        path, _, frag = href.partition("#")
        if not path.endswith(".md"):
            return mo.group(0)
        target = (here / path).as_posix()
        target = str(Path(target)).replace("\\", "/")
        parts = []
        for p in target.split("/"):
            if p == "..":
                parts.pop()
            elif p != ".":
                parts.append(p)
        target = "/".join(parts)
        new = rel(out_path(src), out_path(target)) + (f"#{frag}" if frag else "")
        return f'href="{new}"'

    body = re.sub(r'href="([^"]+)"', fix, body)
    body = re.sub(r"(<table>.*?</table>)", r'<div class="tw">\1</div>', body, flags=re.S)
    return title, body


def nav_html(cur: str) -> str:
    me = out_path(cur)
    parts = [f'<a class="brand" href="{rel(me, "index.html")}">deploy1020 · 技术笔记</a>']
    for group, items in NAV:
        if group:
            parts.append(f"<h4>{group}</h4>")
        for src, label in items:
            if not (ROOT / src).exists():
                continue
            cls = ' class="on"' if src == cur else ""
            parts.append(f'<a{cls} href="{rel(me, out_path(src))}">{label}</a>')
    return "\n".join(parts)


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    sources = ["README.md"] + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").rglob("*.md"))
    for src in sources:
        title, body = render(src)
        has_mermaid = 'class="mermaid"' in body
        page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<script>{JS}</script>
</head>
<body>
<button class="toggle" type="button" aria-label="切换深浅色">◐</button>
<div class="layout">
<nav class="side">
{nav_html(src)}
</nav>
<main>
{body}
<footer>源文件：<code>{src}</code></footer>
</main>
</div>
{MERMAID if has_mermaid else ""}
</body>
</html>
"""
        dst = OUT / out_path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(page, encoding="utf8")
    (OUT / ".nojekyll").write_text("")
    (OUT / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    print(f"built {len(sources)} pages into {OUT}")


if __name__ == "__main__":
    build()
