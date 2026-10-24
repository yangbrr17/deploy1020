"""Render README.md and docs/**/*.md into a static site under _site/.

python3 site/build.py            # needs: pip install markdown==3.7 pymdown-extensions==10.12

Math: write $...$ inline and $$...$$ on their own lines; KaTeX renders them in the browser.
Every code block gets a copy button; every display formula gets a "copy LaTeX" button.
Password-protected pages: site/encrypted/<slug>.json (made by site/encrypt_page.py from a Markdown file
that is NOT committed) become notes/<slug>.html; the browser decrypts them with the password.
Python files under src/ become source pages (foo.py -> foo.py.html) with a table of their top-level
functions and classes; Markdown links to a .py file are rewritten to that page.
Report theme: report/example.html is built by src/reportkit (needs numpy) and report/ also gets theme.css,
theme.js, template.html and fonts/; NAV entries starting with "@" link to such built files directly.
"""
import ast
import html
import json
import re
import shutil
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"

# Sidebar order: (source, label, indent level). Pages not listed here are still built, just not in the nav.
NAV = [
    ("", [("README.md", "首页", 0)]),
    ("交付报告", [
        ("docs/report-theme/README.md", "报告主题：规范与用法", 0),
        ("@report/example.html", "示例报告 ↗", 1),
        ("@report/template.html", "空白模板 ↗", 1),
    ]),
    ("信号建模", [
        ("docs/signal-model/modeling-guide.md", "建模总览：输入、输出、改进路线", 0),
        ("docs/signal-model/features-and-targets.md", "变量与预测目标", 0),
        ("docs/signal-model/event-arrivals.md", "事件到达：下一笔什么时候来", 0),
        ("docs/signal-model/arrival-tests.md", "到达分布检验：数据与方法", 1),
        ("docs/signal-model/ml-pipeline.md", "机器学习路线与 pipeline", 0),
        ("docs/signal-model/deep-learning.md", "深度学习：总览与训练", 0),
        ("docs/signal-model/dl/deeplob.md", "DeepLOB", 1),
        ("docs/signal-model/dl/bin-ctabl.md", "BiN-CTABL", 1),
        ("docs/signal-model/dl/deep-ofi.md", "Deep OFI", 1),
        ("docs/signal-model/dl/tlob.md", "TLOB", 1),
        ("docs/signal-model/dl/ofi-tcn-gru.md", "推荐：OFI-TCN-GRU", 1),
        ("docs/signal-model/reinforcement-learning.md", "强化学习：建模与选型", 0),
        ("docs/signal-model/execution-signal-plan.md", "执行层：信号怎么用", 0),
    ]),
    ("代码", [
        ("src/README.md", "代码包说明（arrivals、execrl）", 0),
        (".github/copilot-instructions.md", "交接说明与待办", 0),
    ]),
    ("Agent 平台", [
        ("docs/agent-platform/architecture.md", "选型与架构", 0),
        ("docs/agent-platform/skill-chain-backend.md", "Skill 链后台", 0),
        ("docs/agent-platform/deployment.md", "部署方案", 0),
    ]),
]

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

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
nav.side a.sub{padding-left:24px;font-size:13.5px;color:var(--muted)}
nav.side a.on{background:var(--bg);color:var(--accent);font-weight:600}
main{padding:32px 48px 80px;max-width:1000px;width:100%}
h1{font-size:1.75rem;line-height:1.3;margin:.2rem 0 1rem}
h2{font-size:1.3rem;margin:2.2rem 0 .8rem;padding-top:.6rem;border-top:1px solid var(--line)}
h3{font-size:1.08rem;margin:1.6rem 0 .5rem}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:1rem 0}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--panel)}
code{background:var(--code);padding:.1em .35em;border-radius:4px;font-size:.9em;overflow-wrap:anywhere}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto;font-size:13.5px;line-height:1.55;margin:0}
pre code{background:none;padding:0;overflow-wrap:normal}
pre.mermaid{background:var(--panel);text-align:center;margin:1rem 0}
.cb{position:relative;margin:1rem 0}
.cb .lang{position:absolute;top:6px;left:12px;font-size:11.5px;color:var(--muted);font-family:ui-monospace,monospace}
.cb.has-lang pre{padding-top:28px}
.copy{position:absolute;top:6px;right:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:1px 9px;font-size:12px;cursor:pointer;opacity:.85}
.copy:hover{opacity:1;border-color:var(--accent)}
.math{position:relative;margin:1rem 0;padding:6px 0;overflow-x:auto;overflow-y:hidden}
.math .copy{top:2px}
.math .katex-display{margin:.4rem 0;padding-right:84px}
blockquote{margin:1rem 0;padding:.4rem 1rem;border-left:3px solid var(--accent);background:var(--panel);color:var(--ink)}
.toggle{position:fixed;top:12px;right:14px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:4px 10px;font-size:13px;cursor:pointer;z-index:5}
footer{margin-top:3rem;color:var(--muted);font-size:13px}
.unlock{margin:1.2rem 0;padding:16px;border:1px solid var(--line);border-radius:8px;background:var(--panel);max-width:420px}
.unlock input{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);width:min(100%,240px)}
.unlock button{font:inherit;padding:6px 14px;border:1px solid var(--accent);border-radius:6px;background:var(--accent);color:var(--bg);cursor:pointer}
.muted{color:var(--muted)}
@media (max-width:860px){
  .layout{display:block}
  nav.side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
  main{padding:20px 16px 60px}
  h1{font-size:1.45rem}
  .math .katex-display{padding-right:0;padding-top:22px}
}
"""

THEME_JS = """
(function(){var k='theme',r=document.documentElement;try{var t=localStorage.getItem(k);if(t)r.setAttribute('data-theme',t)}catch(e){}})();
"""

PAGE_JS = r"""
(function(){
var r=document.documentElement;
function copyText(t,btn){
  function done(){var o=btn.textContent;btn.textContent='已复制';setTimeout(function(){btn.textContent=o},1200)}
  if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(t).then(done,fallback)}else{fallback()}
  function fallback(){var a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';document.body.appendChild(a);a.select();try{document.execCommand('copy');done()}catch(e){}document.body.removeChild(a)}
}
function button(label,get){var b=document.createElement('button');b.type='button';b.className='copy';b.textContent=label;b.onclick=function(){copyText(get(),b)};return b}
function enhance(root){
  root.querySelectorAll('pre > code').forEach(function(c){
    var pre=c.parentNode,w=document.createElement('div');w.className='cb';
    var m=(c.className||'').match(/language-([\w+-]+)/);
    pre.parentNode.insertBefore(w,pre);w.appendChild(pre);
    if(m){var l=document.createElement('span');l.className='lang';l.textContent=m[1];w.appendChild(l);w.classList.add('has-lang')}
    w.appendChild(button('复制',function(){return c.textContent}));
  });
  root.querySelectorAll('div.arithmatex').forEach(function(d){
    var src=d.textContent.trim().replace(/^\\\[/,'').replace(/\\\]$/,'').trim();
    d.classList.add('math');d.setAttribute('data-latex',src);
    d.appendChild(button('复制 LaTeX',function(){return src}));
  });
  if(window.renderMathInElement)renderMathInElement(root,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],throwOnError:false,ignoredClasses:['copy']});
}
window.enhancePage=enhance;
document.addEventListener('DOMContentLoaded',function(){
  var tg=document.querySelector('.toggle');
  if(tg)tg.onclick=function(){var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);var n=dark?'light':'dark';r.setAttribute('data-theme',n);try{localStorage.setItem('theme',n)}catch(e){}location.reload()};
  if(!document.querySelector('.locked'))enhance(document.querySelector('main'));
});
})();
"""

MERMAID = """<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
const r=document.documentElement,dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);
mermaid.initialize({startOnLoad:true,theme:dark?'dark':'neutral'});
</script>"""


def out_path(src: str) -> str:
    if src == "README.md":
        return "index.html"
    if src.startswith(".github/"):            # Pages artifacts drop dot-directories
        src = "meta/" + src[len(".github/"):]
    if src.endswith(".py"):
        return src + ".html"
    return src[:-3] + ".html"


def rel(from_out: str, to_out: str) -> str:
    return "../" * from_out.count("/") + to_out


def render(src: str) -> tuple[str, str]:
    return render_text((ROOT / src).read_text(encoding="utf8"), src)


def render_text(text: str, src: str) -> tuple[str, str]:
    """Markdown -> (title, body HTML). `src` is the page's repo path, used to rewrite relative links."""
    md = markdown.Markdown(extensions=[
        "tables", "toc", "sane_lists",
        "pymdownx.arithmatex", "pymdownx.superfences",
    ], extension_configs={
        "pymdownx.arithmatex": {"generic": True},
        "pymdownx.superfences": {"custom_fences": [
            {"name": "mermaid", "class": "mermaid",
             "format": lambda s, *a, **k: f'<pre class="mermaid">{html.escape(s)}</pre>'},
        ]},
    })
    body = md.convert(text)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub("<[^>]+>", "", m.group(1)) if m else src
    here = Path(src).parent

    def fix(mo):
        href = mo.group(1)
        if re.match(r"^[a-z]+:", href) or href.startswith("#"):
            return mo.group(0)
        path, _, frag = href.partition("#")
        if not path.endswith((".md", ".py")):
            return mo.group(0)
        parts = []
        for p in (here / path).as_posix().split("/"):
            if p == "..":
                parts.pop()
            elif p != ".":
                parts.append(p)
        new = rel(out_path(src), out_path("/".join(parts))) + (f"#{frag}" if frag else "")
        return f'href="{new}"'

    body = re.sub(r'href="([^"]+)"', fix, body)
    body = re.sub(r"(<table>.*?</table>)", r'<div class="tw">\1</div>', body, flags=re.S)
    return title, body


def render_py(src: str) -> tuple[str, str]:
    """Source page for a Python file: module docstring, an index of top-level defs, then the code."""
    code = (ROOT / src).read_text(encoding="utf8")
    tree = ast.parse(code)
    doc = ast.get_docstring(tree) or ""
    rows = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            first = (ast.get_docstring(node) or "").strip().split("\n")[0]
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            rows.append(f"<tr><td><code>{kind} {html.escape(node.name)}</code></td>"
                        f"<td>{node.lineno}</td><td>{html.escape(first)}</td></tr>")
    index = ("<div class=\"tw\"><table><tr><th>定义</th><th>行</th><th>说明（docstring 第一行）</th></tr>"
             + "".join(rows) + "</table></div>") if rows else ""
    body = (f"<h1>{html.escape(src)}</h1>"
            + (f"<pre><code class=\"language-text\">{html.escape(doc)}</code></pre>" if doc else "")
            + index
            + f"<h2>源码（{len(code.splitlines())} 行）</h2>"
            + f"<pre><code class=\"language-python\">{html.escape(code)}</code></pre>")
    return src, body


def nav_html(cur: str) -> str:
    me = out_path(cur)
    parts = [f'<a class="brand" href="{rel(me, "index.html")}">deploy1020 · 技术笔记</a>']
    for group, items in NAV:
        if group:
            parts.append(f"<h4>{group}</h4>")
        for src, label, level in items:
            if src.startswith("@"):                  # a built file, not a Markdown source
                parts.append(f'<a class="{"sub" if level else ""}" href="{rel(me, src[1:])}">{label}</a>')
                continue
            if not (ROOT / src).exists():
                continue
            cls = " ".join(c for c in ("sub" if level else "", "on" if src == cur else "") if c)
            cls = f' class="{cls}"' if cls else ""
            parts.append(f'<a{cls} href="{rel(me, out_path(src))}">{label}</a>')
    locked = encrypted_pages()
    if locked:
        parts.append("<h4>笔记（需密码）</h4>")
        for d in locked:
            on = ' class="on"' if d["src"] == cur else ""
            parts.append(f'<a{on} href="{rel(me, out_path(d["src"]))}">{html.escape(d["nav"])}</a>')
    return "\n".join(parts)


LOCKED_JS = r"""
(function(){
function b64(s){var b=atob(s),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u}
async function unlock(pw){
  var d=window.LOCKED,enc=new TextEncoder();
  var base=await crypto.subtle.importKey('raw',enc.encode(pw),'PBKDF2',false,['deriveKey']);
  var key=await crypto.subtle.deriveKey({name:'PBKDF2',salt:b64(d.salt),iterations:d.iter,hash:'SHA-256'},base,{name:'AES-GCM',length:256},false,['decrypt']);
  var pt=await crypto.subtle.decrypt({name:'AES-GCM',iv:b64(d.iv)},key,b64(d.ct));
  return new TextDecoder().decode(pt);
}
document.addEventListener('DOMContentLoaded',function(){
  var f=document.getElementById('unlock'),msg=document.getElementById('unlock-msg'),box=document.getElementById('content');
  if(!window.isSecureContext||!crypto.subtle){msg.textContent='需要 https 才能解密。';return}
  f.onsubmit=function(e){
    e.preventDefault();msg.textContent='解密中…';
    unlock(f.pw.value).then(function(h){
      box.innerHTML=h;f.parentNode.removeChild(f);msg.textContent='';var t=document.getElementById('locked-title');if(t)t.parentNode.removeChild(t);
      document.querySelector('main').classList.remove('locked');
      if(window.enhancePage)window.enhancePage(box);
    },function(){msg.textContent='密码不对，或者页面数据损坏。'});
  };
});
})();
"""

ENC_DIR = ROOT / "site" / "encrypted"


def encrypted_pages() -> list[dict]:
    """Password-protected pages: site/encrypted/<slug>.json made by site/encrypt_page.py."""
    out = []
    for p in sorted(ENC_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf8"))
        d["src"] = f"notes/{p.stem}.md"            # virtual source path -> notes/<slug>.html
        out.append(d)
    return out


def page_html(src, title, body, has_math, has_mermaid, footer, extra_js=""):
    math_head = (f'<link rel="stylesheet" href="{KATEX}/katex.min.css">\n'
                 f'<script defer src="{KATEX}/katex.min.js"></script>\n'
                 f'<script defer src="{KATEX}/contrib/auto-render.min.js"></script>\n') if has_math else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<script>{THEME_JS}</script>
{math_head}<script defer>{PAGE_JS}</script>{extra_js}
</head>
<body>
<button class="toggle" type="button" aria-label="切换深浅色">◐</button>
<div class="layout">
<nav class="side">
{nav_html(src)}
</nav>
<main{' class="locked"' if extra_js else ''}>
{body}
<footer>{footer}</footer>
</main>
</div>
{MERMAID if has_mermaid else ""}
</body>
</html>
"""


BACK = '<a class="rk-back rk-hide-s" href="../docs/report-theme/README.html">← 规范与用法</a><span class="rk-sep rk-hide-s"></span>'


def build_reports():
    """report/: the example report (synthetic data), the blank template and the theme files."""
    sys.path.insert(0, str(ROOT / "src"))
    from reportkit.example import build as build_example
    from reportkit.report import ASSETS
    out = OUT / "report"
    build_example(out / "example.html")
    page = (out / "example.html").read_text(encoding="utf8")
    (out / "example.html").write_text(page.replace('<div class="rk-mast-l">', '<div class="rk-mast-l">' + BACK, 1), encoding="utf8")
    for f in ("theme.css", "theme.js", "template.html"):
        shutil.copy(ASSETS / f, out / f)
    shutil.copytree(ASSETS / "fonts", out / "fonts")


def write(src, page):
    dst = OUT / out_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page, encoding="utf8")


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    sources = (["README.md", ".github/copilot-instructions.md"]
               + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").rglob("*.md"))
               + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.md"))
               + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")
                        if "__pycache__" not in p.parts))
    for src in sources:
        title, body = render_py(src) if src.endswith(".py") else render(src)
        write(src, page_html(src, title, body, 'class="arithmatex"' in body, 'class="mermaid"' in body,
                             f"源文件：<code>{src}</code>"))
    locked = encrypted_pages()
    for d in locked:
        payload = json.dumps({k: d[k] for k in ("salt", "iv", "iter", "ct")})
        body = (f"<h1 id=\"locked-title\">{html.escape(d['title'])}</h1>\n"
                '<form id="unlock" class="unlock"><p>这一页需要密码。</p>'
                '<input name="pw" type="password" autocomplete="current-password" placeholder="密码" aria-label="密码"> '
                '<button type="submit">打开</button></form>\n'
                '<p id="unlock-msg" class="muted"></p>\n<div id="content"></div>')
        extra = f"\n<script>window.LOCKED={payload};</script>\n<script>{LOCKED_JS}</script>"
        write(d["src"], page_html(d["src"], d["title"], body, True, False,
                                  "内容在浏览器里用密码解密（AES-GCM，PBKDF2-SHA256）", extra))
    build_reports()
    (OUT / ".nojekyll").write_text("")
    (OUT / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    print(f"built {len(sources) + len(locked)} pages into {OUT} ({len(locked)} password-protected)")


if __name__ == "__main__":
    build()
