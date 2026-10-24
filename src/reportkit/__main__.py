"""Command line.

    python -m reportkit example out.html           # build the example report (synthetic data, needs numpy)
    python -m reportkit template out.html          # copy the hand-written template, assets inlined
    python -m reportkit inline in.html out.html    # inline theme.css / theme.js into a hand-written report
    python -m reportkit assets DIR                 # copy theme.css, theme.js, template.html, fonts/ into DIR
"""
import argparse
import shutil
import sys
from pathlib import Path

from .report import ASSETS, inline_assets


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m reportkit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("example"); e.add_argument("out"); e.add_argument("--generated")
    t = sub.add_parser("template"); t.add_argument("out")
    i = sub.add_parser("inline"); i.add_argument("src"); i.add_argument("out")
    a = sub.add_parser("assets"); a.add_argument("dir")
    args = ap.parse_args(argv)
    if args.cmd == "example":
        from .example import build
        build(args.out, generated=args.generated)
    elif args.cmd == "template":
        Path(args.out).write_text(inline_assets((ASSETS / "template.html").read_text(encoding="utf8")), encoding="utf8")
    elif args.cmd == "inline":
        Path(args.out).write_text(inline_assets(Path(args.src).read_text(encoding="utf8")), encoding="utf8")
    elif args.cmd == "assets":
        d = Path(args.dir)
        d.mkdir(parents=True, exist_ok=True)
        for f in ("theme.css", "theme.js", "template.html"):
            shutil.copy(ASSETS / f, d / f)
        shutil.copytree(ASSETS / "fonts", d / "fonts", dirs_exist_ok=True)
    print(f"wrote {getattr(args, 'out', None) or args.dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
