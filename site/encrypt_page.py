"""Encrypt a Markdown page so it can be published on the public site behind a password.

    python3 site/encrypt_page.py path/to/note.md 1021 --nav "1021 讨论记录"
    # asks for the password (or reads PAGE_PASSWORD); writes site/encrypted/1021.json

Only the ciphertext goes into the repository. Keep the Markdown source and the password OUT of the repo
(the source path should be outside the working tree or git-ignored). site/build.py turns the JSON into
notes/<slug>.html, where the browser derives the key from the password and decrypts the page.

Crypto: key = PBKDF2-HMAC-SHA256(password, 16-byte random salt, 600 000 iterations), 32 bytes;
AES-256-GCM with a 12-byte random IV; the GCM tag (appended to the ciphertext) makes a wrong password fail
cleanly. Strength depends entirely on the password: anyone can download the JSON and guess offline, so
a short password only slows a determined attacker down. Needs `pip install cryptography`.
"""
import argparse
import base64
import getpass
import html
import json
import re
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402  (reuse the site's Markdown renderer)

ITERATIONS = 600_000


def encrypt(plaintext: str, password: str) -> dict:
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, plaintext.encode("utf8"), None)
    b = lambda x: base64.b64encode(x).decode()
    return {"salt": b(salt), "iv": b(iv), "iter": ITERATIONS, "ct": b(ct)}


def decrypt(d: dict, password: str) -> str:
    """Inverse of encrypt (used to check a file round-trips)."""
    u = lambda x: base64.b64decode(x)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=u(d["salt"]), iterations=d["iter"]).derive(password.encode())
    return AESGCM(key).decrypt(u(d["iv"]), u(d["ct"]), None).decode("utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Markdown file (keep it outside the repository)")
    ap.add_argument("slug", help="output name: site/encrypted/<slug>.json -> notes/<slug>.html")
    ap.add_argument("--nav", help="sidebar label (default: the slug)")
    ap.add_argument("--title", help="public title shown before unlocking (default: '<slug> · 笔记');"
                    " the page's own heading stays inside the encrypted part")
    a = ap.parse_args()
    pw = os.environ.get("PAGE_PASSWORD") or getpass.getpass("password: ")
    text = Path(a.source).read_text(encoding="utf8")
    _, body = build.render_text(text, f"notes/{a.slug}.md")
    # A stray "<" inside display math can be parsed as an HTML tag and swallow the rest of the page:
    # make sure every "## " heading of the source made it into the HTML.
    lost = [h for h in re.findall(r"^## (.+)$", text, re.M) if html.escape(h.strip(), quote=False) not in body]
    if lost:
        sys.exit(f"these sections did not render (write '<' as \\lt in math): {lost}")
    d = {"title": a.title or f"{a.slug} · 笔记", "nav": a.nav or a.slug, **encrypt(body, pw)}
    assert decrypt(d, pw) == body
    out = build.ENC_DIR / f"{a.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"wrote {out.relative_to(build.ROOT)} ({len(body)} chars of HTML encrypted)")


if __name__ == "__main__":
    main()
