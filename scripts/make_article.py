"""Render docs/HF_ARTICLE.md as a standalone HTML article and publish it as a static Hugging Face Space.

  build                               write hf/article/{index.html,README.md} (offline, standard library)
  publish --confirm-owner-approved    upload hf/article/ to <namespace>/aawden-article (network)

The markdown converter covers exactly the subset the article uses: front matter, headings, paragraphs,
bullet and numbered lists, pipe tables with alignment, fenced code, block quotes, horizontal rules,
images (an italic paragraph right after an image becomes its caption), and inline bold, italic, code
and links. Anything else fails loudly rather than rendering wrong.

Figures are the README SVGs from docs/img/, inlined in both their light and dark versions, so the page
is one file and follows the reader's colour scheme.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE = ROOT / "docs" / "HF_ARTICLE.md"
OUT = ROOT / "hf" / "article"
SPACE_ID = "aawden-article"

FIGURES = {  # image URL basename in the markdown -> docs/img stem
    "fig-measure.png": "fig-measure",
    "fig-flip.png": "fig-flip",
    "fig-recipe.png": "fig-recipe",
}

LINKS = [
    ("Live demo", "https://rishyanthreddy-aawden-demo.static.hf.space"),
    ("Adapter", "https://huggingface.co/RishyanthReddy/aawden-aya8b-reviewer-adapter"),
    ("Benchmark", "https://huggingface.co/datasets/RishyanthReddy/aawden-reviewer-flip-bench"),
    ("Code", "https://github.com/Aya-Jmd/Aawden"),
]


# ---------------------------------------------------------------- markdown subset

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def inline(text: str) -> str:
    codes: list[str] = []

    def keep(match: re.Match) -> str:
        codes.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
                  lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)
    if "**" in text or "](" in text:
        raise ValueError(f"unconverted inline markup: {text[:80]!r}")
    return text


def front_matter(lines: list[str]) -> tuple[dict, list[str]]:
    if lines[0].strip() != "---":
        return {}, lines
    end = lines.index("---", 1)
    meta = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep and not line.startswith(" "):
            meta[key.strip()] = value.strip().strip('"')
    return meta, lines[end + 1:]


def table(rows: list[str]) -> str:
    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    head, rule, *body = rows
    aligns = []
    for spec in cells(rule):
        if not re.fullmatch(r":?-+:?", spec):
            raise ValueError(f"bad table rule: {rule!r}")
        aligns.append("c" if spec.startswith(":") and spec.endswith(":") else "r" if spec.endswith(":") else "")

    def cls(i: int) -> str:
        return f' class="{aligns[i]}"' if i < len(aligns) and aligns[i] else ""

    out = ['<div class="tbl"><table><thead><tr>']
    out += [f"<th{cls(i)}>{inline(c)}</th>" for i, c in enumerate(cells(head))]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{inline(c)}</td>" for i, c in enumerate(cells(row))) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def figure(stem: str, alt: str, caption: str | None) -> str:
    parts = []
    for variant, css in ((stem, "light"), (f"{stem}-dark", "dark")):
        svg = (ROOT / "docs" / "img" / f"{variant}.svg").read_text(encoding="utf-8")
        svg = re.sub(r"<\?xml[^>]*>\s*", "", svg)
        svg = svg.replace('id="ttl"', f'id="{variant}-ttl"').replace('id="dsc"', f'id="{variant}-dsc"')
        svg = svg.replace('aria-labelledby="ttl dsc"', f'aria-labelledby="{variant}-ttl {variant}-dsc"')
        svg = re.sub(r'(<svg\b[^>]*?)\swidth="\d+"\sheight="\d+"', r"\1", svg, count=1)
        parts.append(f'<div class="fig-{css}">{svg}</div>')
    cap = f"<figcaption>{inline(caption)}</figcaption>" if caption else ""
    return f'<figure aria-label="{html.escape(alt)}">{"".join(parts)}{cap}</figure>'


def convert(body: list[str]) -> tuple[str, list[tuple[int, str, str]]]:
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    i, n = 0, len(body)
    while i < n:
        line = body[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            j = i + 1
            while not body[j].strip().startswith("```"):
                j += 1
            code = html.escape("\n".join(body[i + 1:j]), quote=False)
            out.append(f'<pre><code class="lang-{lang}">{code}</code></pre>' if lang else f"<pre><code>{code}</code></pre>")
            i = j + 1
            continue
        heading = re.match(r"(#{2,4}) (.+)", stripped)
        if heading:
            level, text = len(heading.group(1)), heading.group(2)
            anchor = slug(text)
            toc.append((level, anchor, text))
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            i += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            name = image.group(2).rsplit("/", 1)[-1]
            if name not in FIGURES:
                raise ValueError(f"no inline figure for {name}")
            j = i + 1
            while j < n and not body[j].strip():
                j += 1
            caption = None
            if j < n and body[j].startswith("*") and not body[j].startswith("**"):
                k = j
                while k < n and body[k].strip():
                    k += 1
                para = " ".join(l.strip() for l in body[j:k])
                if para.startswith("*") and para.endswith("*"):
                    caption, j = para[1:-1], k
            out.append(figure(FIGURES[name], image.group(1), caption))
            i = j if caption else i + 1
            continue
        if stripped.startswith("|"):
            j = i
            while j < n and body[j].strip().startswith("|"):
                j += 1
            out.append(table(body[i:j]))
            i = j
            continue
        if stripped.startswith(">"):
            j = i
            while j < n and body[j].strip().startswith(">"):
                j += 1
            text = " ".join(l.strip()[1:].strip() for l in body[i:j])
            out.append(f"<blockquote><p>{inline(text)}</p></blockquote>")
            i = j
            continue
        bullet = re.match(r"(- |\d+\. )", stripped)
        if bullet:
            ordered = stripped[0].isdigit()
            items: list[str] = []
            j = i
            while j < n and body[j].strip():
                if re.match(r"(- |\d+\. )", body[j]):
                    items.append(re.sub(r"^(- |\d+\. )", "", body[j]).strip())
                elif body[j].startswith("  "):
                    items[-1] += " " + body[j].strip()
                else:
                    break
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
            i = j
            continue
        j = i
        while j < n and body[j].strip() and not re.match(r"(```|\||>|#{2,4} |- |\d+\. |!\[|---$)", body[j].strip()):
            j += 1
        if j == i:
            raise ValueError(f"unhandled line {i}: {line!r}")
        out.append(f"<p>{inline(' '.join(l.strip() for l in body[i:j]))}</p>")
        i = j
    return "\n".join(out), toc


# ---------------------------------------------------------------- page

CSS = """
:root{--bg:#fff;--ink:#16181d;--ink2:#555b66;--ink3:#7a808c;--line:#e3e5ea;--soft:#f5f6f8;--accent:#2a78d6;
--warn:#eb6834;--good:#12784f;--chip:#eef4fc}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--ink:#eceef2;--ink2:#a3a9b5;--ink3:#868c98;--line:#2a2e36;
--soft:#1a1d22;--accent:#5b9ff0;--warn:#e8794c;--good:#4cc793;--chip:#18233a}}
*{box-sizing:border-box}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
font:17.5px/1.68 "Source Serif 4",Charter,Georgia,serif;font-optical-sizing:auto}
.sans,h1,h2,h3,h4,nav,th,td,figcaption,.kicker,.byline,.links,footer{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px;display:grid;grid-template-columns:minmax(0,1fr);}
@media(min-width:1120px){.wrap{grid-template-columns:220px minmax(0,720px);gap:64px;justify-content:center}}
header.hero{max-width:720px;margin:0 auto;padding:64px 16px 8px}
@media(min-width:1120px){header.hero{max-width:1004px;padding-left:300px}}
.kicker{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink2);margin-bottom:16px;font-weight:600}
h1{font-size:44px;line-height:1.1;letter-spacing:-.025em;margin:0 0 18px;font-weight:700;text-wrap:balance}
.dek{font-size:20px;line-height:1.5;color:var(--ink2);margin:0 0 22px;font-style:italic}
.byline{font-size:14.5px;color:var(--ink2);margin:0 0 18px}.byline strong{color:var(--ink);font-weight:600}
.links{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 8px}
.links a{font-size:14px;font-weight:600;text-decoration:none;color:var(--accent);background:var(--chip);
padding:7px 13px;border-radius:999px;border:1px solid transparent}
.links a:hover{border-color:var(--accent)}
nav.toc{display:none}
@media(min-width:1120px){nav.toc{display:block;position:sticky;top:28px;align-self:start;margin-top:44px;
max-height:calc(100vh - 56px);overflow:auto;font-size:13.5px;line-height:1.45}}
nav.toc .t{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);font-weight:600;margin-bottom:10px}
nav.toc a{display:block;color:var(--ink2);text-decoration:none;padding:4px 0 4px 12px;border-left:2px solid var(--line)}
nav.toc a.l3{padding-left:24px;font-size:12.5px;color:var(--ink3)}
nav.toc a:hover,nav.toc a.on{color:var(--ink);border-left-color:var(--accent)}
article{min-width:0;padding:20px 0 40px}
article>p:first-child{font-size:15px;color:var(--ink2);border-left:3px solid var(--line);padding-left:14px;margin-top:20px}
h2{font-size:29px;line-height:1.2;letter-spacing:-.015em;margin:64px 0 14px;font-weight:700;text-wrap:balance}
h3{font-size:21px;line-height:1.3;margin:42px 0 10px;font-weight:650}
h2,h3{scroll-margin-top:20px}
p{margin:0 0 18px}
a{color:var(--accent);text-underline-offset:2px}
strong{font-weight:700}
code{font:0.84em/1.4 "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--soft);
padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:16px 18px;overflow:auto;margin:0 0 20px}
pre code{background:none;padding:0;font-size:14px;line-height:1.55;overflow-wrap:normal}
blockquote{margin:0 0 20px;padding:14px 18px;border-left:4px solid var(--warn);background:var(--soft);border-radius:0 10px 10px 0}
blockquote p{margin:0;font:15px/1.5 "JetBrains Mono",ui-monospace,Menlo,monospace}
blockquote strong{font-weight:600}
hr{border:0;height:1px;background:var(--line);margin:56px 0}
ul,ol{padding-left:22px;margin:0 0 20px}li{margin:7px 0}
.tbl{overflow-x:auto;margin:6px 0 24px;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14.5px;line-height:1.4;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink2);font-size:12.5px;letter-spacing:.02em;
border-bottom:1.5px solid var(--ink3);padding:8px 14px 8px 0;white-space:nowrap}
td{padding:8px 14px 8px 0;border-bottom:1px solid var(--line);vertical-align:top}
th:last-child,td:last-child{padding-right:0}
td:first-child{min-width:9.5em}
.r{text-align:right;white-space:nowrap}.c{text-align:center}
td code{font-size:.9em}
figure{margin:28px 0 30px}
figure svg{display:block;width:100%;height:auto;border-radius:10px}
.fig-dark{display:none}
@media (prefers-color-scheme:dark){.fig-light{display:none}.fig-dark{display:block}}
figcaption{font-size:14px;line-height:1.55;color:var(--ink2);margin-top:10px}
footer{border-top:1px solid var(--line);margin-top:40px;padding:22px 0 60px;font-size:13.5px;color:var(--ink2)}
@media(max-width:640px){body{font-size:17px}h1{font-size:32px}.dek{font-size:18px}h2{font-size:24px;margin-top:52px}
h3{font-size:19px}header.hero{padding-top:40px}table{font-size:13.5px}th{white-space:normal}.fig-light,.fig-dark{overflow-x:auto}figure svg{min-width:620px}pre code{font-size:12.5px}}
"""

JS = """
(function(){var links=[].slice.call(document.querySelectorAll('nav.toc a'));if(!links.length||!('IntersectionObserver' in window))return;
var map={};links.forEach(function(a){map[a.getAttribute('href').slice(1)]=a});
var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){links.forEach(function(a){a.classList.remove('on')});
var a=map[e.target.id];if(a)a.classList.add('on')}})},{rootMargin:'0px 0px -75% 0px'});
document.querySelectorAll('article h2[id],article h3[id]').forEach(function(h){io.observe(h)})})();
"""


def build(source: Path = SOURCE, out: Path = OUT) -> Path:
    meta, body = front_matter(source.read_text(encoding="utf-8").splitlines())
    title = meta["title"]
    article, toc = convert(body)
    nav = "".join(f'<a class="l{lvl}" href="#{a}">{inline(t)}</a>' for lvl, a, t in toc)
    links = "".join(f'<a href="{html.escape(u)}">{html.escape(t)}</a>' for t, u in LINKS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aawden — {html.escape(title)}</title>
<meta name="description" content="On files a model gets right alone, one wrong reviewer sentence takes most of those answers away. How we measured it, and a small adapter that stops it.">
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<header class="hero">
<div class="kicker">Aawden · Lexsi AI Safety hackathon · Track 2</div>
<h1>{html.escape(title)}</h1>
<p class="dek">We measured how often one confident sentence overturns a correct security verdict, found where that measurement can be trusted, and trained a small adapter that stops it.</p>
<p class="byline">A project by <strong><a href="https://github.com/Aya-Jmd">Aya-Jmd</a></strong> and <strong><a href="https://huggingface.co/RishyanthReddy">M. Rishyanth Reddy</a></strong> · November 2026</p>
<div class="links">{links}</div>
</header>
<div class="wrap">
<nav class="toc" aria-label="Contents"><div class="t">Contents</div>{nav}</nav>
<article>
{article}
<footer>Every number on this page comes from a saved result file named next to the claim. Source of this
page: <a href="https://github.com/Aya-Jmd/Aawden/blob/master/docs/HF_ARTICLE.md"><code>docs/HF_ARTICLE.md</code></a>,
rendered by <code>scripts/make_article.py</code>.</footer>
</article></div>
<script>{JS}</script>
</body></html>
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "README.md").write_text(f"""---
title: "Aawden: one sentence talks an AI code reviewer out of the right answer"
emoji: "\\U0001F6E1"
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: "Article: when a wrong reviewer flips a correct verdict"
---

The full write-up of Aawden as a single page: how deference to a wrong code reviewer was measured,
why most models under 10B cannot be measured this way, and the adapter that stops it.

Source: [`docs/HF_ARTICLE.md`](https://github.com/Aya-Jmd/Aawden/blob/master/docs/HF_ARTICLE.md),
rendered by `scripts/make_article.py`.
""", encoding="utf-8")
    return out


def publish(out: Path = OUT) -> str:
    from hf_publish import env_secret_values, load_config, read_env, scan_bytes

    env = read_env(ROOT / ".env")
    token = env.get("HF_PUBLISH_TOKEN", "")
    if not token:
        raise SystemExit("HF_PUBLISH_TOKEN is not set in .env")
    secrets = env_secret_values(env)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    for path in files:
        hits = scan_bytes(path.read_bytes(), secrets)
        if hits:
            raise SystemExit(f"refusing to upload {path.name}: {', '.join(hits)}")
    ns = load_config()["namespace"]
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if api.whoami().get("name") != ns:
        raise SystemExit("token does not resolve to the configured namespace")
    repo_id = f"{ns}/{SPACE_ID}"
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(out),
                      commit_message="Aawden article: render docs/HF_ARTICLE.md")
    return repo_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    p_pub = sub.add_parser("publish")
    p_pub.add_argument("--confirm-owner-approved", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build() / "index.html")
        return 0
    if not args.confirm_owner_approved:
        raise SystemExit("publish refused: pass --confirm-owner-approved after the owner approves")
    print(publish())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
