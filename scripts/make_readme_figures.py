"""Build the three README figures as static SVG. Every number comes from runs/fix/round2-summary.json.

GitHub renders an SVG in a README but will not run a script, so the interactive HTML figures under
docs/figures/ cannot be embedded. These are the same data drawn as plain SVG: no web fonts, no
JavaScript, one file per panel, and a light and a dark variant that each sit on their own surface
colour rather than inheriting the page background.

  docs/img/fig-measure.svg  what "flip rate on solved cases" counts, as one bar of 129 files
  docs/img/fig-flip.svg     the headline: unedited against adapter, three competent models
  docs/img/fig-recipe.svg   Aya 8B: unchanged against the two-stage recipe, per reviewer wording

Run: python scripts/make_readme_figures.py
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

# Checked against its own surface, not the page. Same three hues in both themes, so a reader who
# switches theme keeps the colour mapping.
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0",
         "axis": "#c9c8c2", "c1": "#2a78d6", "c2": "#eb6834", "c3": "#1baf7a"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#2e2e2c",
        "axis": "#4a4a46", "c1": "#3987e5", "c2": "#d95926", "c3": "#199e70"}

FONT = "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
TRAINED = {"reviewer", "authority", "scanner", "code_comment"}
WORDINGS = [("reviewer", "Reviewer, the trained wording"), ("authority", "Senior reviewer"),
            ("scanner", "Static analysis tool"), ("code_comment", "Comment left in the code"),
            ("ciso", "CISO sign-off"), ("consensus", "Three reviewers agree"),
            ("argued", "Reviewer gives a reason"), ("insistent", "Reviewer insists and threatens"),
            ("pushback", "Pushback on the model's own answer")]

W = 900


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def text_width(text: str, size: float) -> float:
    """Rough advance width in px, good enough to keep a label inside its box without a font query."""
    narrow, wide = set("iljI.,:;'|! "), set("MWmw@")
    units = sum(0.27 if ch in narrow else 0.86 if ch in wide else 0.56 for ch in text)
    return units * size


def wrap(text: str, size: float, limit: float) -> list[str]:
    """Greedy word wrap to a pixel limit. Keeps every subtitle line inside the frame."""
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and text_width(trial, size) > limit:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def frame(title: str, subtitle: str, c: dict, pad_bottom: int = 46) -> tuple[list[str], int, int]:
    """Title block shared by every panel. Returns (markup, height, first free y)."""
    sub_size, limit = 12.5, W - 4
    lines = wrap(subtitle, sub_size, limit)
    y = 42
    head = [f'<text x="0" y="18" fill="{c["ink"]}" font-family="{FONT}" font-size="15" '
            f'font-weight="600">{esc(title)}</text>']
    for line in lines:
        head.append(f'<text x="0" y="{y}" fill="{c["muted"]}" font-family="{FONT}" '
                    f'font-size="{sub_size}">{esc(line)}</text>')
        y += 18
    body_top = y + 12
    # Height is only known after the body is laid out, so the caller passes the body height back in
    # through the return of this function's companion below.
    return head, body_top, pad_bottom


def svg_open(title: str, subtitle: str, height: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img" aria-labelledby="ttl dsc">',
            f'<title id="ttl">{esc(title)}</title><desc id="dsc">{esc(subtitle)}</desc>']


def footnote(y: int, text: str, c: dict) -> str:
    """Source line, wrapped so it never leaves the frame."""
    out = []
    for line in wrap(text, 11.5, W - 4):
        out.append(f'<text x="0" y="{y}" fill="{c["muted"]}" font-family="{FONT}" '
                   f'font-size="11.5">{esc(line)}</text>')
        y += 15
    return "\n".join(out)


def legend(items: list[tuple[str, str, str]], x: int, y: int, c: dict) -> tuple[list[str], float]:
    """Swatch plus label, left to right. Identity is never carried by colour alone."""
    out, cursor = [], x
    for kind, label, colour in items:
        if kind == "bar":
            out.append(f'<rect x="{cursor:.1f}" y="{y - 9.5}" width="11" height="11" rx="2" fill="{colour}"/>')
        elif kind == "band":
            out.append(f'<rect x="{cursor:.1f}" y="{y - 9.5}" width="11" height="11" rx="2" '
                       f'fill="{c["grid"]}"/>')
        else:
            out.append(f'<circle cx="{cursor + 5.5:.1f}" cy="{y - 4}" r="5" fill="{colour}"/>')
        out.append(f'<text x="{cursor + 18:.1f}" y="{y}" fill="{c["muted"]}" font-family="{FONT}" '
                   f'font-size="12">{esc(label)}</text>')
        cursor += 18 + text_width(label, 12) + 26
    return out, cursor


# ---------------------------------------------------------------- figure 1: what is counted

def figure_measure(summary: dict, c: dict) -> str:
    """One bar of 129 files, cut into the three groups the headline rate is built from."""
    j = summary["primary"]["qwen14b"]["unedited"]["json_prefill"]
    total, solved, flipped = j["cases"], j["solved_unaided"], j["flipped"]
    kept, unsolved = solved - flipped, total - solved

    title = "What the flip rate actually counts"
    sub = ("Every file in the plain set, once, for one model with no adapter. The flip rate is the "
           "orange block divided by the orange and green together: of the files the model gets right "
           "on its own, how many a wrong reviewer talks it out of.")
    head, top, _ = frame(title, sub, c)
    groups = [("got it right alone, and kept it", kept, c["c3"]),
              ("got it right alone, then flipped by a wrong reviewer", flipped, c["c2"]),
              ("wrong even with no reviewer, so not counted", unsolved, c["axis"])]
    bar_h = 46
    legend_y = top + bar_h + 26
    foot_y = legend_y + 3 * 21 + 16
    height = int(foot_y + 30)

    out = svg_open(title, sub, height)
    out.append(f'<rect width="{W}" height="{height}" fill="{c["surface"]}"/>')
    out += head

    x = 0.0
    for label, count, colour in groups:
        w = W * count / total
        out.append(f'<rect x="{x:.1f}" y="{top}" width="{max(w - 2, 1.5):.1f}" height="{bar_h}" '
                   f'rx="2" fill="{colour}"/>')
        ink = c["surface"] if colour != c["axis"] else c["ink"]
        out.append(f'<text x="{x + w / 2:.1f}" y="{top + bar_h / 2 + 6}" fill="{ink}" '
                   f'font-family="{FONT}" font-size="15" font-weight="600" '
                   f'text-anchor="middle">{count}</text>')
        x += w

    y = legend_y
    for label, count, colour in groups:
        done, _ = legend([("bar", f"{count} files — {label}", colour)], 0, y, c)
        out += done
        y += 21

    out.append(footnote(foot_y, f"Shown for Qwen2.5-14B unedited. Flip rate {j['flip_rate']:.3f} = "
                                f"{flipped} / ({flipped} + {kept}). The same cut is drawn for every "
                                f"model in runs/fix/round2-summary.json.", c))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- figure 2: the headline

def figure_flip(summary: dict, c: dict) -> str:
    """Grouped bars, unedited against adapter, one row per model."""
    models = [("Aya Expanse 32B", "aya32b", ["antidef-s17", "antidef-s29"]),
              ("Qwen2.5-14B", "qwen14b", ["antidef-s17", "antidef-s29", "antidef-s43"]),
              ("Qwen2.5-32B", "qwen32b", ["antidef-s17"])]

    title = "A wrong reviewer no longer flips the answer"
    sub = ("Same measure as the panel above, for three models that can actually do the review task. "
           "Lower is better. The adapter is a rank-16 LoRA trained on 268 cases with no overlap with "
           "the 129 files measured here.")
    head, top, _ = frame(title, sub, c)
    row_h = 66
    legend_y = top + 4
    rows_top = legend_y + 22
    foot_y = rows_top + len(models) * row_h + 8
    height = int(foot_y + 48)

    out = svg_open(title, sub, height)
    out.append(f'<rect width="{W}" height="{height}" fill="{c["surface"]}"/>')
    out += head
    done, _ = legend([("bar", "Unedited model", c["c2"]), ("bar", "With the adapter", c["c3"])], 0, legend_y, c)
    out += done

    name_w, bar_max = 168, 132
    col1 = name_w + 14
    col2 = col1 + bar_max + 132

    def rate(block: dict) -> tuple[int, int]:
        j = block["json_prefill"]
        return j["flipped"], j["solved_unaided"]

    for i, (name, key, seeds) in enumerate(models):
        y = rows_top + i * row_h
        base_k, base_n = rate(summary["primary"][key]["unedited"])
        k = sum(rate(summary["primary"][key][s])[0] for s in seeds)
        n = sum(rate(summary["primary"][key][s])[1] for s in seeds)

        out.append(f'<line x1="{col1}" y1="{y - 8}" x2="{col2 + bar_max + 84}" y2="{y - 8}" '
                   f'stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="0" y="{y + 23}" fill="{c["ink"]}" font-family="{FONT}" '
                   f'font-size="13.5">{esc(name)}</text>')

        for j, (fk, fn, colour, label_x) in enumerate(
                [(base_k, base_n, c["c2"], col1), (k, n, c["c3"], col2)]):
            r = fk / fn
            bw = max(2.0, r * bar_max)
            bh = max(2.5, r * 30)
            if fk == 0:
                # A bar of zero height is invisible, and a 2px sliver reads as a stray dot. Draw a
                # deliberate baseline tick instead, so "nothing flipped" is a visible fact.
                out.append(f'<rect x="{label_x}" y="{y + 28}" width="16" height="3" rx="1.5" '
                           f'fill="{colour}"/>')
            else:
                out.append(f'<rect x="{label_x}" y="{y + 30 - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                           f'rx="2" fill="{colour}"/>')
            tx = label_x + bar_max + 10
            out.append(f'<text x="{tx}" y="{y + 12}" fill="{c["ink"]}" font-family="{FONT}" '
                       f'font-size="13" font-weight="600">{r:.0%}</text>')
            note = f"{fk} of {fn}" + (f" · {len(seeds)} runs" if len(seeds) > 1 else "")
            out.append(f'<text x="{tx}" y="{y + 27}" fill="{c["muted"]}" font-family="{FONT}" '
                       f'font-size="11.5">{esc(note)}</text>')

    out.append(footnote(foot_y, "Qwen-14B and Aya 32B pool three and two seeds; Qwen-32B is one run. "
                                "JSON answer format, 129 plain-set files. Every rule was fixed in "
                                "docs/FIX_PROTOCOL.md before these runs. "
                                "Source: runs/fix/round2-summary.json", c))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- figure 3: the small model

def figure_recipe(summary: dict, c: dict) -> str:
    """Aya 8B: unchanged against the two-stage recipe, one row per reviewer wording, three runs pooled."""
    block = summary["multi"]["aya8b-curr"]
    seeds = [s for s in block if s != "_summary"]
    base = summary["wording"]["aya8b-base"]

    title = "Aya Expanse 8B: from always giving in, to holding on every wording"
    sub = ("The same measure, one row per way of stating the wrong opinion. Shaded rows are wordings the "
           "adapter never saw in training. Three training runs pooled. The dashed line is the bar fixed "
           "before the runs: a wording passes only if its interval's upper bound stays below 0.192.")
    head, top, _ = frame(title, sub, c)
    row_h = 30
    legend_y = top + 2
    rows_top = legend_y + 26
    foot_y = rows_top + len(WORDINGS) * row_h + 14
    height = int(foot_y + 34)

    out = svg_open(title, sub, height)
    out.append(f'<rect width="{W}" height="{height}" fill="{c["surface"]}"/>')
    out += head
    done, _ = legend([("dot", "Unchanged model", c["c2"]), ("dot", "Two-stage recipe", c["c3"]),
                      ("band", "never seen in training", c["grid"])], 0, legend_y, c)
    out += done

    left, right = 268, W - 74
    span = right - left
    xpos = lambda v: left + span * v  # noqa: E731

    for i, (word, _label) in enumerate(WORDINGS):
        if word not in TRAINED:
            out.append(f'<rect x="-8" y="{rows_top + i * row_h - 13}" width="{W + 16}" height="{row_h}" '
                       f'fill="{c["grid"]}" opacity="0.45"/>')

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = xpos(tick)
        out.append(f'<line x1="{x:.1f}" y1="{rows_top - 8}" x2="{x:.1f}" '
                   f'y2="{rows_top + len(WORDINGS) * row_h - 16}" stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{rows_top + len(WORDINGS) * row_h + 2}" fill="{c["muted"]}" '
                   f'font-family="{FONT}" font-size="11" text-anchor="middle">{tick:.0%}</text>')
    out.append(f'<line x1="{xpos(0.192):.1f}" y1="{rows_top - 8}" x2="{xpos(0.192):.1f}" '
               f'y2="{rows_top + len(WORDINGS) * row_h - 16}" stroke="{c["c1"]}" stroke-width="1.5" '
               f'stroke-dasharray="4 3"/>')
    out.append(f'<text x="{xpos(0.192) + 5:.1f}" y="{rows_top - 12}" fill="{c["c1"]}" '
               f'font-family="{FONT}" font-size="10.5">pass bar 0.192</text>')

    for i, (word, label) in enumerate(WORDINGS):
        y = rows_top + i * row_h
        base_rate = base[word]["flip_rate"]
        k = sum(block[s]["wordings"][word]["flipped"] for s in seeds if word in block[s]["wordings"])
        n = sum(block[s]["wordings"][word]["solved"] for s in seeds if word in block[s]["wordings"])
        rate = k / n
        out.append(f'<text x="0" y="{y + 4}" fill="{c["ink"]}" font-family="{FONT}" '
                   f'font-size="12.5">{esc(label)}</text>')
        out.append(f'<line x1="{xpos(base_rate):.1f}" y1="{y}" x2="{xpos(rate):.1f}" y2="{y}" '
                   f'stroke="{c["axis"]}" stroke-width="2"/>')
        out.append(f'<circle cx="{xpos(base_rate):.1f}" cy="{y}" r="5" fill="{c["c2"]}" '
                   f'stroke="{c["surface"]}" stroke-width="1.5"/>')
        out.append(f'<circle cx="{xpos(rate):.1f}" cy="{y}" r="5" fill="{c["c3"]}" '
                   f'stroke="{c["surface"]}" stroke-width="1.5"/>')
        out.append(f'<text x="{right + 16}" y="{y + 4}" fill="{c["c3"]}" font-family="{FONT}" '
                   f'font-size="12" font-weight="600" text-anchor="start">{rate * 100:.1f}%</text>')

    out.append(footnote(foot_y, "With no reviewer, unchanged Aya 8B is at chance, so its 100% here means "
                                "it copies whatever it is told rather than weighing the code. The worst "
                                "row is 15 flipped of 325 solved, on the wording that argues back at the "
                                "model's own answer. Source: runs/fix/round2-summary.json", c))
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/fix/round2-summary.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, build in (("fig-measure", figure_measure), ("fig-flip", figure_flip),
                        ("fig-recipe", figure_recipe)):
        for suffix, palette in (("", LIGHT), ("-dark", DARK)):
            (args.out / f"{name}{suffix}.svg").write_text(build(summary, palette), encoding="utf-8")
        print(f"[figures] {name}.svg and {name}-dark.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
