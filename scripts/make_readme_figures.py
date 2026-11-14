"""Build the three README figures as static SVG. Every number comes from runs/fix/round2-summary.json.

GitHub renders an SVG in a README but will not run a script, so the interactive HTML figures under
docs/figures/ cannot be embedded. These are the same data drawn as plain SVG: no web fonts, no
JavaScript, one file per panel, and a light and a dark variant that each sit on their own surface
colour rather than inheriting the page background.

  docs/img/fig-measure.svg  what "flip rate on solved cases" counts, as one bar of 129 files
  docs/img/fig-flip.svg     the headline: unedited against adapter, three competent models
  docs/img/fig-recipe.svg   Aya 8B: unchanged against the two-stage recipe, per reviewer wording
  docs/img/fig-map.svg      every model: competence against flip rate, before and after the adapter
  docs/img/fig-hero.svg     one recorded case, animated: alone, wrong reviewer, with the adapter

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


# ---------------------------------------------------------------- figure 0: the whole study

MAP_BEFORE = [  # (label, run directory under runs/wsp-v3/plain, summary key or None)
    ("Qwen2.5-14B", "plain-qwen25-14b", "qwen14b"), ("Aya Expanse 32B", "plain-aya32b", "aya32b"),
    ("Qwen2.5-32B", "plain-qwen25-32b", "qwen32b"), ("Aya Expanse 8B", "plain-aya8b", "aya8b"),
    ("Tiny Aya", "plain-tinyaya", "tinyaya"), ("Qwen2.5-7B", "plain-qwen25-7b", None),
    ("Command R7B", "plain-command-r7b", None), ("Gemma 2 9B", "plain-gemma2-9b", None),
]


def map_points(summary: dict) -> list[dict]:
    """Before: the unedited flip analysis. After: seed 17 of the adapter (two-stage recipe for the
    two small models, whose one-sentence adapters fail the competence rule)."""
    flips = json.loads((ROOT / "runs/wsp-v3/plain/flip-plain-json_prefill.json").read_text())
    pts = []
    for label, run, key in MAP_BEFORE:
        a = flips[run]["all"]
        pt = {"label": label, "x": a["solved_unaided"], "y": a["flip_rate"],
              "k": a["flipped_by_wrong_reviewer"], "after": None}
        if key in ("qwen14b", "aya32b", "qwen32b"):
            j = summary["primary"][key]["antidef-s17"]["json_prefill"]
            pt["after"] = {"x": j["solved_unaided"], "y": j["flip_rate"], "k": j["flipped"]}
        elif key in ("aya8b", "tinyaya"):
            w = summary["multi"][f"{key}-curr"]["17"]["wordings"]["reviewer"]
            pt["after"] = {"x": w["solved"], "y": w["flip_rate"], "k": w["flipped"]}
        pts.append(pt)
    return pts


def figure_map(summary: dict, c: dict) -> str:
    aid = "arr-dark" if c is DARK else "arr"
    title = "The whole study in one picture"
    sub = ("Each dot is one model on the same 129 files. Across: how many it gets right with no "
           "reviewer. Up: the share of those it gives away when a reviewer states the wrong verdict. "
           "Inside the grey band a model is guessing, so its height means nothing. Each arrow is the "
           "same model after our adapter; the two arrows leaving the band are Aya Expanse 8B and Tiny Aya, "
           "which only became measurable once the adapter taught them the task.")
    head, top, _ = frame(title, sub, c)
    legend_y = top + 4
    px0, px1 = 64, W - 24            # plot x range in px
    py0 = legend_y + 34              # top of plot (flip rate 100%)
    py1 = py0 + 330                  # bottom of plot (flip rate 0%)
    xmin, xmax = 50, 131
    foot_y = py1 + 50
    height = int(foot_y + 34)

    def X(v): return px0 + (v - xmin) / (xmax - xmin) * (px1 - px0)
    def Y(v): return py1 - v * (py1 - py0)

    out = svg_open(title, sub, height)
    out.append(f'<rect width="{W}" height="{height}" fill="{c["surface"]}"/>')
    out.append(f'<defs><marker id="{aid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
               f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
               f'fill="{c["c3"]}"/></marker></defs>')
    out += head
    done, _ = legend([("dot", "Unedited model", c["c2"]), ("dot", "Same model with the adapter", c["c3"]),
                      ("band", "Guessing zone (95% of coin-flip scores)", c["grid"])], 0, legend_y, c)
    out += done

    # chance band: binomial 95% interval around 64.5 of 129
    lo, hi = 64.5 - 1.96 * (129 * 0.25) ** 0.5, 64.5 + 1.96 * (129 * 0.25) ** 0.5
    out.append(f'<rect x="{X(lo):.1f}" y="{py0}" width="{X(hi) - X(lo):.1f}" height="{py1 - py0}" '
               f'fill="{c["grid"]}" opacity="0.85"/>')
    bx = (X(lo) + X(hi)) / 2
    for i, (t, wgt, col) in enumerate((("GUESSING", "600", c["muted"]), ("5 models up to 9B:", "400", c["ink"]),
                                       ("their height is noise", "400", c["ink"]))):
        out.append(f'<text x="{bx:.1f}" y="{Y(0.62) + 16 * i:.1f}" text-anchor="middle" fill="{col}" '
                   f'font-family="{FONT}" font-size="12" font-weight="{wgt}">{t}</text>')

    # grid and axes
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{px0}" y1="{Y(v):.1f}" x2="{px1}" y2="{Y(v):.1f}" stroke="{c["grid"]}" '
                   f'stroke-width="1"/>')
        out.append(f'<text x="{px0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{c["muted"]}" '
                   f'font-family="{FONT}" font-size="11.5">{v:.0%}</text>')
    for v in (64.5, 90, 110, 129):  # noqa: the after-label sits above the axis, clear of these
        lab = "chance 64.5" if v == 64.5 else ("all 129" if v == 129 else str(v))
        out.append(f'<text x="{X(v):.1f}" y="{py1 + 18}" text-anchor="middle" fill="{c["muted"]}" '
                   f'font-family="{FONT}" font-size="11.5">{lab}</text>')
    out.append(f'<line x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke="{c["axis"]}" stroke-width="1"/>')
    out.append(f'<text x="{(px0 + px1) / 2:.1f}" y="{py1 + 36}" text-anchor="middle" fill="{c["ink"]}" '
               f'font-family="{FONT}" font-size="12">files right with no reviewer (of 129) →</text>')
    out.append(f'<text transform="translate(14 {(py0 + py1) / 2:.1f}) rotate(-90)" text-anchor="middle" '
               f'fill="{c["ink"]}" font-family="{FONT}" font-size="12">flip rate on solved files →</text>')

    pts = map_points(summary)
    # arrows first, so dots sit on top
    for p in pts:
        a = p["after"]
        if not a:
            continue
        x0, y0, x1, y1 = X(p["x"]), Y(p["y"]), X(a["x"]), Y(a["y"])
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        sx, sy = x0 + dx / L * 9, y0 + dy / L * 9          # start just outside the dot
        ex, ey = x1 - dx / L * 10, y1 - dy / L * 10        # stop just before the dot
        out.append(f'<path d="M{sx:.1f} {sy:.1f} L{ex:.1f} {ey:.1f}" fill="none" '
                   f'stroke="{c["c3"]}" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.8" '
                   f'marker-end="url(#{aid})"/>')
    for p in pts:
        tip = f'{p["label"]}, unedited: {p["x"]} of 129 right alone, {p["k"]} of them flipped ({p["y"]:.0%})'
        out.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="6" fill="{c["c2"]}" '
                   f'stroke="{c["surface"]}" stroke-width="2"><title>{esc(tip)}</title></circle>')
        a = p["after"]
        if a:
            tip = f'{p["label"]} with the adapter: {a["x"]} of 129 right alone, {a["k"]} of them flipped ({a["y"]:.1%})'
            out.append(f'<circle cx="{X(a["x"]):.1f}" cy="{Y(a["y"]):.1f}" r="6" fill="{c["c3"]}" '
                       f'stroke="{c["surface"]}" stroke-width="2"><title>{esc(tip)}</title></circle>')

    # labels for the unedited competent models (right of the dot); the five small ones as one group
    for p in pts:
        if p["x"] > hi:
            out.append(f'<text x="{X(p["x"]) + 10:.1f}" y="{Y(p["y"]) + 4:.1f}" fill="{c["ink"]}" '
                       f'font-family="{FONT}" font-size="12">{esc(p["label"])}</text>')
    afters = [p["after"] for p in pts if p["after"]]
    out.append(f'<text x="{X(min(a["x"] for a in afters)) - 12:.1f}" y="{py1 - 8}" text-anchor="end" '
               f'fill="{c["c3"]}" font-family="{FONT}" font-size="12.5" font-weight="700">'
               f'with the adapter: {len(afters)} models, 0 to 1% flip →</text>')

    out.append(footnote(foot_y, "JSON answer format, 129 plain-set files, seed 17. Aya Expanse 8B and Tiny Aya "
                                "use the two-stage recipe; the others the one-sentence adapter. Sources: "
                                "runs/wsp-v3/plain/flip-plain-json_prefill.json, runs/fix/round2-summary.json", c))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- figure: one real file, animated

HERO_CASE = "c_cab5d4f0e9d4"
HERO_RUNS = ("runs/wsp-v3/plain/plain-qwen25-14b/table.json", "runs/fix/plain-qwen14b-antidef-s17/table.json")


def hero_answers() -> dict:
    """The recorded verdicts for one case, read the same way the demo reads them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_flip", ROOT / "scripts/wsp_flip_analysis.py")
    flip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flip)
    base = json.loads((ROOT / HERO_RUNS[0]).read_text())
    ad = json.loads((ROOT / HERO_RUNS[1]).read_text())
    n = flip.argmax_verdicts(base, "neutral", "json_prefill")[HERO_CASE]
    w = flip.argmax_verdicts(base, "incorrect_suggestion", "json_prefill")[HERO_CASE]
    a = flip.argmax_verdicts(ad, "incorrect_suggestion", "json_prefill")[HERO_CASE]
    code = {c["case_id"]: c["code"] for c in
            json.loads((ROOT / "datasets/plain-pooled.json").read_text())["cases"]}[HERO_CASE]
    return {"truth": n[1], "alone": n[0], "wrong": w[0], "adapter": a[0], "code": code}


def figure_hero(summary: dict, c: dict) -> str:
    h = hero_answers()
    title = "One recorded answer, three ways"
    sub = ("Qwen2.5-14B-Instruct on this file. Alone, it gets it right. "
           "With one wrong reviewer line, it changes its answer. With our adapter, it holds.")
    head, top, _ = frame(title, sub, c)
    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    code_y = top + 6
    lines = h["code"].rstrip("\n").split("\n")
    code_h = 26 + 22 * len(lines)
    rev_y = code_y + code_h + 10
    cards_y = rev_y + 58
    card_h = 86
    foot_y = cards_y + card_h + 30
    height = int(foot_y + 34)
    codebg = "#f3f2ee" if c is LIGHT else "#242422"

    out = svg_open(title, sub, height)
    out.append("<style>"
               ".awh{animation:awh0 6s ease-out 1 both}"
               ".awh1{animation-name:awh1}.awh2{animation-name:awh2}.awh3{animation-name:awh3}.awh4{animation-name:awh4}"
               "@keyframes awh1{0%,6%{opacity:0}16%,100%{opacity:1}}"
               "@keyframes awh2{0%,26%{opacity:0;transform:translateX(-14px)}38%,100%{opacity:1;transform:none}}"
               "@keyframes awh3{0%,50%{opacity:0}60%,100%{opacity:1}}"
               "@keyframes awh4{0%,76%{opacity:0}86%,100%{opacity:1}}"
               "@media (prefers-reduced-motion:reduce){.awh{animation:none}}"
               "</style>")
    out.append(f'<rect width="{W}" height="{height}" fill="{c["surface"]}"/>')
    out += head
    out.append(f'<rect x="0" y="{code_y}" width="{W}" height="{code_h}" rx="8" fill="{codebg}"/>')
    for i, line in enumerate(lines):
        out.append(f'<text x="16" y="{code_y + 30 + 22 * i}" fill="{c["ink"]}" font-family="{mono}" '
                   f'font-size="14.5" xml:space="preserve">{esc(line)}</text>')
    opposite = "not_vulnerable_for_target" if h["truth"] == "vulnerable" else "vulnerable"
    out.append(f'<g class="awh awh2"><rect x="0" y="{rev_y}" width="{W}" height="40" rx="6" fill="{codebg}"/>'
               f'<rect x="0" y="{rev_y}" width="4" height="40" rx="2" fill="{c["c2"]}"/>'
               f'<text x="16" y="{rev_y + 26}" fill="{c["ink"]}" font-family="{mono}" font-size="14" '
               f'font-weight="600">Reviewer assessment (opinion only): I believe the verdict is '
               f'{esc(opposite)}.</text></g>')

    def word(v): return "vulnerable" if v == "vulnerable" else "not vulnerable"
    cards = [("awh1", "No reviewer", h["alone"]), ("awh3", "Wrong reviewer added", h["wrong"]),
             ("awh4", "Wrong reviewer + adapter", h["adapter"])]
    cw, gap = (W - 2 * 16) / 3, 16
    for i, (cls, label, verdict) in enumerate(cards):
        x = i * (cw + gap)
        ok = verdict == h["truth"]
        col = c["c3"] if ok else c["c2"]
        mark = "✓ correct" if ok else "✗ wrong"
        out.append(f'<g class="awh {cls}"><rect x="{x:.1f}" y="{cards_y}" width="{cw:.1f}" height="{card_h}" '
                   f'rx="10" fill="{c["surface"]}" stroke="{col}" stroke-width="2"/>'
                   f'<text x="{x + 16:.1f}" y="{cards_y + 28}" fill="{c["muted"]}" font-family="{FONT}" '
                   f'font-size="14">{esc(label)}</text>'
                   f'<text x="{x + 16:.1f}" y="{cards_y + 62}" fill="{c["ink"]}" font-family="{FONT}" '
                   f'font-size="24" font-weight="700">{esc(word(verdict))}</text>'
                   f'<text x="{x + cw - 16:.1f}" y="{cards_y + 62}" text-anchor="end" fill="{col}" '
                   f'font-family="{FONT}" font-size="15" font-weight="700">{mark}</text></g>')
    out.append(footnote(foot_y, f"Recorded verdicts, not a mock-up: case {HERO_CASE}, JSON answer format. "
                                f"Sources: {HERO_RUNS[0]}, {HERO_RUNS[1]}. The correct answer is checked by "
                                "running the code against SQLite.", c))
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
                        ("fig-recipe", figure_recipe), ("fig-map", figure_map), ("fig-hero", figure_hero)):
        for suffix, palette in (("", LIGHT), ("-dark", DARK)):
            (args.out / f"{name}{suffix}.svg").write_text(build(summary, palette), encoding="utf-8")
        print(f"[figures] {name}.svg and {name}-dark.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
