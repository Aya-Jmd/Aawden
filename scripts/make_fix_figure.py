"""Build docs/figures/fix.html from the generated summaries. No number is typed by hand.

Reads `runs/fix/round2-summary.json` (scripts/fix_report_r2.py). Writes one self-contained HTML file:
panel 1 is the flip rate on the plain set per model (unedited, task-only control, anti-deference
adapter); panel 2 is the flip rate per reviewer wording, unedited against adapter, one block per
model. A table view carries every value with its counts and interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen14b": "Qwen2.5-14B", "aya32b": "Aya Expanse 32B", "qwen32b": "Qwen2.5-32B"}
WORDING_NAMES = {"reviewer": "Reviewer (trained on)", "authority": "Senior reviewer", "ciso": "CISO sign-off",
                 "scanner": "Static analysis tool", "consensus": "Three reviewers agree",
                 "code_comment": "Comment in the code", "argued": "Reviewer gives a reason",
                 "insistent": "Reviewer insists, threatens escalation", "pushback": "Pushback on the model's own answer"}
SERIES = ["Unedited", "Task-only training", "Anti-deference adapter"]

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Aawden: an adapter that stops the model deferring</title>
<style>
:root{--s:#fcfcfb;--t1:#0b0b0b;--t2:#52514e;--grid:#e6e5e0;--axis:#c9c8c2;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a}
@media (prefers-color-scheme:dark){:root{--s:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--grid:#2e2e2c;--axis:#4a4a46;--c1:#3987e5;--c2:#d95926;--c3:#199e70}}
body{margin:0;background:var(--s);color:var(--t1);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:28px 24px}
h1{font-size:19px;margin:0 0 4px;font-weight:600}h2{font-size:15px;margin:26px 0 2px;font-weight:600}h3{font-size:13px;margin:14px 0 0;font-weight:600}
.sub{color:var(--t2);margin:0 0 10px;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:18px;color:var(--t2);font-size:12.5px;margin:6px 0 4px}.legend span{display:inline-flex;align-items:center;gap:6px}
svg{display:block;max-width:100%}svg text{fill:var(--t2);font-size:12px}svg .name{fill:var(--t1);font-size:13px}
.tip{position:fixed;pointer-events:none;background:var(--s);color:var(--t1);border:1px solid var(--axis);border-radius:6px;padding:6px 9px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none}
details{margin-top:22px;color:var(--t2)}table{border-collapse:collapse;font-size:12.5px;margin-top:8px}td,th{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:right}td:first-child,th:first-child{text-align:left}
.note{color:var(--t2);font-size:12.5px;margin-top:6px}
</style></head><body><div class="wrap">
<h1>A small adapter stops competent models giving in to a wrong reviewer</h1>
<p class="sub">Of the plain-set cases each model solves with no reviewer, the share it gets wrong once a reviewer states the wrong verdict. JSON answer format, 129 cases.</p>
<div class="legend" id="lg"></div>
<svg id="p1" width="832"></svg>
<p class="note">Task-only training uses the same cases and the same number of training steps, but never shows the model a reviewer.</p>
<h2>It holds for reviewers the adapter never saw in training</h2>
<p class="sub">Same measure, one row per way of stating the opinion. The adapter was trained on the first wording only.</p>
<div id="p2"></div>
<details><summary>Table view</summary><div id="tbl"></div></details>
<p class="note">Source: runs/fix/round2-summary.json · 95% Wilson intervals in the table.</p>
</div><div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const F=[{k:"Unedited",c:"var(--c1)",m:"circle"},{k:"Task-only training",c:"var(--c2)",m:"square"},{k:"Anti-deference adapter",c:"var(--c3)",m:"triangle"}];
const NS="http://www.w3.org/2000/svg",tip=document.getElementById("tip");
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p&&p.appendChild(e);return e}
function marker(p,m,x,y,c){const r=5.5;let e;
 if(m=="circle")e=el("circle",{cx:x,cy:y,r},p);
 else if(m=="square")e=el("rect",{x:x-r,y:y-r,width:2*r,height:2*r,rx:1.5},p);
 else e=el("path",{d:`M${x},${y-r-1}L${x+r+1},${y+r}L${x-r-1},${y+r}Z`},p);
 e.setAttribute("fill",c);e.setAttribute("stroke","var(--s)");e.setAttribute("stroke-width",2);return e}
const pct=v=>(v*100).toFixed(v>0&&v<0.1?1:0).replace(/\.0$/,"")+"%";
const used=new Set();D.p1.forEach(r=>r.pts.forEach(p=>used.add(p.s)));
const lg=document.getElementById("lg");F.forEach((f,i)=>{if(!used.has(i))return;const s=document.createElement("span");const v=document.createElementNS(NS,"svg");v.setAttribute("width",14);v.setAttribute("height",14);marker(v,f.m,7,7,f.c);s.appendChild(v);s.appendChild(document.createTextNode(f.k));lg.appendChild(s)});
function panel(svg,rows,L){const R=150,W=832,rh=36,top=10,H=top+rows.length*rh+30;svg.setAttribute("height",H);svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
 const x=v=>L+v*(W-L-R);
 for(let t=0;t<=1.0001;t+=0.25){el("line",{x1:x(t),x2:x(t),y1:top,y2:top+rows.length*rh,stroke:"var(--grid)"},svg);const tx=el("text",{x:x(t),y:top+rows.length*rh+18,"text-anchor":"middle"},svg);tx.textContent=Math.round(t*100)+"%"}
 rows.forEach((r,i)=>{const y=top+i*rh+rh/2;const nm=el("text",{x:0,y:y+4,class:"name"},svg);nm.textContent=r.name;
  const vs=r.pts.map(p=>p.v),lo=Math.min(...vs),hi=Math.max(...vs);el("line",{x1:x(lo),x2:x(hi),y1:y,y2:y,stroke:"var(--axis)","stroke-width":2,"stroke-linecap":"round"},svg);
  const first=r.pts[0],last=r.pts[r.pts.length-1];const s=el("text",{x:W-R+12,y:y+4},svg);s.textContent=r.pts.length>1?pct(first.v)+" → "+pct(last.v):pct(first.v);
  r.pts.forEach((p,j)=>{const dup=r.pts.slice(0,j).some(u=>Math.abs(u.v-p.v)<0.012);const yy=dup?y+9:y;const g=el("g",{},svg);el("circle",{cx:x(p.v),cy:yy,r:12,fill:"transparent"},g);marker(g,F[p.s].m,x(p.v),yy,F[p.s].c);
   g.addEventListener("mousemove",ev=>{tip.style.display="block";tip.style.left=ev.clientX+12+"px";tip.style.top=ev.clientY+12+"px";
    tip.innerHTML=`<b>${r.name}</b><br>${F[p.s].k}${p.note?" ("+p.note+")":""}<br>flipped ${p.k} of ${p.n} solved (${pct(p.v)})<br>95% CI ${p.ci.map(c=>pct(Math.max(0,c))).join("–")}`});
   g.addEventListener("mouseleave",()=>tip.style.display="none")})})}
panel(document.getElementById("p1"),D.p1,190);
const p2=document.getElementById("p2");D.p2.forEach(b=>{const h=document.createElement("h3");h.textContent=b.model;p2.appendChild(h);const svg=document.createElementNS(NS,"svg");svg.setAttribute("width",832);p2.appendChild(svg);panel(svg,b.rows,270)});
let h="<table><tr><th>row</th><th>series</th><th>flipped / solved</th><th>flip rate</th><th>95% CI</th></tr>";
const add=(name,p)=>{h+=`<tr><td>${name}</td><td>${F[p.s].k}${p.note?" ("+p.note+")":""}</td><td>${p.k} / ${p.n}</td><td>${pct(p.v)}</td><td>${p.ci.map(c=>pct(Math.max(0,c))).join("–")}</td></tr>`};
D.p1.forEach(r=>r.all.forEach(p=>add(r.name,p)));D.p2.forEach(b=>b.rows.forEach(r=>r.pts.forEach(p=>add(b.model+" · "+r.name,p))));
document.getElementById("tbl").innerHTML=h+"</table>";
</script></body></html>
"""


def point(series: int, flipped, solved, rate, ci, note=None) -> dict:
    return {"s": series, "k": flipped, "n": solved, "v": rate, "ci": ci, "note": note}


def build(summary: dict) -> dict:
    p1, p2 = [], []
    for key, name in MODELS.items():
        block = summary["primary"].get(key)
        if not block:
            continue
        def pt(series, arm, note=None):
            j = block[arm]["json_prefill"]
            return point(series, j["flipped"], j["solved_unaided"], j["flip_rate"], j["flip_rate_ci95"], note)
        adapters = sorted(a for a in block if a.startswith("antidef-"))
        if not adapters:
            continue
        everything = [pt(0, "unedited")]
        shown = [pt(0, "unedited")]
        if "neutralonly-s17" in block:
            everything.append(pt(1, "neutralonly-s17"))
            shown.append(pt(1, "neutralonly-s17"))
        for a in adapters:
            everything.append(pt(2, a, a.replace("antidef-s", "seed ")))
        shown.append(pt(2, adapters[0], f"{len(adapters)} seed{'s' if len(adapters) > 1 else ''}, seed 17 shown"))
        p1.append({"name": name, "pts": shown, "all": everything})
        base, ad = summary["wording"].get(f"{key}-base", {}), summary["wording"].get(f"{key}-antidef", {})
        rows = []
        for w, label in WORDING_NAMES.items():
            if w in base and w in ad:
                rows.append({"name": label, "pts": [
                    point(0, base[w]["flipped"], base[w]["solved"], base[w]["flip_rate"], base[w]["ci"]),
                    point(2, ad[w]["flipped"], ad[w]["solved"], ad[w]["flip_rate"], ad[w]["ci"])]})
        if rows:
            p2.append({"model": name, "rows": rows})

    # Small models: the round-5/6 runs. The plain-set plate adapter for these leaked or collapsed, so the
    # figure shows the two-stage recipe instead, pooled over the seeds of each arm.
    for key, name in (("aya8b", "Aya Expanse 8B"), ("tinyaya", "Tiny Aya (3B)")):
        block = summary["primary"].get(key)
        cur = summary.get("multi", {}).get(f"{key}-curr")
        if not block or not cur:
            continue
        seeds = sorted(s for s in cur if s != "_summary")
        rows = []
        for w, label in WORDING_NAMES.items():
            base = summary["wording"].get(f"{key}-base", {}).get(w)
            cells = [cur[s]["wordings"].get(w) for s in seeds]
            cells = [c for c in cells if c]
            if not base or not cells:
                continue
            k, n = sum(c["flipped"] for c in cells), sum(c["solved"] for c in cells)
            ci = wilson(k, n)
            rows.append({"name": label, "pts": [
                point(0, base["flipped"], base["solved"], base["flip_rate"], base["ci"]),
                point(2, k, n, k / n, ci, f"pooled over {len(cells)} runs")]})
        if rows:
            p2.append({"model": name + ", two-stage recipe", "rows": rows})
    return {"p1": p1, "p2": p2}


def wilson(k: int, n: int) -> list[float]:
    """95% Wilson interval for k of n, rounded to three decimals."""
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round(max(0.0, (c - half) / d), 3), round(min(1.0, (c + half) / d), 3)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/fix/round2-summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/fix.html")
    args = parser.parse_args(argv)
    data = build(json.loads(args.summary.read_text()))
    args.output.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"[figure] {len(data['p1'])} models, {sum(len(b['rows']) for b in data['p2'])} wording rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
