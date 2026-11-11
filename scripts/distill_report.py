"""Assemble the distillation arm's before/after table from saved files only.

Why this exists
    Every number in the write-up has to be re-readable from a file. This reads the baseline and
    post-SFT artifacts and emits one JSON plus a markdown table, so the submission text is copied
    from a generated file rather than retyped from a log.

What it answers
    1. Competence: did training move coding ability (HumanEval pass@1) and review skill
       (solved_unaided on the plain set)?
    2. Deference: of the cases the model NOW solves unaided, what share does a wrong reviewer flip?

    The second number is only interpretable if the first clears chance. The script says so
    explicitly rather than leaving a reader to spot it: `competent` is False when the Wilson
    interval for solved_unaided includes the 50% chance line.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int) -> list | None:
    if not n:
        return None
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def flip_for(run_dir: Path, instrument: str) -> dict | None:
    """Run the project's own flip analysis on a saved table, so the metric matches the main study."""
    import importlib.util
    import sys
    table = read_json(run_dir / "table.json")
    if table is None:
        return None
    spec = importlib.util.spec_from_file_location("_flip", ROOT / "scripts" / "wsp_flip_analysis.py")
    flip = importlib.util.module_from_spec(spec)
    sys.modules["_flip"] = flip
    spec.loader.exec_module(flip)
    return flip.analyse(table, instrument)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bench", type=Path, default=ROOT / "runs/bench/aya8b-humaneval.json")
    parser.add_argument("--sft-bench", type=Path, default=ROOT / "runs/bench/aya8b-sft-humaneval.json")
    parser.add_argument("--baseline-mbpp", type=Path, default=ROOT / "runs/bench/aya8b-mbpp.json")
    parser.add_argument("--sft-mbpp", type=Path, default=None)
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "runs/wsp-v3/plain-aya8b")
    parser.add_argument("--sft-run", type=Path, default=ROOT / "runs/distill/plain-aya8b-sft")
    parser.add_argument("--train-meta", type=Path, default=ROOT / "runs/distill/aya8b-sft/train_meta.json")
    parser.add_argument("--instruments", default="json_prefill,letters_a_vulnerable,letters_b_vulnerable")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/distill/distill-summary.json")
    args = parser.parse_args(argv)

    instruments = [i for i in args.instruments.split(",") if i]
    out: dict = {"train_meta": read_json(args.train_meta), "coding": {}, "review": {}}

    for label, path in (("baseline_humaneval", args.baseline_bench), ("sft_humaneval", args.sft_bench),
                        ("baseline_mbpp", args.baseline_mbpp), ("sft_mbpp", args.sft_mbpp)):
        d = read_json(path) if path else None
        out["coding"][label] = None if d is None else {
            "benchmark": d["benchmark"], "passed": d["passed"], "tasks": d["tasks"],
            "pass_at_1": d["pass_at_1"], "ci95": d["pass_at_1_ci95"],
            "adapter_id": d.get("adapter_id"), "source": str(path)}

    for label, run in (("baseline", args.baseline_run), ("sft", args.sft_run)):
        out["review"][label] = {"source": str(run)}
        for inst in instruments:
            res = flip_for(run, inst)
            if res is None:
                out["review"][label][inst] = None
                continue
            a = res["all"]
            n, solved = a["cases"], a["solved_unaided"]
            band = wilson(solved, n)
            out["review"][label][inst] = {
                "cases": n, "solved_unaided": solved, "solved_ci95": band,
                # Chance on a balanced two-way task is 0.5; competence requires the interval to clear it.
                "competent": bool(band and band[0] > 0.5),
                "flipped": a["flipped_by_wrong_reviewer"], "flip_rate": a["flip_rate"],
                "flip_rate_ci95": a["flip_rate_ci95"],
                "rescued_by_correct_reviewer": a["rescued_by_correct_reviewer"]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")

    # ---- human-readable table
    lines = ["", "### Coding ability (public benchmarks, greedy)", "",
             "| arm | benchmark | pass@1 | 95% CI |", "|---|---|---|---|"]
    for label, d in out["coding"].items():
        if d:
            lines.append(f"| {label} | {d['benchmark']} | {d['passed']}/{d['tasks']} = "
                         f"{d['pass_at_1']} | {d['ci95']} |")
    lines += ["", "### Review skill and deference (plain-pooled, 129 cases)", "",
              "| arm | instrument | solved unaided | competent? | flip rate | 95% CI |",
              "|---|---|---|---|---|---|"]
    for label in ("baseline", "sft"):
        for inst in instruments:
            d = out["review"][label].get(inst)
            if not d:
                continue
            lines.append(f"| {label} | {inst} | {d['solved_unaided']}/{d['cases']} "
                         f"{d['solved_ci95']} | {'YES' if d['competent'] else 'no (at chance)'} | "
                         f"{d['flip_rate']} | {d['flip_rate_ci95']} |")
    table = "\n".join(lines)
    (args.output.parent / "distill-summary.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {args.output} and {args.output.parent / 'distill-summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
