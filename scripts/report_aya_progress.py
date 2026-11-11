"""Recompute the competence-study ledger from verified, saved Aya answers. No GPU calls."""
from __future__ import annotations
import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from security_wsp.artifacts import atomic_json, file_hash, read_json, verify_run
from security_wsp.statistics import metrics, percentile
from security_wsp.parsing import parse_response


def compile_progress(root):
    records = []
    for study in ("adapter-v1", "adapter-v2", "adapter-v3", "adapter-v5", "adapter-v6",
                  "adapter-review-v1", "mental-review-v1", "mental-review-v2", "adapter-v7"):
        for stage in ("development", "confirmation"):
            path = root / study / "study" / stage
            if not (path / "manifest.json").is_file():
                continue
            verification = verify_run(path)
            summary = read_json(path / "summary.json")
            rows = read_json(path / "results.json")
            arms = defaultdict(list)
            for row in rows:
                parsed, error = parse_response(row["raw_output"], category=row["category"], line_count=row["source_line_count"])
                if row.get("generation_error"):
                    parsed, error = None, row["generation_error"]
                verdict = parsed["verdict"] if parsed else None
                if (verdict != row["predicted_verdict"] or error != row["parse_error"] or
                        (verdict == row["oracle_verdict"]) != row["correct"]):
                    raise ValueError(f"Stored answer disagrees with strict re-parsing in {path}")
                arms[row["method"]].append(row)
            for method, answers in arms.items():
                measured = metrics(answers)
                stored = summary["metrics"]
                if "balanced_accuracy" not in stored:
                    stored = stored[method]
                for field in ("balanced_accuracy", "n", "invalid_count", "true_positive", "true_negative"):
                    if measured[field] != stored[field]:
                        raise ValueError(f"Stored score disagrees with answers in {path}: {field}")
                by_category = {category: metrics([row for row in answers if row["category"] == category])
                               for category in sorted({row["category"] for row in answers})}
                records.append({"study": study, "stage": stage, "method": method, "metrics": measured,
                                "by_category": by_category, "verification": verification,
                                "manifest_sha256": file_hash(path / "manifest.json"),
                                "adapter_sha256": summary["model"]["adapter_sha256"],
                                "evidence_path": path.relative_to(ROOT).as_posix()})
    confirmations = [record for record in records if record["stage"] == "confirmation"]
    best = max(confirmations, key=lambda record: record["metrics"]["balanced_accuracy"]) if confirmations else None
    if best:
        answers = read_json(root / best["study"] / "study/confirmation/results.json")
        families = sorted({row["family_id"] for row in answers})
        groups = {family: [row for row in answers if row["family_id"] == family] for family in families}
        rng, samples = random.Random(42), []
        for _ in range(10000):
            sampled = [row for family in rng.choices(families, k=len(families)) for row in groups[family]]
            samples.append(metrics(sampled)["balanced_accuracy"])
        best["family_cluster_ci95"] = [percentile(samples, .025), percentile(samples, .975)]
        best["bootstrap"] = {"replicates": 10000, "seed": 42, "families": len(families)}
    return {"scope": "Neutral competence on successive small single-author prospective checks. Earlier checks become exposed after scoring.",
            "threshold": .75, "threshold_exceeded": bool(best and best["metrics"]["balanced_accuracy"] > .75),
            "best_observed_confirmation": best, "records": records,
            "limitations": ["Adaptive sequence of experiments; best observed result has selection bias.",
                            "Developer-authored synthetic families, not an independent final benchmark.",
                            "Neutral accuracy does not establish opinion robustness, explanation quality, or general coding preservation.",
                            "CircuitKIT numerical parity failed; no causal circuit repair is claimed."]}


def render(report):
    lines = ["# Aya competence progress", "", report["scope"], "",
             "Scores below are recomputed from saved answers after manifest verification. Invalid responses remain errors.", "",
             "| Experiment | Stage | Method | Cases | Strict balanced accuracy | Invalid | SQL | Access control |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    percent = lambda value: f"{100 * value:.1f}%"
    for record in report["records"]:
        m, categories = record["metrics"], record["by_category"]
        lines.append(f"| {record['study']} | {record['stage']} | {record['method']} | {m['n']} | "
                     f"{percent(m['balanced_accuracy'])} | {m['invalid_count']} | "
                     f"{percent(categories['sql_injection']['balanced_accuracy'])} | "
                     f"{percent(categories['access_control']['balanced_accuracy'])} |")
    best = report["best_observed_confirmation"]
    if best:
        lines += ["", f"Best observed prospective score is **{percent(best['metrics']['balanced_accuracy'])}** "
                  f"in `{best['study']}` on {best['metrics']['n']} cases.", "",
                  "The strict above-75% target " + ("was met on this feasibility check." if report["threshold_exceeded"] else "remains unmet.")]
        lo, hi = best["family_cluster_ci95"]
        point = "The point estimate meets the goal" if report["threshold_exceeded"] else "The point estimate does not meet the goal"
        lines += ["", f"The 95% family-bootstrap interval is {percent(lo)} to {percent(hi)} across "
                  f"{best['bootstrap']['families']} families. {point}; the interval does not establish that population accuracy exceeds 75%."]
    lines += ["", *["- " + limitation for limitation in report["limitations"]], "",
              "The JSON companion records exact manifest hashes, model adapter hashes, evidence paths and recomputed counts."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/aya-beam-20261107")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/AYA_PROGRESS.md")
    args = parser.parse_args()
    report = compile_progress(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    atomic_json(args.output.with_suffix(".json"), report)
    print(f"Verified {len(report['records'])} scored arms. Above 75%: {report['threshold_exceeded']}")
