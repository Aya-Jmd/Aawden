"""Replay a completed baseline after removing an enclosing Markdown JSON fence."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from security_wsp.artifacts import file_hash, read_json, verify_run
from security_wsp.evaluation import summarize
from security_wsp.parsing import parse_response
from security_wsp.statistics import paired_cluster_interval

JSON_FENCE = re.compile(r"\A```json[ \t]*\r?\n(?P<body>.+?)\r?\n```\Z", re.DOTALL)


def diagnose(run: Path) -> dict:
    verification = verify_run(run)
    manifest = read_json(run / "manifest.json")
    original = read_json(run / "summary.json")
    if manifest["stage"] != "screen" or set(original["metrics"]) != {"A", "B"}:
        raise ValueError("A completed screen of the original model with both prompts is required")

    rows = read_json(run / "results.json")
    fenced = 0
    for row in rows:
        match = JSON_FENCE.fullmatch(row["raw_output"].strip())
        if not match:
            continue
        fenced += 1
        if row["generation_error"]:
            continue
        parsed, error = parse_response(match["body"], category=row["category"], line_count=row["source_line_count"])
        verdict = parsed["verdict"] if parsed else None
        row.update(parsed=parsed, parse_error=error, predicted_verdict=verdict, correct=verdict == row["oracle_verdict"])

    protocol = manifest["identity"]["protocol"]
    outcome = summarize(rows, protocol)
    intervals = {}
    for method in ("A", "B"):
        paired = [dict(row, method=row["condition"], condition="paired") for row in rows
                  if row["method"] == method and row["condition"] in ("neutral", "incorrect_suggestion")]
        intervals[method] = paired_cluster_interval(paired, left="neutral", right="incorrect_suggestion",
                                                   condition="paired", replicates=protocol["bootstrap_replicates"],
                                                   seed=protocol["seed"])
    return {"evidence_kind": "post_hoc_format_diagnostic",
            "scope": "Remove only an enclosing json code fence and apply the original strict parser. The sealed baseline governs the decision to proceed.",
            "run_id": verification["run_id"], "baseline_manifest_sha256": file_hash(run / "manifest.json"),
            "baseline_protocol": protocol["version"], "answers": len(rows), "fenced_answers": fenced,
            "valid_after_unwrapping": sum(row["parse_error"] is None for row in rows),
            "original_screen": original["screen"], "diagnostic_screen": outcome["screen"],
            "metrics": outcome["metrics"], "flip_rates": outcome["flip_rates"],
            "paired_neutral_minus_misleading": intervals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run), indent=2, allow_nan=False))
