"""Regression baseline on the three committed regression sets (protocol v3).

Scores one model arm — normally the unedited checkpoint — on coding, general and format, so the
final report can say what an edit cost or preserved. It reuses the exact scoring helpers the frozen
final evaluation uses (``scripts/wsp_final.py``: ``evaluate_general``, ``evaluate_format``, and
``security_wsp.regressions.evaluate_coding``), so a baseline measured here is comparable to the
after-numbers later.

Inputs are verified against ``configs/wsp-regressions.json`` before any generation: a regression set
whose sha256 differs from its commitment is refused rather than scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import RunStore, atomic_json, file_hash, read_json, source_fingerprint  # noqa: E402
from security_wsp.regressions import coding_summary, evaluate_coding, load_coding_cases  # noqa: E402

COMMITMENTS = ROOT / "configs" / "wsp-regressions.json"
FINAL = ROOT / "scripts" / "wsp_final.py"


KIND_PATHS = {"coding": "datasets/wsp-coding.json", "general": "datasets/wsp-general.json",
              "format": "datasets/wsp-format.json"}


def load_commitments(path: Path = COMMITMENTS) -> dict:
    """configs/wsp-regressions.json keys each entry by its repo-relative path, with a "kind" field."""
    document = read_json(path)
    files = document.get("files") or {}
    entries = {}
    for kind, relative in KIND_PATHS.items():
        entry = files.get(relative)
        if entry is None:
            raise SystemExit(f"Refused: {path} has no commitment for {relative}")
        entries[kind] = dict(entry, path=relative)
    return entries


def verify(path: Path, commitment: dict) -> str:
    if not path.is_file():
        raise SystemExit(f"Refused: {path} is missing")
    actual = file_hash(path)
    expected = str(commitment.get("sha256", "")).lower()
    if actual != expected:
        raise SystemExit(f"Refused: {path} sha256 {actual} differs from its commitment {expected}")
    return actual


def load_final_helpers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_wsp_final_helpers", FINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "wsp-v3" / "regressions-original")
    parser.add_argument("--label", default="original")
    parser.add_argument("--limit", type=int, default=None, help="first N items per set (smoke runs)")
    args = parser.parse_args(argv)

    commitments = load_commitments()
    paths = {}
    hashes = {}
    for kind in ("coding", "general", "format"):
        entry = commitments[kind]
        path = ROOT / entry["path"]
        hashes[kind] = verify(path, entry)
        paths[kind] = path

    config = read_json(args.config)
    helpers = load_final_helpers()

    identity = {"config": config, "commitments": hashes, "label": args.label, "limit": args.limit,
                "source": source_fingerprint()}
    store = RunStore(args.output, identity, stage="wsp-v3-regressions", evidence_kind="real_model")
    store.start(max_attempts=3)

    from security_wsp.models import HuggingFaceAdapter
    adapter = HuggingFaceAdapter(config)
    store.bind_model(args.label, adapter.metadata)

    counter = helpers.ForwardCounter()
    arms = {args.label: adapter}
    # The shared helpers index by the names "original"/"edited"; normalise so either label works.
    internal = {"original": adapter, "edited": adapter}

    coding_cases = load_coding_cases(paths["coding"])
    if args.limit:
        coding_cases = coding_cases[: args.limit]
    coding_rows = evaluate_coding(coding_cases, adapter, args.label, store, config)
    # coding_summary() compares two arms and refuses a single-arm call, so a one-arm baseline is
    # summarised here with the same arithmetic it would use. The frozen final evaluation still calls
    # coding_summary for its before/after comparison once both arms exist.
    def single_arm_summary(rows, sha):
        ids = {row["case_id"] for row in rows}
        if not ids or len(ids) != len(rows):
            raise SystemExit("Refused: coding rows are duplicated or empty")
        correct = sum(row["correct"] for row in rows)
        return {"dataset_sha256": sha,
                "arm": args.label,
                "cases": len(ids), "correct": correct, "accuracy": correct / len(ids),
                "invalid": sum(row["error"] is not None for row in rows),
                "scope": ("Exact answers to ordinary Python code questions. No model-generated code "
                          "is executed. Single arm: no difference is computed.")}

    coding = single_arm_summary(coding_rows, hashes["coding"])

    general = helpers.evaluate_general(paths["general"], internal, counter)
    verify_format = helpers._load_registry_verifier()
    format_results = helpers.evaluate_format(paths["format"], internal, counter, verify_format)

    summary = {
        "label": args.label,
        "model": {k: v for k, v in adapter.metadata.items() if k != "checkpoint_hashes"},
        "coding": coding,
        "general": general["arms"]["original"],
        "format": format_results["arms"]["original"],
        "forward": {"calls": counter.calls, "prompts": counter.prompts},
        "dataset_sha256": hashes,
        "note": ("A single arm measured with the same helpers the frozen final evaluation uses. "
                 "Differences are only meaningful once an edited arm is measured identically."),
    }
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "coding-rows.json", coding_rows)
    store.finish([r["storage_id"] for r in coding_rows] if coding_rows and "storage_id" in coding_rows[0] else [],
                 ["summary.json", "coding-rows.json"])
    print(json.dumps(summary, indent=1)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
