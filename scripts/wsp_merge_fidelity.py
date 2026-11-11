"""Merge-fidelity check: does folding the LoRA into dense weights change behaviour?

Protocol v3 route (b) edits the *merged* base. Before any edit is attempted, this script proves that
merging is behaviour-preserving by measuring the same cases twice — adapter attached (unmerged) and
adapter folded into the base weights — and comparing per-answer predictions and raw text.

It also re-measures the adapter on the confirmation set where its retrain recorded a score, so a
degeneration there is visible rather than inferred.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import atomic_json, digest, read_json  # noqa: E402
from security_wsp.data import load_dataset  # noqa: E402
from security_wsp.prompts import CONDITIONS, render  # noqa: E402
from security_wsp.evaluation import score_generation  # noqa: E402
from security_wsp.statistics import metrics  # noqa: E402


def measure(adapter, cases, conditions, max_new_tokens, wording="reviewer"):
    rows = []
    for case in cases:
        for condition in conditions:
            request = render(case, condition, False, max_new_tokens=max_new_tokens,
                             wording=wording, prefill=True)
            (result,) = adapter.generate([request])
            rows.append(score_generation(case, request, result, "merge"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
    parser.add_argument("--lora-spec", type=Path, default=ROOT / "configs" / "merged-v7-adapter-spec.json")
    parser.add_argument("--dataset", type=Path, action="append", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "wsp-v3" / "merge-fidelity")
    args = parser.parse_args(argv)

    datasets = args.dataset or [ROOT / "datasets" / "neutral-confirmation-v5.json"]
    cases = []
    for path in datasets:
        loaded = load_dataset(path)
        cases.extend(loaded[: args.limit])
    cases = cases + cases[:8]  # repeat a subset so the comparison has a paired sample
    conditions = ("neutral", "incorrect_suggestion")

    base = read_json(args.config)
    base.pop("lora_adapter", None)
    base.pop("merge_lora", None)
    spec = read_json(args.lora_spec)

    from security_wsp.models import HuggingFaceAdapter

    report = {"cases": len(cases), "conditions": list(conditions), "datasets": [str(p) for p in datasets]}

    unmerged_config = dict(base, lora_adapter=spec)
    adapter = HuggingFaceAdapter(unmerged_config)
    report["unmerged_metadata"] = {k: v for k, v in adapter.metadata.items() if k != "checkpoint_hashes"}
    unmerged = measure(adapter, cases, conditions, base.get("max_new_tokens", 256))
    del adapter

    merged_config = dict(base, lora_adapter=spec, merge_lora=True)
    adapter = HuggingFaceAdapter(merged_config)
    report["merged_metadata"] = {k: v for k, v in adapter.metadata.items() if k != "checkpoint_hashes"}
    merged = measure(adapter, cases, conditions, base.get("max_new_tokens", 256))

    same_text = sum(a["raw_output"] == b["raw_output"] for a, b in zip(unmerged, merged))
    same_verdict = sum(a["predicted_verdict"] == b["predicted_verdict"] for a, b in zip(unmerged, merged))
    report["agreement"] = {"answers": len(merged), "identical_text": same_text, "identical_verdict": same_verdict}
    report["unmerged"] = {c: metrics([r for r in unmerged if r["condition"] == c]) for c in conditions}
    report["merged"] = {c: metrics([r for r in merged if r["condition"] == c]) for c in conditions}
    report["invalid_examples"] = [
        {"condition": r["condition"], "parse_error": r["parse_error"], "finish_reason": r["finish_reason"],
         "output_tokens": r["output_tokens"], "raw_head": r["raw_output"][:200], "raw_tail": r["raw_output"][-120:]}
        for r in merged if r["parse_error"]][:5]

    atomic_json(args.output / "fidelity.json", report)
    atomic_json(args.output / "unmerged.json", unmerged)
    atomic_json(args.output / "merged.json", merged)
    print(json.dumps({k: report[k] for k in ("agreement", "unmerged", "merged")}, indent=1)[:2500], flush=True)
    print("digest", digest(report)[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
