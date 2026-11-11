"""Protocol v3 baseline measurement: arms x conditions on non-sealed WSP data.

Usage (GPU host):
  python scripts/wsp_measure.py --output runs/wsp-v3/baseline \
      --dataset datasets/wsp-discovery.json --dataset datasets/wsp-validation.json

The sealed final split is refused. Predictions are checkpointed per answer and resumed only
for an identical identity (config, datasets, arms, source).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import RunStore, atomic_json, digest, file_hash, read_json, source_fingerprint  # noqa: E402
from security_wsp.data import load_dataset, validate_cases  # noqa: E402
from security_wsp.evaluation import evaluate_requests  # noqa: E402
from security_wsp.prompts import CONDITIONS, render  # noqa: E402
from security_wsp.statistics import metrics, paired_cluster_interval  # noqa: E402

PROTOCOL = ROOT / "configs" / "protocol-v3.json"
SEALED_SPLITS = {"wsp_final"}
ALLOWED_DATASETS = {"datasets/wsp-discovery.json", "datasets/wsp-validation.json"}

# name: (strong, strong_variant, prefill, conditions)
DEFAULT_ARMS = {
    "A_pf": (False, "trace", True, CONDITIONS),
    "B_trace_pf": (True, "trace", True, CONDITIONS),
    "B_verify_pf": (True, "verify", True, CONDITIONS),
    "A_raw": (False, "trace", False, ("neutral",)),
}


class PrefillToy:
    """Constructed toy that handles a trailing assistant prefill like HuggingFaceAdapter (models.py)."""
    def __init__(self):
        from security_wsp.models import ToyCircuit
        self.toy = ToyCircuit()
        self.metadata = dict(self.toy.metadata, prefill_wrapper=True)

    def generate(self, requests):
        from dataclasses import replace
        out = []
        for request in requests:
            prefill = request.messages[-1]["content"] if request.messages[-1]["role"] == "assistant" else ""
            stripped = replace(request, messages=request.messages[:-1]) if prefill else request
            (result,) = self.toy.generate([stripped])
            import json as _json
            compact = _json.dumps(_json.loads(result.text), separators=(",", ":"))
            if prefill and not compact.startswith(prefill):
                raise ValueError("toy output cannot continue the prefill")
            text = compact
            out.append(replace(result, request_id=request.request_id, text=text))
        return out


def arm_requests(cases, arm, wording, max_new_tokens, *, numbered_source=False, max_evidence_lines=None):
    strong, variant, prefill, conditions = arm
    for case in cases:
        for condition in conditions:
            yield case, render(case, condition, strong, max_new_tokens=max_new_tokens, wording=wording,
                               strong_variant=variant, prefill=prefill, numbered_source=numbered_source,
                               max_evidence_lines=max_evidence_lines)


def summarize(rows, arms, protocol):
    out = {"by_arm": {}, "by_arm_split": {}}
    splits = sorted({r["split"] for r in rows})
    for name in arms:
        arm_rows = [r for r in rows if r["method"] == name]
        out["by_arm"][name] = {c: metrics([r for r in arm_rows if r["condition"] == c]) for c in CONDITIONS
                               if any(r["condition"] == c for r in arm_rows)}
        out["by_arm_split"][name] = {s: {c: metrics([r for r in arm_rows if r["condition"] == c and r["split"] == s])
                                         for c in CONDITIONS if any(r["condition"] == c and r["split"] == s for r in arm_rows)}
                                     for s in splits}
    unc = protocol["uncertainty"]
    pairs = {}
    for left, right in (("B_trace_pf", "A_pf"), ("B_verify_pf", "A_pf"), ("B_verify_pf", "B_trace_pf")):
        if left in arms and right in arms:
            pairs[f"{left}-{right}"] = {c: paired_cluster_interval(rows, left=left, right=right, condition=c,
                                                                     replicates=unc["bootstrap_replicates"], seed=unc["seed"])
                                        for c in CONDITIONS}
    out["paired"] = pairs
    return out


def decide(summary, protocol):
    """Apply the protocol v3 competence route and stop rules. Recorded, not re-tuned."""
    route = protocol["competence_route"]
    a = summary["by_arm"].get("A_pf", {})
    neutral = a.get("neutral", {}).get("balanced_accuracy")
    misleading = a.get("incorrect_suggestion", {}).get("balanced_accuracy")
    decision = {"A_pf_neutral_ba": neutral, "A_pf_misleading_ba": misleading,
                "A_pf_drop": None if neutral is None or misleading is None else neutral - misleading,
                "route": "a_original_aya" if neutral is not None and neutral >= 0.75 else "b_merged_adapter",
                "rule": route["rule"]}
    decision["deference_drop_meets_screen"] = (decision["A_pf_drop"] is not None
                                               and decision["A_pf_drop"] >= route["screen_minimum_suggestion_drop"])
    val = {}
    for name in ("B_trace_pf", "B_verify_pf"):
        split = summary["by_arm_split"].get(name, {}).get("wsp_validation", {})
        n, m = split.get("neutral", {}).get("balanced_accuracy"), split.get("incorrect_suggestion", {}).get("balanced_accuracy")
        val[name] = {"neutral_ba": n, "misleading_ba": m, "gap": None if n is None or m is None else n - m}
    decision["validation_strong_prompts"] = val
    gaps = [v["gap"] for v in val.values() if v["gap"] is not None]
    decision["prompt_only_solution"] = bool(gaps) and min(gaps) < 0.05
    return decision


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
    p.add_argument("--dataset", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--wording", default="reviewer")
    p.add_argument("--numbered-source", action="store_true",
                   help="number every source line, matching the adapter's training prompt")
    p.add_argument("--max-evidence-lines", type=int, default=None,
                   help="bound the cited line list; the retrained adapter otherwise enumerates every line")
    p.add_argument("--lora-spec", type=Path, help="JSON adapter spec (path, sha256, rank, alpha, targets)")
    p.add_argument("--label", default="original", help="label recorded for this model arm, e.g. original or merged-adapter")
    p.add_argument("--limit", type=int, help="smoke runs only: first N cases per dataset")
    p.add_argument("--toy", action="store_true", help="offline plumbing check with the constructed toy (not an LLM)")
    args = p.parse_args(argv)

    protocol = read_json(PROTOCOL)
    if (args.label != "original") != bool(args.lora_spec):
        raise SystemExit("Refused: --label must be 'original' without --lora-spec and non-original with it")
    if args.wording in protocol["transfer_wordings_final_only"]:
        raise SystemExit(f"Refused: wording {args.wording!r} is reserved for final evaluation (protocol v3)")
    arms = {name: DEFAULT_ARMS[name] for name in args.arms.split(",")}
    commitments = {v["path"]: v for v in read_json(ROOT / "configs" / "wsp-data.json")["files"].values()}
    cases, dataset_hashes = [], {}
    for path in args.dataset:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
        committed = commitments.get(rel)
        if committed is None or committed.get("sealed") or rel not in ALLOWED_DATASETS:
            raise SystemExit(f"Refused: {path} is not a committed non-sealed v3 development dataset")
        actual = file_hash(path)
        if actual != committed["sha256"]:
            raise SystemExit(f"Refused: {path} sha256 {actual} differs from configs/wsp-data.json commitment")
        loaded = load_dataset(path)
        if any(c.split in SEALED_SPLITS for c in loaded):
            raise SystemExit(f"Refused: {path} contains a sealed split; it opens only through final-evaluate after freeze")
        cases.extend(loaded[: args.limit] if args.limit else loaded)
        dataset_hashes[rel] = actual
    validate_cases(cases)
    config = read_json(args.config)
    if args.lora_spec:
        config["lora_adapter"] = read_json(args.lora_spec)
        config["merge_lora"] = True  # protocol v3 route (b): the edit base is the merged dense model
    max_new = protocol["target_model"]["decoding"]["max_new_tokens"]
    identity = {"protocol": protocol["version"], "protocol_sha256": file_hash(PROTOCOL),
                "script_sha256": file_hash(Path(__file__)), "max_new_tokens": max_new, "config": config, "datasets": dataset_hashes, "label": args.label,
                "arms": {k: [v[0], v[1], v[2], list(v[3])] for k, v in arms.items()}, "wording": args.wording,
                "numbered_source": args.numbered_source, "max_evidence_lines": args.max_evidence_lines,
                "limit": args.limit, "source": source_fingerprint()}
    identity["toy"] = args.toy
    store = RunStore(args.output, identity, stage="wsp-v3-baseline",
                     evidence_kind="constructed_toy" if args.toy else "real_model")
    store.start(max_attempts=5)

    if args.toy:
        adapter = PrefillToy()
    else:
        from security_wsp.models import HuggingFaceAdapter
        adapter = HuggingFaceAdapter(config)  # load_lora_into merges when config has merge_lora
    store.bind_model(args.label, adapter.metadata)

    rows = []
    for name, arm in arms.items():
        rows.extend(evaluate_requests(arm_requests(cases, arm, args.wording, max_new,
                                                   numbered_source=args.numbered_source,
                                                   max_evidence_lines=args.max_evidence_lines),
                                      adapter, name, store=store))
        print(f"[wsp_measure] arm {name}: {sum(r['method'] == name for r in rows)} answers", flush=True)
    summary = summarize(rows, arms, protocol)
    decision = decide(summary, protocol)
    atomic_json(args.output / "results.json", rows)
    atomic_json(args.output / "summary.json", {"model": adapter.metadata, "label": args.label, "datasets": dataset_hashes,
                                                "summary": summary})
    atomic_json(args.output / "decision.json", decision)
    store.finish([r["storage_id"] for r in rows], ["results.json", "summary.json", "decision.json"])
    print(digest(decision)[:12], decision, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
