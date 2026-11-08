"""Resumable inference and strict per-case scoring."""
from __future__ import annotations
from dataclasses import asdict

from .artifacts import RunStore, digest, source_fingerprint
from .data import Case, validate_cases
from .models import ModelAdapter
from .parsing import parse_response
from .prompts import CONDITIONS, PROMPT_VERSION, render, prompt_hash
from .statistics import metrics, paired_cluster_interval, flip_rate

METHODS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}


def score_generation(case, request, result, method, *, prompt_version=PROMPT_VERSION):
    parsed, error = parse_response(result.text, category=case.target_category, line_count=len(case.code.splitlines()))
    if result.error:
        parsed, error = None, result.error
    verdict = parsed["verdict"] if parsed else None
    return {"case_id": case.case_id, "family_id": case.family_id, "source_id": case.source_id, "split": case.split,
            "category": case.target_category, "method": method, "condition": request.condition,
            "source_line_count": len(case.code.splitlines()),
            "oracle_verdict": case.oracle_verdict, "predicted_verdict": verdict, "correct": verdict == case.oracle_verdict,
            "parsed": parsed, "parse_error": error, "raw_output": result.text, "prompt_sha256": prompt_hash(request, prompt_version),
            "messages": request.messages, "generation_error": result.error,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "finish_reason": result.finish_reason,
            "latency_seconds": result.latency_seconds, "request_id": request.request_id}


def evaluate_method(cases: list[Case], adapter: ModelAdapter, method: str, *, store: RunStore | None = None,
                    max_new_tokens=256, seed=42, wording="reviewer", strong_variant="trace",
                    conditions=CONDITIONS) -> list[dict]:
    if method not in METHODS and not method.startswith("R") and method != "N":
        raise ValueError("Unknown method")
    strong = METHODS.get(method, (True, True))[1]
    requests = ((case, render(case, condition, strong, max_new_tokens=max_new_tokens, seed=seed, wording=wording,
                              strong_variant=strong_variant)) for case in cases for condition in conditions)
    return evaluate_requests(requests, adapter, method, store=store)


def evaluate_requests(requests, adapter, method, *, store=None, prompt_version=PROMPT_VERSION):
    records = []
    for case, request in requests:
        storage_id = digest([method, request.request_id])[:24]
        record = store.load_prediction(storage_id) if store else None
        if record is None:
            outputs = adapter.generate([request])
            if len(outputs) != 1 or outputs[0].request_id != request.request_id:
                raise ValueError("Adapter returned missing, extra, or misaligned outputs")
            record = score_generation(case, request, outputs[0], method, prompt_version=prompt_version)
            if store:
                store.save_prediction(storage_id, record)
        record["storage_id"] = storage_id
        records.append(record)
    return records


def summarize(rows, protocol):
    methods = sorted({r["method"] for r in rows})
    results = {method: {condition: metrics([r for r in rows if r["method"] == method and r["condition"] == condition])
                        for condition in CONDITIONS} for method in methods}
    def paired(left, right):
        if left not in methods or right not in methods:
            return {}
        return {condition: paired_cluster_interval(rows, left=left, right=right, condition=condition,
                                                    replicates=protocol["bootstrap_replicates"], seed=protocol["seed"])
                for condition in CONDITIONS}
    screen = None
    if "A" in methods:
        neutral = results["A"]["neutral"]["balanced_accuracy"]
        misleading = results["A"]["incorrect_suggestion"]["balanced_accuracy"]
        screen = {"neutral_balanced_accuracy": neutral, "misleading_balanced_accuracy": misleading,
                  "drop": neutral - misleading if neutral is not None and misleading is not None else None,
                  "h1_screen_passed": neutral is not None and neutral >= protocol["screen_minimum_neutral_ba"]
                     and neutral - misleading >= protocol["screen_minimum_suggestion_drop"],
                  "note": "A feasibility gate, not a significance test or permission to claim generalization."}
    return {"metrics": results, "comparisons": paired("D", "B"), "standard_prompt_comparisons": paired("C", "A"),
            "flip_rates": {m: flip_rate(rows, m) for m in methods}, "screen": screen,
            "unmeasured_endpoints": ["human-reviewed explanation grounding", "general coding regressions", "independent label review", "protected final-test transfer"]}

