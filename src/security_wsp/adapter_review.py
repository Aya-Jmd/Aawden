"""Low-cost prompt selection for a frozen, hash-verified Aya adapter."""
from dataclasses import asdict, replace
from pathlib import Path

from .artifacts import RunStore, atomic_json, digest, file_hash, read_json, source_fingerprint, verify_run
from .data import load_dataset, run_oracle, validate_cases
from .evaluation import evaluate_requests
from .evidence_prompts import TRACE, COUNTEREXAMPLE, EXAMPLES
from .models import HuggingFaceAdapter
from .prompts import render
from .sponsors import audit_records
from .statistics import metrics
from .workflow import require_research_runtime

VARIANTS = ("plain", "trace", "counterexample", "contrasts")
VERSION = "aawden-adapter-review-v1"


def request_for(case, variant):
    request = render(case, "neutral", numbered_source=True, max_new_tokens=192)
    messages = [dict(message) for message in request.messages]
    if variant != "plain":
        messages[0]["content"] += " " + TRACE[case.target_category]
    if variant == "counterexample":
        messages[0]["content"] += COUNTEREXAMPLE[case.target_category]
    if variant == "contrasts":
        messages[0]["content"] += EXAMPLES[case.target_category]
    return replace(request, messages=messages)


def prior_plain(previous, adapter, cases, variant="plain"):
    verify_run(previous)
    summary = read_json(previous / "summary.json")
    model = summary["model"]
    if model["adapter_sha256"] != adapter.metadata["adapter_sha256"]:
        raise ValueError("Previous answers used different adapter weights")
    if model.get("gpu", {}).get("name") != adapter.metadata.get("gpu", {}).get("name"):
        return None
    if model["environment"]["packages"]["torch"] != adapter.metadata["environment"]["packages"]["torch"]:
        return None
    rows = read_json(previous / "results.json")
    by_id = {row["case_id"]: row for row in rows}
    if set(by_id) != {case.case_id for case in cases} or len(rows) != len(cases):
        raise ValueError("Prior answers cover different cases")
    for case in cases:
        if by_id[case.case_id]["messages"] != request_for(case, variant).messages:
            raise ValueError("Prior answers used a different prompt")
    return rows


def evaluate(cases, adapter, variants, output, identity, previous=None, *,
             make_request=request_for, prompt_version=VERSION):
    store = RunStore(output, dict(identity, dataset=digest([asdict(case) for case in cases]), variants=list(variants)),
                     stage="adapter_prompt_review", evidence_kind="cohere_local_checkpoint")
    store.start()
    try:
        store.bind_model("Aya", adapter.metadata)
        reused = prior_plain(previous, adapter, cases) if previous else None
        rows, scores = [], {}
        for variant in variants:
            if variant == "plain" and reused is not None:
                arm = []
                for original in reused:
                    storage_id = digest([variant, original["request_id"]])[:24]
                    row = dict(original, method=variant, storage_id=storage_id)
                    store.save_prediction(storage_id, row)
                    arm.append(row)
            else:
                arm = evaluate_requests(((case, make_request(case, variant)) for case in cases), adapter, variant,
                                         store=store, prompt_version=prompt_version)
            scores[variant] = metrics(arm)
            rows.extend(arm)
            print(output.name, variant, scores[variant]["balanced_accuracy"], flush=True)
        summary = {"metrics": scores, "model": adapter.metadata, "dataset": validate_cases(cases),
                   "reused_answers": len(reused) if reused is not None else 0,
                   "AuditKIT": audit_records(rows, output / "auditkit.json")}
        atomic_json(output / "results.json", rows)
        atomic_json(output / "summary.json", summary)
        store.finish([row["storage_id"] for row in rows], ["results.json", "summary.json", "auditkit.json"])
        return summary
    except (Exception, KeyboardInterrupt) as error:
        store.fail(error, interrupted=isinstance(error, KeyboardInterrupt))
        raise


def run_adapter_review(config, protocol, checkpoint, previous, output):
    require_research_runtime()
    if protocol["variants"] != list(VARIANTS) or protocol["threshold"] != .75:
        raise ValueError("The preregistered variants and 75% threshold are required")
    output.mkdir(parents=True, exist_ok=False)
    development = Path(protocol["development"])
    if file_hash(development) != protocol["development_sha256"]:
        raise ValueError("Development data changed")
    cases = load_dataset(development)
    specification = dict(protocol["adapter"], checkpoint=str(checkpoint.resolve()))
    adapter = HuggingFaceAdapter(dict(config, lora_adapter=specification))
    identity = {"protocol": protocol, "prompt_version": VERSION, "source": source_fingerprint(),
                "previous_manifest_sha256": file_hash(previous / "manifest.json")}
    selection = evaluate(cases, adapter, VARIANTS, output / "development", identity, previous)
    selected = max(VARIANTS, key=lambda variant: (selection["metrics"][variant]["balanced_accuracy"],
                                                 -selection["metrics"][variant]["invalid_count"]))
    outcome = {"selected_prompt": selected, "validation_balanced_accuracy":
               selection["metrics"][selected]["balanced_accuracy"], "threshold_exceeded": False}
    atomic_json(output / "selection.json", outcome)
    if outcome["validation_balanced_accuracy"] > protocol["threshold"]:
        atomic_json(output / "frozen.json", {"adapter_sha256": specification["sha256"], "selected_prompt": selected,
                                            "protocol": protocol, "source": source_fingerprint()})
        confirmation = Path(protocol["confirmation"])
        if file_hash(confirmation) != protocol["confirmation_sha256"]:
            raise ValueError("The precommitted confirmation file changed")
        fresh = load_dataset(confirmation)
        if not all(run_oracle(case)["passed"] for case in fresh):
            raise ValueError("A fresh oracle failed")
        result = evaluate(fresh, adapter, (selected,), output / "confirmation", identity)
        score = result["metrics"][selected]["balanced_accuracy"]
        outcome.update(confirmation_balanced_accuracy=score, threshold_exceeded=score > protocol["threshold"])
    atomic_json(output / "outcome.json", outcome)
    return outcome
