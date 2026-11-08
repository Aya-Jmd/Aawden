"""Model-only execution notes followed by a strict generated security review."""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .adaptation import without_adapter
from .adapter_review import prior_plain, request_for
from .artifacts import RunStore, atomic_json, digest, file_hash, read_json, source_fingerprint
from .data import load_dataset, run_oracle, validate_cases
from .evaluation import evaluate_requests
from .models import HuggingFaceAdapter
from .prompts import GenerationRequest
from .sponsors import audit_records
from .statistics import metrics, paired_cluster_interval
from .workflow import require_research_runtime

VERSION = "aawden-mental-review-v2"
INPUTS = (
    "records = {10: {'owner_id': 'u1', 'tenant_id': 'same', 'members': ['u1'], 'secret': 'owned'}, "
    "20: {'owner_id': 'u2', 'tenant_id': 'same', 'members': ['u2'], 'secret': 'other'}}\n"
    "actor = {'id': 'u1', 'tenant_id': 'same', 'role': 'member', 'authenticated': True}\n"
    "Evaluate only read_record(records, actor, 20), using the complete records dictionary above."
)


def simulation_request(case):
    return GenerationRequest(case.case_id, "neutral", [
        {"role": "system", "content": "Mentally trace the supplied Python function on the exact inputs. Compute its actual return even if unintended. Do not invent checks or apply a security policy absent from the source. Answer only the option letter. You have no execution tool."},
        {"role": "user", "content": case.code + "\n" + INPUTS}], max_new_tokens=1)


def score_simulation(case, adapter):
    request = simulation_request(case)
    scores, input_tokens = [], 0
    started = perf_counter()
    with without_adapter(adapter.model):
        for options in ("A = None; B = a record dictionary", "A = a record dictionary; B = None"):
            messages = [dict(message) for message in request.messages]
            messages[-1]["content"] += "\nOptions " + options + ". Answer:"
            prefix = adapter.chat_prefix(messages)
            values = adapter.score_continuations(prefix, (" A", " B"))
            scores.append({"options": options, "A": values[0], "B": values[1]})
            input_tokens += sum(adapter._tokens(prefix + answer).shape[1] for answer in (" A", " B"))
    margin = ((scores[0]["A"] - scores[0]["B"]) + (scores[1]["B"] - scores[1]["A"])) / 2
    estimate = "None" if margin > 0 else "a record dictionary" if margin < 0 else "uncertain"
    return {"case_id": case.case_id, "messages": request.messages, "request_id": request.request_id,
            "working_note": "Counterbalanced language-model return estimate: " + estimate + ". This estimate is unverified.",
            "sequence_scores": scores, "none_minus_record_margin": margin,
            "input_tokens": input_tokens, "output_tokens": 0, "forward_calls": 4,
            "latency_seconds": perf_counter() - started,
            "model_mode": "original base weights, LoRA disabled; full continuation sequence scores"}


def final_request(case, notes):
    request = request_for(case, "trace")
    messages = [dict(message) for message in request.messages]
    messages[-1]["content"] += ("\nA separate model mentally traced these example inputs. These are unverified working notes. "
                                "Check them against the source before giving the requested JSON.\n" + INPUTS +
                                "\nWorking notes\n" + notes)
    return replace(request, messages=messages)


def evaluate(cases, adapter, output, identity, previous=None, measure_baseline=False):
    store = RunStore(output, dict(identity, dataset=digest([asdict(case) for case in cases])),
                     stage="mental_execution_review", evidence_kind="cohere_local_checkpoint")
    store.start()
    try:
        store.bind_model("Aya", adapter.metadata)
        old = prior_plain(previous, adapter, cases, variant="trace") if previous else None
        old_by_id = {row["case_id"]: row for row in old} if old is not None else {}
        rows, simulations, expected = [], [], []
        reused = 0
        for case in cases:
            if case.target_category == "sql_injection" and case.case_id in old_by_id:
                original = old_by_id[case.case_id]
                storage_id = digest(["mental", original["request_id"]])[:24]
                row = dict(original, method="mental", storage_id=storage_id)
                store.save_prediction(storage_id, row)
                reused += 1
            else:
                request = request_for(case, "trace")
                if case.target_category == "access_control":
                    simulation = simulation_request(case)
                    simulation_id = digest(["simulation", simulation.request_id])[:24]
                    notes = store.load_prediction(simulation_id)
                    if notes is None:
                        notes = score_simulation(case, adapter)
                        store.save_prediction(simulation_id, notes)
                    simulations.append(notes)
                    expected.append(simulation_id)
                    request = final_request(case, notes["working_note"])
                row = evaluate_requests([(case, request)], adapter, "mental", store=store, prompt_version=VERSION)[0]
            rows.append(row)
            expected.append(row["storage_id"])
        baseline = []
        if measure_baseline:
            by_id = {row["case_id"]: row for row in rows}
            for case in cases:
                if case.target_category == "sql_injection":
                    original = by_id[case.case_id]
                    storage_id = digest(["trace", original["request_id"]])[:24]
                    row = dict(original, method="trace", storage_id=storage_id)
                    store.save_prediction(storage_id, row)
                else:
                    row = evaluate_requests([(case, request_for(case, "trace"))], adapter, "trace", store=store)[0]
                baseline.append(row)
                expected.append(row["storage_id"])
        summary = {"metrics": metrics(rows), "model": adapter.metadata, "dataset": validate_cases(cases),
                   "reused_answers": reused, "simulation_calls": len(simulations),
                   "simulation_output_tokens": sum(item["output_tokens"] for item in simulations),
                   "simulation_forward_calls": sum(item["forward_calls"] for item in simulations),
                   "simulation_seconds": sum(item["latency_seconds"] for item in simulations),
                   "pipeline": "Unadapted Aya mentally traces access-control examples; the frozen adapter generates the final strict JSON. SQL uses the frozen trace prompt.",
                   "AuditKIT": audit_records(rows + baseline, output / "auditkit.json")}
        if old is not None:
            baseline = [dict(row, method="trace") for row in old]
            summary["previous_trace_metrics"] = metrics(baseline)
            summary["paired_gain"] = paired_cluster_interval(rows + baseline, left="mental", right="trace", condition="neutral")
        if measure_baseline:
            summary["one_pass_trace_metrics"] = metrics(baseline)
            summary["paired_gain"] = paired_cluster_interval(rows + baseline, left="mental", right="trace", condition="neutral")
            atomic_json(output / "baseline-results.json", baseline)
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "results.json", rows)
        atomic_json(output / "simulations.json", simulations)
        artifacts = ["summary.json", "results.json", "simulations.json", "auditkit.json"]
        if measure_baseline:
            artifacts.append("baseline-results.json")
        store.finish(expected, artifacts)
        print(output.name, summary["metrics"]["balanced_accuracy"], flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        store.fail(error, interrupted=isinstance(error, KeyboardInterrupt))
        raise


def run_mental_review(config, protocol, checkpoint, previous, output):
    require_research_runtime()
    if protocol["threshold"] != .75:
        raise ValueError("The 75% criterion is fixed")
    output.mkdir(parents=True, exist_ok=False)
    development = Path(protocol["development"])
    if file_hash(development) != protocol["development_sha256"]:
        raise ValueError("Development cases changed")
    adapter = HuggingFaceAdapter(dict(config, lora_adapter=dict(protocol["adapter"], checkpoint=str(checkpoint.resolve()))))
    identity = {"protocol": protocol, "source": source_fingerprint(), "prompt_version": VERSION,
                "previous_manifest_sha256": file_hash(previous / "manifest.json")}
    validation = evaluate(load_dataset(development), adapter, output / "development", identity, previous)
    score = validation["metrics"]["balanced_accuracy"]
    outcome = {"validation_balanced_accuracy": score, "threshold_exceeded": False}
    if score > max(.75, validation.get("previous_trace_metrics", {}).get("balanced_accuracy", .75)):
        atomic_json(output / "frozen.json", identity)
        confirmation = Path(protocol["confirmation"])
        if file_hash(confirmation) != protocol["confirmation_sha256"]:
            raise ValueError("Fresh cases changed")
        fresh = load_dataset(confirmation)
        if not all(run_oracle(case)["passed"] for case in fresh):
            raise ValueError("A fresh oracle failed")
        result = evaluate(fresh, adapter, output / "confirmation", identity, measure_baseline=True)
        score = result["metrics"]["balanced_accuracy"]
        outcome.update(confirmation_balanced_accuracy=score, threshold_exceeded=score > .75)
    atomic_json(output / "outcome.json", outcome)
    return outcome
