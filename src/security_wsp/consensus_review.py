"""A frozen second-opinion rule over strict, independently generated Aya answers."""
from .adapter_review import evaluate
from .artifacts import RunStore, atomic_json, file_hash, read_json, source_fingerprint, verify_run
from .data import CLEAN, VULNERABLE, validate_cases
from .evaluation import score_generation
from .models import GenerationResult
from .rigorous_review import checked_dataset, error_report, rigorous_request
from .sponsors import audit_records


VERSION = "aawden-rigorous-review-v10"
RULE = "Use the policy answer unless it reports a bug and a valid compact answer reports clean."


def select_answers(primary, second):
    by_id = {row["case_id"]: row for row in second}
    if len(by_id) != len(second):
        raise ValueError("Duplicate second opinions")
    selected = []
    for row in primary:
        alternate = by_id.get(row["case_id"])
        if row["predicted_verdict"] == VULNERABLE and alternate and alternate["predicted_verdict"] == CLEAN:
            row = alternate
        selected.append(row)
    return selected


def read_answers(path, cases, method, version):
    verify_run(path)
    rows = read_json(path / "results.json")
    by_id = {row["case_id"]: row for row in rows}
    if len(rows) != len(cases) or set(by_id) != {case.case_id for case in cases}:
        raise ValueError("Answer coverage differs from the frozen cases")
    for case in cases:
        row = by_id[case.case_id]
        request = rigorous_request(case, method)
        output = GenerationResult(request.request_id, row["raw_output"], row["input_tokens"], row["output_tokens"],
                                  row["finish_reason"], row["generation_error"], row["latency_seconds"])
        rescored = score_generation(case, request, output, method, prompt_version=version)
        if dict(rescored, storage_id=row["storage_id"]) != row:
            raise ValueError("The original prompt or model output differs")
    return rows


def validate_reference(root, protocol, cases, adapter):
    spec = protocol["validation_reference"]
    path = root / spec["path"]
    if file_hash(path / "manifest.json") != spec["manifest_sha256"]:
        raise ValueError("The saved development manifest changed")
    verify_run(path)
    model = read_json(path / "summary.json")["model"]
    if (model["adapter_sha256"] != adapter.metadata["adapter_sha256"] or
            model["checkpoint_hashes"] != adapter.metadata["checkpoint_hashes"]):
        raise ValueError("Development reference used different weights")
    rows = read_json(path / "results.json")
    by_case = {case.case_id: case for case in cases}
    expected = {(case.case_id, method) for case in cases for method in ("policy", "compact")}
    if len(rows) != len(expected) or {(row["case_id"], row["method"]) for row in rows} != expected:
        raise ValueError("Development reference coverage differs")
    for row in rows:
        case = by_case[row["case_id"]]
        if row["messages"] != rigorous_request(case, row["method"]).messages or row["oracle_verdict"] != case.oracle_verdict:
            raise ValueError("Development prompts or labels differ")
    return error_report(select_answers([row for row in rows if row["method"] == "policy"],
                                       [row for row in rows if row["method"] == "compact"]))


def consensus_study(root, adapter, protocol, output):
    if (protocol["adapter_sha256"] != adapter.metadata["adapter_sha256"]
            or protocol["decision_rule"] != RULE or protocol["target_accuracy"] != 1.0):
        raise ValueError("Unexpected model or decision rule")
    cases = checked_dataset(root, protocol["development"]) + checked_dataset(root, protocol["regression"])
    development = validate_reference(root, protocol, cases, adapter)
    if development["overall"]["accuracy"] != 1.0:
        raise ValueError("The saved development answers did not meet the consensus gate")
    output.mkdir(parents=True, exist_ok=False)
    identity = {"protocol": protocol, "source": source_fingerprint(), "decision_rule": RULE}
    atomic_json(output / "frozen.json", dict(identity, adapter_sha256=protocol["adapter_sha256"],
                confirmation_sha256=protocol["confirmation"]["sha256"], selected_prompt="second_opinion"))
    fresh = checked_dataset(root, protocol["confirmation"])
    validate_cases(cases + fresh)
    evaluate(fresh, adapter, ("policy",), output / "primary", identity,
             make_request=rigorous_request, prompt_version=VERSION)
    primary = read_json(output / "primary/results.json")
    flagged = {row["case_id"] for row in primary if row["predicted_verdict"] == VULNERABLE}
    second = []
    if flagged:
        evaluate([case for case in fresh if case.case_id in flagged], adapter, ("compact",), output / "second", identity,
                 make_request=rigorous_request, prompt_version=VERSION)
        second = read_json(output / "second/results.json")
    selected = select_answers(primary, second)
    report = error_report(selected)
    destination = output / "confirmation"
    store = RunStore(destination, identity, stage="second_opinion_review", evidence_kind="cohere_local_checkpoint")
    store.start()
    store.bind_model("Aya", adapter.metadata)
    for row in selected:
        store.save_prediction(row["storage_id"], row)
    atomic_json(destination / "results.json", selected)
    audit = audit_records(selected, destination / "auditkit.json")
    atomic_json(destination / "summary.json", {"model": adapter.metadata, "report": report, "AuditKIT": audit})
    store.finish([row["storage_id"] for row in selected], ["results.json", "summary.json", "auditkit.json"])
    outcome = {"selected_prompt": "second_opinion", "development": development, "confirmation": report,
               "second_opinions": len(second), "primary_confirmation": error_report(primary),
               "target_reached": report["overall"]["accuracy"] == 1.0}
    atomic_json(output / "outcome.json", outcome)
    return outcome


def verify_consensus(root, output, protocol):
    frozen = read_json(output / "frozen.json")
    if frozen["protocol"] != protocol or frozen["decision_rule"] != RULE:
        raise ValueError("Frozen consensus identity differs")
    fresh = checked_dataset(root, protocol["confirmation"])
    primary = read_answers(output / "primary", fresh, "policy", VERSION)
    flagged = {row["case_id"] for row in primary if row["predicted_verdict"] == VULNERABLE}
    second = read_answers(output / "second", [case for case in fresh if case.case_id in flagged], "compact", VERSION) if flagged else []
    selected = select_answers(primary, second)
    result = verify_run(output / "confirmation")
    report = error_report(selected)
    outcome = read_json(output / "outcome.json")
    stages = ("primary", "second", "confirmation") if flagged else ("primary", "confirmation")
    if any(read_json(output / stage / "summary.json")["model"]["adapter_sha256"] != protocol["adapter_sha256"]
           for stage in stages):
        raise ValueError("The answer stages used different adapter weights")
    if (selected != read_json(output / "confirmation/results.json") or report != outcome["confirmation"]
            or outcome["target_reached"] != (report["overall"]["accuracy"] == 1.0)):
        raise ValueError("Consensus result differs from the strict raw model answers")
    return dict(result, cases=len(selected), accuracy=report["overall"]["accuracy"],
                errors=report["errors"], target_reached=outcome["target_reached"])
