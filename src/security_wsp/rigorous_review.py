"""Strict error accounting and independent holdout evaluation for the Aya adapter."""
from dataclasses import replace
import json

from .adapter_review import evaluate, request_for
from .artifacts import atomic_json, file_hash, read_json, source_fingerprint, verify_run
from .data import CLEAN, VULNERABLE, load_dataset, oracle_cases, run_oracle, validate_cases
from .evidence_prompts import render_evidence
from .policy_notes import compact_projection, counterfactual_notes, policy_notes
from .source_notes import source_notes
from .statistics import metrics
from .symbolic_review import flow_notes


VERSION = "aawden-rigorous-review-v1"
VARIANTS = ("trace", "walkthrough", "evidence_first")
MAP_VARIANTS = ("source_map", "source_map_first")
MAP_VERSION = "aawden-rigorous-review-v2"
FLOW_VARIANTS = ("flow", "flow_first")
FLOW_VERSION = "aawden-rigorous-review-v3"
POLICY_VARIANTS = ("policy", "policy_first")
POLICY_VERSION = "aawden-rigorous-review-v4"
CHECK_VARIANTS = ("check",)
CHECK_VERSION = "aawden-rigorous-review-v5"
SCENARIO_VARIANTS = ("policy", "scenarios")
SCENARIO_VERSION = "aawden-rigorous-review-v6"
COMPACT_VARIANTS = ("policy", "compact")
COMPACT_VERSION = "aawden-rigorous-review-v8"
WALKTHROUGH = {
    "sql_injection": (
        "Pick a username containing a single quote. Trace which branch handles it and what query text and "
        "arguments reach execute. Inspect dead alternatives, reassignment and escaping in the order they run. "
        "A bind argument does not sanitize text that was already interpolated elsewhere in the query. "
        "For SQLite, two adjacent single quotes inside a literal represent one literal quote. "
        "Do not confuse formatting of trusted constants with interpolation of the user input."
    ),
    "access_control": (
        "Mentally execute the code with a logged-in non-owner from the same tenant, using an existing record ID. "
        "Also try a non-owner outside the tenant and a legitimate owner. Substitute the resulting booleans "
        "into each guard in execution order, including any/all, and/or, negation and indices into returned values. "
        "Track which record is checked, later assignments, and every early return. Return vulnerable only if an "
        "unauthorized actor can reach a return of the requested data. Under an owner-only policy, a correctly "
        "enforced ownership check alone is enough."
    ),
}


def rigorous_request(case, variant):
    if variant not in (*VARIANTS, *MAP_VARIANTS, *FLOW_VARIANTS, *POLICY_VARIANTS, *SCENARIO_VARIANTS, *COMPACT_VARIANTS):
        raise ValueError("Unknown review variant")
    request = request_for(case, "trace")
    if variant == "trace":
        return request
    guidance = WALKTHROUGH[case.target_category]
    if variant == "evidence_first":
        guidance += (
            " In the JSON put explanation first and verdict last. In explanation, describe the decisive input, "
            "intermediate value or branch, and executed outcome in two short sentences before choosing the verdict. "
            "Only cite lines that support this outcome."
        )
    if variant in ("source_map_first", "flow_first", "policy_first"):
        request = render_evidence(case, "neutral", "trace", max_new_tokens=256)
    system, user, *prefill = request.messages
    if variant in MAP_VARIANTS:
        guidance += (
            " The syntax notes are derived from the source and may omit paths. Verify the specific "
            "input-to-output behavior directly. In the explanation, name the decisive operator or "
            "argument and why that behavior does or does not violate the stated policy."
        )
        user = dict(user, content=user["content"] + "\n\nPython syntax notes\n" + source_notes(case.visible()["code"]))
    if variant in FLOW_VARIANTS:
        guidance += (
            " The supplemental paths substitute local assignments from the displayed code. They are static "
            "expressions, with external values left unknown. Verify any omitted branches or calls against the "
            "original source. Use the actual query and bindings, or the guard governing the returned row."
        )
        user = dict(user, content=user["content"] + "\n\nStatic return paths\n" + flow_notes(case.visible()["code"]))
    if variant in POLICY_VARIANTS:
        guidance += (
            " Use the symbolic path notes to trace the policy or executed query. Other input values in the "
            "notes remain unknown. Verify the notes against the original code, including any paths or calls "
            "the notes omit. The explanation should identify the relevant branch and its consequence."
        )
        user = dict(user, content=user["content"] + "\n\n" + policy_notes(case))
    if variant == "scenarios":
        guidance += (
            " The hypothetical paths substitute only the stated input assumptions into Python expressions. "
            "Use them to trace the requested row or the query text. Verify paths omitted from the notes "
            "against the original code before deciding."
        )
        threat, source = user["content"].split("<source>", 1)
        user = dict(user, content=threat + "\nSymbolic paths under hypothetical inputs\n" +
                    counterfactual_notes(case) + "\n\n<source>" + source)
    if variant == "compact":
        projection = compact_projection(case)
        if projection:
            threat, _ = user["content"].split("<source>", 1)
            guidance += (" Review the projected source expressions below. Their line numbers refer to the "
                         "original code. Trace each path and match the verdict to the stated rule.")
            user = dict(user, content=threat + "\n<source>\n" + projection + "\n</source>\n"
                        "Give your security assessment using the original line numbers shown.")
        else:
            return rigorous_request(case, "scenarios")
    return replace(request, messages=[dict(system, content=system["content"] + " " + guidance), user, *prefill],
                   max_new_tokens=256 if prefill else 224)


def checked_dataset(root, spec):
    path = root / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError("A preregistered dataset changed")
    cases = load_dataset(path)
    known = oracle_cases()
    if any(known.get(case.case_id) != case for case in cases):
        raise ValueError("Dataset must match the original authored fixtures")
    if len(cases) != spec["count"] or not all(run_oracle(case)["passed"] for case in cases):
        raise ValueError("Dataset count or behavioral oracle check failed")
    return cases


def error_report(rows):
    by_category = {name: metrics([row for row in rows if row["category"] == name])
                   for name in sorted({row["category"] for row in rows})}
    errors = []
    for row in rows:
        if row["correct"]:
            continue
        predicted = row["predicted_verdict"]
        if predicted == CLEAN:
            kind = "missed_bug"
        elif predicted == VULNERABLE:
            kind = "false_alarm"
        elif row["generation_error"]:
            kind = "generation_error"
        elif row["parse_error"]:
            kind = "invalid_output"
        else:
            kind = "abstention"
        errors.append({key: row[key] for key in ("case_id", "family_id", "category", "oracle_verdict",
                                                  "predicted_verdict", "storage_id")} | {"kind": kind})
    by_family = {}
    for row in rows:
        family = row["family_id"]
        by_family[family] = by_family.get(family, True) and row["correct"]
    return {"overall": metrics(rows), "by_category": by_category, "errors": errors,
            "pairs_correct": sum(by_family.values()), "pairs_total": len(by_family)}


def check_request(case, variant, draft):
    if variant not in CHECK_VARIANTS:
        raise ValueError("Unknown consistency review variant")
    request = rigorous_request(case, "policy")
    if case.target_category == "sql_injection":
        return request
    system, user = request.messages
    system = dict(system, content=system["content"] + (
        " Check the draft trace against the source independently. Choose a verdict that agrees with the behavior "
        "you verified. If every data return enforces the stated owner policy, the verdict is not_vulnerable_for_target. "
        "Use vulnerable when an input or caller can violate the stated rule."
    ))
    explanation = (draft.get("parsed") or {}).get("explanation", "")[:600]
    user = dict(user, content=user["content"] + "\n\nUnverified draft explanation\n" + json.dumps(explanation))
    return replace(request, messages=[system, user])


def draft_requests(path):
    rows = []
    if path.is_dir():
        verify_run(path)
        rows = read_json(path / "results.json")
    drafts = {row["case_id"]: row for row in rows}
    if len(drafts) != len(rows) or any(row["method"] != "policy" for row in rows):
        raise ValueError("Draft stage has unexpected predictions")

    def make_request(case, variant):
        if variant not in CHECK_VARIANTS:
            return rigorous_request(case, variant)
        draft = drafts[case.case_id] if case.target_category == "access_control" else {}
        return check_request(case, variant, draft)

    return make_request


def request_factory(cases, adapter, output, identity, version, selected=None):
    if version != CHECK_VERSION:
        return rigorous_request
    drafts = output.with_name(f"{output.name}_drafts")
    access = [case for case in cases if case.target_category == "access_control"]
    if isinstance(selected, dict) and selected.get("access_control") not in CHECK_VARIANTS:
        access = []
    if access:
        evaluate(access, adapter, ("policy",), drafts, identity, make_request=rigorous_request, prompt_version=version)
    return draft_requests(drafts)


def choose_prompt(cases, adapter, variants, output, identity, version):
    make_request = request_factory(cases, adapter, output, identity, version)
    evaluation = evaluate(cases, adapter, variants, output, identity,
                          make_request=make_request, prompt_version=version)
    scores = evaluation["metrics"]
    rows = read_json(output / "results.json")
    atomic_json(output.parent / f"{output.name}-errors.json",
                {variant: error_report([row for row in rows if row["method"] == variant]) for variant in variants})
    selected = max(variants, key=lambda variant: (scores[variant]["accuracy"], -scores[variant]["invalid_count"]))
    return scores, selected


def choose_by_category(root, spec, validation, adapter, current_rows):
    cases = {case.case_id: case for case in validation}
    prior_rows = []
    if spec is not None:
        reference = root / spec["path"]
        if file_hash(reference / "manifest.json") != spec["manifest_sha256"]:
            raise ValueError("Validation reference manifest changed")
        verify_run(reference)
        model = read_json(reference / "summary.json")["model"]
        if (model["adapter_sha256"] != adapter.metadata["adapter_sha256"] or
                model["checkpoint_hashes"] != adapter.metadata["checkpoint_hashes"]):
            raise ValueError("Validation reference used different weights")
        prior_rows = read_json(reference / "results.json")
        if {row["case_id"] for row in prior_rows} != cases.keys():
            raise ValueError("Validation reference cases changed")
        for row in prior_rows:
            case = cases[row["case_id"]]
            request = rigorous_request(case, row["method"])
            if row["messages"] != request.messages or row["oracle_verdict"] != case.oracle_verdict:
                raise ValueError("Validation reference prompt or label changed")
    rows = current_rows + prior_rows
    categories = sorted({row["category"] for row in rows})
    methods = list(dict.fromkeys(row["method"] for row in rows))
    routing, selected_rows = {}, []
    for category in categories:
        arms = {method: [row for row in rows if row["category"] == category and row["method"] == method]
                for method in methods}
        expected = {case.case_id for case in cases.values() if case.target_category == category}
        if any(len(arm) != len(expected) or {row["case_id"] for row in arm} != expected for arm in arms.values()):
            raise ValueError("Validation arms differ in case coverage")
        routing[category] = max(methods, key=lambda method: (metrics(arms[method])["accuracy"],
                                                            -metrics(arms[method])["invalid_count"]))
        selected_rows.extend(arms[routing[category]])
    return routing, error_report(selected_rows)


def selected_request(case, variant):
    return rigorous_request(case, variant[case.target_category] if isinstance(variant, dict) else variant)


def study(root, adapter, protocol, output):
    version = protocol["prompt_version"]
    variants = {VERSION: VARIANTS, MAP_VERSION: MAP_VARIANTS, FLOW_VERSION: FLOW_VARIANTS,
                POLICY_VERSION: POLICY_VARIANTS, CHECK_VERSION: CHECK_VARIANTS, SCENARIO_VERSION: SCENARIO_VARIANTS,
                COMPACT_VERSION: COMPACT_VARIANTS}.get(version)
    if variants is None or protocol["variants"] != list(variants) or protocol["target_accuracy"] != 1.0:
        raise ValueError("Review prompts differ from the protocol")
    if adapter.metadata["adapter_sha256"] != protocol["adapter_sha256"]:
        raise ValueError("Wrong adapter weights")
    development = checked_dataset(root, protocol["development"])
    prior = checked_dataset(root, protocol["regression"])
    output.mkdir(parents=True, exist_ok=False)
    identity = {"protocol": protocol, "source": source_fingerprint()}
    validation = development + prior
    training = []
    train_first = protocol.get("training", {}).get("before_validation", False)
    if train_first:
        if "validation_reference" in protocol:
            raise ValueError("Training before validation requires fresh validation answers")
        from .rigorous_training import refine
        training = refine(adapter, root, protocol["training"], validation, output)
    scores, selected = choose_prompt(validation, adapter, variants, output / "development", identity, version)
    minimum = protocol.get("minimum_dev_accuracy", 0.0)
    outcome = {"selected_prompt": selected, "development": scores, "confirmation": None, "target_reached": False}
    validation_accuracy = scores[selected]["accuracy"]
    if "validation_reference" in protocol or protocol.get("route_by_category"):
        if "training" in protocol and not train_first:
            raise ValueError("Category routing with refinement requires training before validation")
        selected, routed = choose_by_category(root, protocol.get("validation_reference"), validation, adapter,
                                              read_json(output / "development" / "results.json"))
        outcome.update(selected_prompt=selected, routed_development=routed)
        validation_accuracy = routed["overall"]["accuracy"]
    elif validation_accuracy < minimum and "training" in protocol and not train_first:
        from .rigorous_training import refine
        training = refine(adapter, root, protocol["training"], validation, output)
        scores, selected = choose_prompt(validation, adapter, variants, output / "development_refined", identity, version)
        outcome.update(selected_prompt=selected, development_refined=scores)
        validation_accuracy = scores[selected]["accuracy"]
    if validation_accuracy < minimum:
        outcome["reason"] = "Development score below the preregistered gate"
        atomic_json(output / "outcome.json", outcome)
        return outcome
    frozen = {"selected_prompt": selected, "adapter_sha256": adapter.metadata["adapter_sha256"],
              "confirmation_sha256": protocol["confirmation"]["sha256"], **identity}
    atomic_json(output / "frozen.json", frozen)
    fresh = checked_dataset(root, protocol["confirmation"])
    validate_cases(training + validation + fresh)
    make_request = request_factory(fresh, adapter, output / "confirmation", identity, version, selected)
    if isinstance(selected, dict):
        evaluate(fresh, adapter, ("category_route",), output / "confirmation", identity,
                 make_request=lambda case, _: make_request(case, selected[case.target_category]), prompt_version=version)
    else:
        evaluate(fresh, adapter, (selected,), output / "confirmation", identity,
                 make_request=make_request, prompt_version=version)
    report = error_report(read_json(output / "confirmation" / "results.json"))
    outcome.update(confirmation=report, target_reached=report["overall"]["accuracy"] == 1.0)
    atomic_json(output / "outcome.json", outcome)
    return outcome
