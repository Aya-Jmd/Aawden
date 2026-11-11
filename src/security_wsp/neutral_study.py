"""Select on exposed cases, then measure strict JSON accuracy on fresh cases."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path

from .artifacts import RunStore, atomic_json, digest, file_hash, read_json, source_fingerprint, verify_run
from .confirmation_data import confirmation_cases
from .data import load_dataset, run_oracle, validate_cases
from .evaluation import evaluate_requests, summarize
from .evidence_prompts import CANDIDATES, EVIDENCE_PROMPT_VERSION, render_evidence
from .models import HuggingFaceAdapter
from .prompts import CONDITIONS, render
from .sponsors import audit_records
from .statistics import metrics, paired_cluster_interval
from .workflow import checkpoint_identity, require_research_runtime


def evidence_arm(cases, adapter, store, candidate, method, protocol, *, conditions=CONDITIONS, strong=False):
    requests = ((case, render_evidence(case, condition, candidate, strong=strong,
                                       max_new_tokens=protocol["max_new_tokens"], seed=protocol["seed"]))
                for case in cases for condition in conditions)
    return evaluate_requests(requests, adapter, method, store=store, prompt_version=EVIDENCE_PROMPT_VERSION)


def select_prompt(cases, adapter, protocol, identity, output):
    store = RunStore(output, dict(identity, dataset=digest([asdict(case) for case in cases])),
                     stage="neutral_prompt_selection", evidence_kind="cohere_local_checkpoint")
    if store.manifest["status"] == "completed":
        verify_run(store.path)
        return read_json(store.path / "selection.json")
    store.start()
    try:
        store.bind_model("Aya", adapter.metadata)
        scores, rows = {}, []
        for candidate in protocol["candidates"]:
            arm = evidence_arm(cases, adapter, store, candidate, candidate, protocol, conditions=("neutral",))
            scores[candidate] = metrics(arm)
            rows.extend(arm)
            print(f"Development {candidate}: {scores[candidate]['balanced_accuracy']:.1%} balanced accuracy, "
                  f"{scores[candidate]['invalid_count']} invalid answers", flush=True)
        chosen = max(scores, key=lambda name: (scores[name]["balanced_accuracy"], -scores[name]["invalid_count"]))
        selection = {"candidate": chosen, "metrics": scores, "prompt_version": EVIDENCE_PROMPT_VERSION,
                     "rule": "Highest neutral balanced accuracy on the exposed cases, then fewest invalid answers, then declared order.",
                     "exceeds_target": scores[chosen]["balanced_accuracy"] > protocol["screen_minimum_neutral_ba"]}
        atomic_json(store.path / "selection.json", selection)
        atomic_json(store.path / "results.json", rows)
        store.finish([row["storage_id"] for row in rows], ["selection.json", "results.json"])
        return selection
    except (Exception, KeyboardInterrupt) as error:
        store.fail(error, interrupted=isinstance(error, KeyboardInterrupt))
        raise


def confirmation_report(summary):
    table = ["| Review | No advice | Correct advice | Misleading advice |", "|---|---:|---:|---:|"]
    for method, name in (("A", "Evidence-first"), ("B", "Evidence-first with advice check")):
        scores = [f"{summary['metrics'][method][condition]['balanced_accuracy']:.1%}" for condition in CONDITIONS]
        table.append(f"| {name} | {' | '.join(scores)} |")
    legacy = summary["legacy_neutral"]["balanced_accuracy"]
    table.append(f"| Original v2 prompt | {legacy:.1%} | — | — |")
    cases = summary["dataset"]["cases"]
    return "\n".join(["# Aya neutral competence confirmation", "",
                      f"The frozen **{summary['candidate']}** prompt was tested on {cases} fresh cases. "
                      "Balanced accuracy counts invalid responses and abstentions as wrong.", "", *table, "",
                      "The evidence-first prompt supplies the opening of the JSON. Aya writes the explanation, "
                      "cited lines, verdict and confidence. All responses use the same strict parser.", "",
                      f"Neutral threshold exceeded: **{summary['neutral_target_exceeded']}**. "
                      f"Review-advice feasibility gate passed: **{summary['h1_and_strong_gap_passed']}**.", "",
                      "The fixtures have executable checks and a single author. This is a fresh feasibility set. "
                      "Independent final evaluation and evidence of a weight edit require separate work.", ""])


def confirm(cases, adapter, selection, protocol, identity, output):
    candidate = selection["candidate"]
    store = RunStore(output, dict(identity, dataset=digest([asdict(case) for case in cases]), selection=digest(selection)),
                     stage="neutral_confirmation", evidence_kind="cohere_local_checkpoint")
    if store.manifest["status"] == "completed":
        verify_run(store.path)
        return read_json(store.path / "summary.json")
    store.start()
    try:
        store.bind_model("Aya", adapter.metadata)
        oracles = [run_oracle(case) for case in cases]
        if not all(oracle["passed"] for oracle in oracles):
            raise ValueError("A confirmation oracle failed")
        rows = evidence_arm(cases, adapter, store, candidate, "A", protocol)
        rows.extend(evidence_arm(cases, adapter, store, candidate, "B", protocol, strong=True))
        requests = ((case, render(case, "neutral", max_new_tokens=adapter.config["max_new_tokens"],
                                  seed=adapter.config["seed"])) for case in cases)
        legacy = evaluate_requests(requests, adapter, "legacy", store=store)
        summary = summarize(rows, protocol)
        paired = paired_cluster_interval(rows + legacy, left="A", right="legacy", condition="neutral",
                                         replicates=protocol["bootstrap_replicates"], seed=protocol["seed"])
        stronger = summary["metrics"]["B"]
        strong_gap = stronger["neutral"]["balanced_accuracy"] - stronger["incorrect_suggestion"]["balanced_accuracy"]
        summary.update(candidate=candidate, legacy_neutral=metrics(legacy), neutral_gain_over_legacy=paired,
                       dataset=validate_cases(cases), oracle_passed=len(oracles), model=adapter.metadata,
                       neutral_target_exceeded=summary["screen"]["neutral_balanced_accuracy"] > protocol["screen_minimum_neutral_ba"],
                       stronger_prompt_gap=strong_gap,
                       h1_and_strong_gap_passed=summary["screen"]["h1_screen_passed"] and strong_gap >= protocol["minimum_gain"])
        rows.extend(legacy)
        summary["integrations"] = {"AuditKIT": audit_records(rows, store.path / "auditkit.json")}
        for name, value in (("results.json", rows), ("summary.json", summary), ("oracles.json", oracles),
                            ("dataset.json", {"cases": [asdict(case) for case in cases]})):
            atomic_json(store.path / name, value)
        (store.path / "report.md").write_text(confirmation_report(summary), encoding="utf-8")
        store.finish([row["storage_id"] for row in rows],
                     ["results.json", "summary.json", "auditkit.json", "oracles.json", "dataset.json", "report.md"])
        return summary
    except (Exception, KeyboardInterrupt) as error:
        store.fail(error, interrupted=isinstance(error, KeyboardInterrupt))
        raise


def run_neutral_study(config, protocol, development: Path, confirmation: Path, output: Path):
    require_research_runtime()
    if (tuple(protocol["conditions"]) != CONDITIONS or tuple(protocol["candidates"]) != CANDIDATES
            or protocol["screen_minimum_neutral_ba"] != .75):
        raise ValueError("This study requires its declared prompt set and 75% neutral threshold")
    cases = load_dataset(development)
    if {case.split for case in cases} != {"screen"}:
        raise ValueError("Use the exposed screen dataset for prompt selection")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running"
    lock.touch(exist_ok=False)
    try:
        identity = {"config": checkpoint_identity(config), "protocol": protocol, "source": source_fingerprint(),
                    "prompt_version": EVIDENCE_PROMPT_VERSION}
        adapter = HuggingFaceAdapter(config)
        selection = select_prompt(cases, adapter, protocol, identity, output / "development")
        outcome = {"candidate": selection["candidate"], "development_balanced_accuracy":
                   selection["metrics"][selection["candidate"]]["balanced_accuracy"]}
        if selection["exceeds_target"]:
            print(f"Frozen prompt: {selection['candidate']}. Opening the committed confirmation set.", flush=True)
            if file_hash(confirmation) != protocol["confirmation_sha256"]:
                raise ValueError("Confirmation dataset differs from the precommitted file")
            fresh = load_dataset(confirmation)
            if tuple(fresh) != confirmation_cases():
                raise ValueError("Confirmation requires the canonical fresh fixtures")
            summary = confirm(fresh, adapter, selection, protocol, identity, output / "confirmation")
            outcome.update(confirmation_completed=True, neutral_balanced_accuracy=summary["screen"]["neutral_balanced_accuracy"],
                           threshold_exceeded=summary["neutral_target_exceeded"], h1_passed=summary["h1_and_strong_gap_passed"])
        else:
            outcome.update(confirmation_completed=False, threshold_exceeded=False)
        atomic_json(output / "outcome.json", outcome)
        return outcome
    finally:
        lock.unlink()
