"""Executable demo with deliberately transparent, non-LLM weight edits."""
from __future__ import annotations
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import RunStore, atomic_json, digest, file_hash, read_json, source_fingerprint, utc_now, verify_run
from .data import build_cases, run_oracle, split_by_family, validate_cases
from .evaluation import evaluate_method, summarize
from .interventions import (TrialRegistry, load_toy_checkpoint, project_vector, random_matched_update,
                            write_toy_checkpoint, vector_hash)
from .models import ToyCircuit
from .prompts import PROMPT_VERSION, render
from .reporting import write_report, dashboard_payload
from .statistics import metrics, paired_cluster_interval
from .sponsors import audit_records, toolkit_status


def run_demo(output: str | Path, protocol: dict, *, use_auditkit=True):
    cases = split_by_family(build_cases(), seed=protocol["seed"])
    identity = {"protocol": protocol, "dataset": digest([asdict(c) for c in cases]), "source": source_fingerprint(),
                "prompt_version": PROMPT_VERSION, "model": ToyCircuit().metadata, "use_auditkit": use_auditkit,
                "stage": "educational_demo"}
    store = RunStore(output, identity, stage="educational_demo", evidence_kind="constructed_toy")
    if store.manifest["status"] == "completed":
        verify_run(store.path)
        return read_json(store.path / "summary.json")
    store.start()
    try:
        oracles = [run_oracle(c) for c in cases]
        if not all(row["passed"] for row in oracles):
            raise ValueError("Fixture oracle failed")
        atomic_json(store.path / "oracles.json", oracles)
        atomic_json(store.path / "dataset.json", {"cases": [asdict(c) for c in cases], "metadata": validate_cases(cases)})
        original = ToyCircuit()
        write_toy_checkpoint(store.path / "original.json", original.weights, original.weights, {"type": "unchanged"})
        validation = [c for c in cases if c.split == "validation"]
        baseline_validation = evaluate_method(validation, original, "B")
        baseline_neutral = metrics([r for r in baseline_validation if r["condition"] == "neutral"])["balanced_accuracy"]
        baseline_correct = metrics([r for r in baseline_validation if r["condition"] == "correct_suggestion"])["accuracy"]
        registry = TrialRegistry(store.path / "trials.json", identity, protocol["max_edit_candidates"])
        selection = []
        for strength in (.25, .5, 1.0):
            recipe = {"kind": "toy_projection", "direction": [0, 1], "strength": strength}
            trial = registry.reserve(recipe)
            weights = project_vector(original.weights, [0, 1], strength)
            candidate = ToyCircuit(weights)
            scores = evaluate_method(validation, candidate, "D")
            conditions = {cond: metrics([r for r in scores if r["condition"] == cond]) for cond in protocol["conditions"]}
            eligible = conditions["neutral"]["balanced_accuracy"] >= baseline_neutral-protocol["neutral_noninferiority_margin"]
            eligible &= conditions["correct_suggestion"]["accuracy"] >= baseline_correct-protocol["neutral_noninferiority_margin"]
            path = store.path / "candidates" / (trial["candidate_id"] + ".json")
            edit = write_toy_checkpoint(path, original.weights, weights, recipe)
            registry.complete(trial["candidate_id"], role="targeted", validation=conditions, eligible=eligible,
                              checkpoint=path.name, checkpoint_sha256=file_hash(path), edit=edit)
            atomic_json(store.path / "candidates" / (trial["candidate_id"] + "-validation.json"), scores)
            if eligible:
                selection.append((conditions["incorrect_suggestion"]["balanced_accuracy"], -edit["update_l2"], trial["candidate_id"], weights))
        if not selection:
            raise ValueError("No candidate met validation guardrails")
        _, _, candidate_id, weights = max(selection)
        selected = next(t for t in registry.state["trials"] if t["candidate_id"] == candidate_id)
        checkpoint = write_toy_checkpoint(store.path / "edited.json", original.weights, weights, selected["recipe"])
        freeze = {"candidate_id": candidate_id, "checkpoint_sha256": file_hash(store.path / "edited.json"), "frozen_at": utc_now(),
                  "validation_families": sorted({c.family_id for c in validation}), "protocol_sha256": digest(protocol),
                  "selection_rule": "Maximize validation incorrect-suggestion BA under neutral and correct-suggestion guardrails; break ties by smaller update.",
                  "evidence_kind": "constructed_toy", "final_test_protected": False}
        atomic_json(store.path / "frozen.json", freeze)
        edited = ToyCircuit(load_toy_checkpoint(store.path / "edited.json"))
        rows = []
        for method, adapter in (("A", original), ("B", original), ("C", edited), ("D", edited), ("N", original)):
            rows.extend(evaluate_method(cases, adapter, method, store=store))
        # Random controls match the selected update norm, dimensionality and
        # precision. Their touched/changed count may differ and is disclosed.
        for seed in protocol["random_control_seeds"]:
            recipe = {"kind": "toy_random_matched_norm", "seed": seed, "update_l2": checkpoint["update_l2"]}
            trial = registry.reserve(recipe)
            random_weights = random_matched_update(original.weights, checkpoint["update_l2"], seed)
            edit = write_toy_checkpoint(store.path / "controls" / f"random-{seed}.json", original.weights, random_weights, recipe)
            random_rows = evaluate_method(cases, ToyCircuit(random_weights), f"R{seed}", store=store)
            rows.extend(random_rows)
            registry.complete(trial["candidate_id"], role="random_control", eligible=False, edit=edit,
                              mismatch="Same update L2 and two touched entries; random control can change both values, targeted edit changes only the opinion weight.")
        # Identical generated text must survive an entirely new interpreter.
        requests = [render(c, condition, strong=True) for c in cases[:4] for condition in protocol["conditions"]]
        atomic_json(store.path / "reload-requests.json", [asdict(r) for r in requests])
        expected = edited.generate(requests)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        process = subprocess.run([sys.executable, "-m", "security_wsp.reload_probe", "--kind", "toy",
                                  "--checkpoint", str((store.path / "edited.json").resolve()),
                                  "--requests", str((store.path / "reload-requests.json").resolve()),
                                  "--output", str((store.path / "reload-outputs.json").resolve())],
                                 env=env, capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise ValueError("Fresh-process reload failed: " + process.stderr[-1000:])
        reloaded = read_json(store.path / "reload-outputs.json")
        reload_ok = all(a.text == b["text"] and a.request_id == b["request_id"] for a, b in zip(expected, reloaded)) and len(expected) == len(reloaded)
        if not reload_ok or vector_hash(original.weights) != checkpoint["base_sha256"]:
            raise ValueError("Reload or source-preservation check failed")
        atomic_json(store.path / "reload-verification.json", {"passed": True, "fresh_process": True, "hooks": False,
                                                            "checked_generations": len(expected), "original_unchanged": True})
        atomic_json(store.path / "results.json", rows)
        integrations = {"CircuitKIT": {"status": "real_weight_run_pending",
                                      "scope": "Real residual-contrast, causal-patch and parity implementation provided; the toy does not execute CircuitKIT discovery."}}
        if use_auditkit:
            try:
                integrations["AuditKIT"] = audit_records(rows, store.path / "auditkit.json")
            except ImportError:
                integrations["AuditKIT"] = {"status": "unavailable", "scope": "Run scripts/fetch_toolkits.py or install pinned research dependencies."}
        else:
            integrations["AuditKIT"] = {"status": "not_requested", "scope": "Core-only demo requested."}
        summary = summarize(rows, protocol)
        exploratory = [r for r in rows if r["split"] == "exploratory_test"]
        summary.update(evidence_kind="constructed_toy", run_id=store.manifest["run_id"], created_at=store.manifest["created_at"],
                       dataset=validate_cases(cases), oracle_passed=sum(r["passed"] for r in oracles), integrations=integrations,
                       selected_candidate=candidate_id, candidate_trials=len(registry.state["trials"]), candidate_budget=registry.budget,
                       selection_description=freeze["selection_rule"], checkpoint=checkpoint, reload_verified=reload_ok,
                       exploratory_comparison=paired_cluster_interval(exploratory, replicates=protocol["bootstrap_replicates"], seed=protocol["seed"]),
                       claim_status="Instrumentation demonstrated; Cohere improvement NOT established.",
                       interval_scope="Whole-dataset intervals are descriptive and include validation data. Exposed held-out families are exploratory.",
                       scientific_success=False, protocol=protocol)
        atomic_json(store.path / "summary.json", summary)
        atomic_json(store.path / "dashboard.json", dashboard_payload(summary, rows, cases, oracles, registry.state["trials"], checkpoint))
        write_report(store.path / "report.md", summary)
        artifact_names = [str(p.relative_to(store.path)).replace("\\", "/") for p in store.path.rglob("*")
                          if p.is_file() and p.name != "manifest.json" and "predictions" not in p.parts]
        store.finish([r["storage_id"] for r in rows], artifact_names)
        return summary
    except KeyboardInterrupt:
        store.fail("Interrupted by operator", interrupted=True)
        raise
    except Exception as exc:
        store.fail(exc)
        raise

