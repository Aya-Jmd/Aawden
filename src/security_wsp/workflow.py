"""Real-checkpoint orchestration with gates, bounded search and final-set sealing."""
from __future__ import annotations
import ast
import gc
import os
import re
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

from .artifacts import (RunStore, atomic_json, digest, file_hash, read_json,
                        source_fingerprint, utc_now, verify_run, environment, edit_manifest_path)
from .data import validate_cases, load_dataset
from .evaluation import evaluate_method, summarize
from .interventions import (TrialRegistry, export_hf_projection, project_output_tensor, tensor_hash)
from .models import HuggingFaceAdapter
from .prompts import CONDITIONS, PROMPT_VERSION, STRONG_VARIANTS, render
from .regressions import coding_summary, evaluate_coding, load_coding_cases
from .reporting import write_report
from .sponsors import audit_records
from .statistics import metrics, paired_cluster_interval


def require_research_runtime():
    from importlib.metadata import version, PackageNotFoundError
    for package, prefix in (("transformers", "5."), ("transformer-lens", "3.8.0")):
        try:
            installed = version(package)
        except PackageNotFoundError:
            raise ValueError(f"Missing research dependency: {package}. See requirements-research.in.")
        if package == "transformers":
            parts = installed.split(".")
            if not installed.startswith(prefix) or int(parts[1]) < 15:
                raise ValueError(f"Research runtime requires transformers>=5.15,<6; found {installed}. Use a separate environment.")
        elif installed != prefix:
            raise ValueError(f"Research runtime requires {package}=={prefix}; found {installed}")


def checkpoint_identity(config):
    identity = dict(config)
    path = Path(config["model_id"])
    if path.is_dir():
        identity["local_files"] = {p.name: file_hash(p) for p in sorted(path.iterdir())
                                   if p.is_file() and p.suffix in (".json", ".safetensors", ".model")}
    return identity


def source_template_identity(code):
    return digest(ast.dump(ast.parse(code), include_attributes=False))


def run_evaluation(cases, protocol, original_config, output, edited_config=None, stage="screen", freeze=None,
                   final_review=None, coding_cases=None):
    require_research_runtime()
    if stage == "screen" and any(c.split not in ("screen", "discovery", "validation", "exploratory_test") for c in cases):
        raise ValueError("Protected final data cannot be used in a development screen")
    identity = {"stage": stage, "protocol": protocol, "dataset": digest([asdict(c) for c in cases]),
                "original": checkpoint_identity(original_config),
                "edited": checkpoint_identity(edited_config) if edited_config else None,
                "prompt_version": PROMPT_VERSION, "source": source_fingerprint(),
                "environment": environment(), "freeze_sha256": digest(freeze) if freeze else None,
                "final_review_sha256": final_review["review_sha256"] if final_review else None}
    store = RunStore(output, identity, stage=stage, evidence_kind="cohere_local_checkpoint")
    if store.manifest["status"] == "completed":
        verify_run(store.path)
        return read_json(store.path / "summary.json")
    store.start()
    try:
        rows = []
        authority_rows = []
        coding_rows = []
        model_metadata = {}
        variant = protocol.get("strong_prompt_variant", "trace")
        for config, methods in ((original_config, ("A", "B")), (edited_config, ("C", "D"))):
            if config is None:
                continue
            adapter = HuggingFaceAdapter(config)
            store.bind_model(methods[0], adapter.metadata)
            if methods[0] == "A" and freeze and "original_checkpoint_hashes" in freeze:
                if adapter.metadata["checkpoint_hashes"] != freeze["original_checkpoint_hashes"]:
                    raise ValueError("The original weights differ from the frozen base checkpoint")
            model_metadata[methods[0]] = adapter.metadata
            generation = {"max_new_tokens": config["max_new_tokens"], "seed": config["seed"], "strong_variant": variant}
            for method in methods:
                rows.extend(evaluate_method(cases, adapter, method, store=store, **generation))
            if freeze and protocol.get("test_authority_wording", False):
                authority_rows.extend(evaluate_method(cases, adapter, methods[1], store=store, wording="authority",
                                                     conditions=("incorrect_suggestion",), **generation))
            if coding_cases:
                arm = "original" if methods[0] == "A" else "edited"
                coding_rows.extend(evaluate_coding(coding_cases, adapter, arm, store, config))
            if methods[0] == "A" and freeze:
                for recipe in freeze["random_control_recipes"]:
                    with temporary_projection(adapter, recipe["tensor"], recipe["direction"], recipe["strength"]):
                        rows.extend(evaluate_method(cases, adapter, "R"+str(recipe["seed"]), store=store, **generation))
                noop = evaluate_method(cases, adapter, "N", store=store, **generation)
                original_strong = [r for r in rows if r["method"] == "B"]
                if [r["raw_output"] for r in noop] != [r["raw_output"] for r in original_strong]:
                    raise ValueError("Final no-op control differs from the original strong-prompt comparison")
                rows.extend(noop)
            del adapter
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_json(store.path / "results.json", rows)
        atomic_json(store.path / "dataset.json", {"metadata": validate_cases(cases), "cases": [asdict(c) for c in cases]})
        integration = audit_records(rows, store.path / "auditkit.json")
        summary = summarize(rows, protocol)
        artifacts = ["results.json", "dataset.json", "summary.json", "report.md", "auditkit.json"]
        if authority_rows:
            atomic_json(store.path / "authority-results.json", authority_rows)
            artifacts.append("authority-results.json")
            summary["authority_wording"] = paired_cluster_interval(authority_rows, replicates=protocol["bootstrap_replicates"],
                                                                  seed=protocol["seed"])
        if coding_rows:
            atomic_json(store.path / "coding-results.json", coding_rows)
            artifacts.append("coding-results.json")
            summary["coding_regression"] = coding_summary(coding_rows, freeze["coding_dataset_sha256"])
            summary["unmeasured_endpoints"].remove("general coding regressions")
        summary.update(evidence_kind="cohere_local_checkpoint", dataset=validate_cases(cases),
                       integrations={"AuditKIT": integration}, models=model_metadata,
                       run_id=store.manifest["run_id"], scientific_success=False,
                       strong_prompt_variant=variant,
                       claim_status="Measured generations; protected transfer and regression gates require separate evidence.",
                       selection_description="No candidate is selected by this evaluation command.")
        if freeze:
            p, n, c = (summary["comparisons"][key] for key in ("incorrect_suggestion", "neutral", "correct_suggestion"))
            gates = {"minimum_gain": p["difference"] >= protocol["minimum_gain"], "positive_paired_interval": p["ci95"][0] > 0,
                     "neutral_noninferiority": n["ci95"][0] >= -protocol["neutral_noninferiority_margin"],
                     "correct_suggestion_guard": c["ci95"][0] >= -protocol["neutral_noninferiority_margin"],
                     "general_coding_regression": False, "independent_grounding_review": False}
            summary["final_gates"] = gates
            if final_review:
                summary["final_review"] = final_review
                summary["dataset"]["label_review"] = final_review["status"]
                summary["unmeasured_endpoints"] = [item for item in summary["unmeasured_endpoints"]
                                                    if item not in ("independent label review", "protected final-test transfer")]
            summary["claim_status"] = "Bounded benchmark evaluated. See separate regression and grounding evidence."
            summary["selection_description"] = freeze["selection_rule"]
            if "trials" in freeze:
                atomic_json(store.path / "trials.json", freeze["trials"])
                artifacts.append("trials.json")
                summary["trials"] = [{"candidate_id": t["candidate_id"], "status": t["status"],
                                      "role": t.get("role", t["recipe"]["kind"]), "eligible": t.get("eligible"),
                                      "attempts": [attempt["status"] for attempt in t.get("attempts", [])]}
                                     for t in freeze["trials"]]
        atomic_json(store.path / "summary.json", summary)
        write_report(store.path / "report.md", summary)
        store.finish([r["storage_id"] for r in rows + authority_rows + coding_rows], artifacts)
        return summary
    except KeyboardInterrupt:
        store.fail("Interrupted by operator", interrupted=True)
        raise
    except Exception as exc:
        store.fail(exc)
        raise


def select_strong_prompt(cases, protocol, config, output):
    require_research_runtime()
    if any(case.split == "final_test" for case in cases):
        raise ValueError("Final cases cannot be used in prompt selection")
    validation = [case for case in cases if case.split == "validation"]
    if len({case.oracle_verdict for case in validation}) != 2:
        raise ValueError("Prompt selection needs both verdicts on validation cases")
    variants = protocol.get("strong_prompt_candidates", ["trace"])
    if not variants or len(set(variants)) != len(variants) or any(v not in STRONG_VARIANTS for v in variants):
        raise ValueError("Declare unique strong prompt candidates in the protocol before evaluation")
    identity = {"validation": digest([asdict(c) for c in validation]), "model": checkpoint_identity(config),
                "protocol": protocol, "source": source_fingerprint(), "environment": environment()}
    store = RunStore(output, identity, stage="prompt_selection", evidence_kind="cohere_local_checkpoint")
    if store.manifest["status"] == "completed":
        verify_run(store.path)
        return read_json(store.path / "selection.json")
    store.start()
    try:
        adapter = HuggingFaceAdapter(config)
        store.bind_model("B", adapter.metadata)
        rows, scores = [], {}
        for variant in variants:
            responses = evaluate_method(validation, adapter, "B", store=store, strong_variant=variant,
                                        max_new_tokens=config["max_new_tokens"], seed=config["seed"])
            scores[variant] = {cond: metrics([r for r in responses if r["condition"] == cond]) for cond in CONDITIONS}
            rows.extend(responses)
        reference = scores[variants[0]]
        margin = protocol["neutral_noninferiority_margin"]
        eligible = [name for name in variants if all(scores[name][cond]["balanced_accuracy"] >=
                    reference[cond]["balanced_accuracy"] - margin for cond in ("neutral", "correct_suggestion"))]
        chosen = max(eligible, key=lambda name: scores[name]["incorrect_suggestion"]["balanced_accuracy"])
        result = {"selected": chosen, "eligible": eligible, "metrics": scores, "model": adapter.metadata,
                  "selected_protocol": dict(protocol, strong_prompt_variant=chosen),
                  "validation_families": sorted({case.family_id for case in validation})}
        atomic_json(store.path / "selection.json", result)
        atomic_json(store.path / "protocol.json", result["selected_protocol"])
        store.finish([row["storage_id"] for row in rows], ["selection.json", "protocol.json"])
        return result
    except KeyboardInterrupt:
        store.fail("Prompt selection interrupted", interrupted=True)
        raise
    except Exception as exc:
        store.fail(type(exc).__name__)
        raise


@contextmanager
def temporary_projection(adapter, name, direction, strength):
    import torch
    params = dict(adapter.model.named_parameters())
    if name not in params or not name.endswith((".mlp.down_proj.weight", ".self_attn.o_proj.weight")):
        raise ValueError("Discovery selected an unsupported output projection")
    parameter = params[name]
    original = parameter.detach().clone()
    original_hash = tensor_hash(original)
    try:
        updated = project_output_tensor(original, torch.tensor(direction, device=parameter.device), strength)
        with torch.no_grad():
            parameter.copy_(updated)
        yield {"name": name, "original_sha256": original_hash, "edited_sha256": tensor_hash(parameter),
               "shape": list(parameter.shape), "dtype": str(parameter.dtype), "touched_entries": parameter.numel(),
               "changed_entries": int((parameter != original).sum().item()),
               "update_l2": (parameter.float()-original.float()).norm().item()}
    finally:
        with torch.no_grad():
            parameter.copy_(original)
        if tensor_hash(parameter) != original_hash:
            raise RuntimeError("Original tensor was not restored")


def evaluate_candidates(cases, protocol, config, discovery_path: Path, parity_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".search-running"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        return search_candidates(cases, protocol, config, discovery_path, parity_path, output)
    finally:
        lock.unlink()


def run_validation(cases, adapter, method, generation, directory, identity, recipe=None):
    store = RunStore(directory, {"search": identity, "method": method, "recipe": recipe},
                     stage="candidate_validation", evidence_kind="cohere_local_checkpoint")
    if store.manifest["status"] == "completed":
        verify_run(store.path)
        return read_json(store.path / "validation.json")
    store.start(max_attempts=2)
    try:
        projection = (temporary_projection(adapter, recipe["tensor"], recipe["direction"], recipe["strength"])
                      if recipe else nullcontext())
        with projection as edit:
            rows = evaluate_method(cases, adapter, method, store=store, **generation)
        record = {"rows": rows, "edit": edit}
        atomic_json(store.path / "validation.json", record)
        store.finish([row["storage_id"] for row in rows], ["validation.json"])
        return record
    except KeyboardInterrupt:
        store.fail("Validation interrupted", interrupted=True)
        raise
    except Exception as exc:
        store.fail(type(exc).__name__)
        raise


def search_candidates(cases, protocol, config, discovery_path: Path, parity_path: Path, output: Path):
    require_research_runtime()
    if any(case.split == "final_test" for case in cases):
        raise ValueError("Final cases cannot be used in candidate selection")
    if protocol.get("strong_prompt_candidates") and protocol.get("strong_prompt_variant") not in protocol["strong_prompt_candidates"]:
        raise ValueError("Run select-prompt and use the selected protocol for candidate evaluation")
    discovery, parity = read_json(discovery_path), read_json(parity_path)
    if (not parity.get("passed") or not parity.get("real_weights_executed")
            or not parity.get("unedited_reload", {}).get("passed") or parity["model_config_sha256"] != digest(config)):
        raise ValueError("A passing real-weight parity gate for this exact model configuration is required")
    if discovery["model_config_sha256"] != digest(config):
        raise ValueError("Discovery and target checkpoint differ")
    validation = [c for c in cases if c.split == "validation"]
    if not validation or set(discovery["family_ids"]) & {c.family_id for c in validation}:
        raise ValueError("Validation families must be disjoint from mechanism discovery")
    hypotheses = [c for c in discovery["candidates"] if c["mean_causal_margin_gain"] > c["mean_random_gain"] and c["mean_causal_margin_gain"] > 0][:2]
    if not hypotheses:
        raise ValueError("No candidate beat the causal random control. Record a negative mechanism result.")
    identity = {"protocol": protocol, "config": checkpoint_identity(config), "source": source_fingerprint(),
                "discovery_sha256": file_hash(discovery_path), "parity_sha256": file_hash(parity_path),
                "validation_sha256": digest([asdict(c) for c in validation])}
    registry = TrialRegistry(output / "trials.json", identity, protocol["max_edit_candidates"])
    adapter = HuggingFaceAdapter(config)
    if parity.get("model", {}).get("checkpoint_hashes") != adapter.metadata.get("checkpoint_hashes"):
        raise ValueError("Candidate search weights differ from the checkpoint used for parity")
    variant = protocol.get("strong_prompt_variant", "trace")
    generation = {"max_new_tokens": config["max_new_tokens"], "seed": config["seed"], "strong_variant": variant}
    validation_root = output / "validation"
    search_identity = digest({"experiment": identity, "loaded_model": adapter.metadata})
    base_rows = run_validation(validation, adapter, "B", generation, validation_root / "original", search_identity)["rows"]
    atomic_json(output / "baseline-validation.json", base_rows)
    base = {cond: metrics([r for r in base_rows if r["condition"] == cond]) for cond in protocol["conditions"]}

    def evaluate_trial(recipe, method):
        trial = registry.reserve(recipe)
        candidate_id = trial["candidate_id"]
        directory = validation_root / candidate_id
        try:
            record = run_validation(validation, adapter, method, generation, directory, search_identity, recipe)
        except (Exception, KeyboardInterrupt) as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            registry.record(candidate_id, status=status, error_type=type(exc).__name__)
            raise
        atomic_json(output / (candidate_id+"-validation.json"), record["rows"])
        attempts = read_json(directory / "manifest.json")["attempts"]
        return candidate_id, record["rows"], record["edit"], attempts

    all_candidates = []
    for hypothesis in hypotheses:
        for strength in (.5, 1.0):
            recipe = {"kind": "output_projection", "tensor": hypothesis["tensor_name"], "direction": hypothesis["direction"], "strength": strength}
            candidate_id, rows, edit, attempts = evaluate_trial(recipe, "D")
            values = {cond: metrics([r for r in rows if r["condition"] == cond]) for cond in protocol["conditions"]}
            eligible = values["neutral"]["balanced_accuracy"] >= base["neutral"]["balanced_accuracy"]-protocol["neutral_noninferiority_margin"]
            eligible &= values["correct_suggestion"]["accuracy"] >= base["correct_suggestion"]["accuracy"]-protocol["neutral_noninferiority_margin"]
            record = registry.complete(candidate_id, role="targeted", eligible=eligible, metrics=values, edit=edit, attempts=attempts)
            all_candidates.append(record)
    eligible = [r for r in all_candidates if r["eligible"] and r["metrics"]["incorrect_suggestion"]["balanced_accuracy"] > base["incorrect_suggestion"]["balanced_accuracy"]]
    if not eligible:
        atomic_json(output / "selection.json", {"status": "negative_result", "reason": "No candidate improved the strong-prompt comparator under validation guardrails."})
        return {"status": "negative_result"}
    chosen = max(eligible, key=lambda r: (r["metrics"]["incorrect_suggestion"]["balanced_accuracy"], -r["edit"]["update_l2"]))
    # Random directions use the SAME tensor; strength is adjusted to match
    # the selected Frobenius update norm and any residual mismatch is logged.
    import torch
    param = dict(adapter.model.named_parameters())[chosen["recipe"]["tensor"]]
    for seed in protocol["random_control_seeds"]:
        generator = torch.Generator().manual_seed(seed)
        direction = torch.randn(param.shape[0], generator=generator)
        direction /= direction.norm()
        unit_edit = project_output_tensor(param.detach(), direction.to(param.device), 1)
        norm = (unit_edit.float()-param.detach().float()).norm().item()
        strength = min(1.0, chosen["edit"]["update_l2"]/max(norm, 1e-12))
        recipe = {"kind": "random_direction_control", "tensor": chosen["recipe"]["tensor"], "seed": seed,
                  "direction": direction.tolist(), "strength": strength}
        candidate_id, rows, edit, attempts = evaluate_trial(recipe, "R"+str(seed))
        values = {cond: metrics([r for r in rows if r["condition"] == cond]) for cond in protocol["conditions"]}
        registry.complete(candidate_id, role="random_control", eligible=False, edit=edit, metrics=values, attempts=attempts,
                          target_norm=chosen["edit"]["update_l2"], norm_mismatch=edit["update_l2"]-chosen["edit"]["update_l2"])
    # A no-op is a separate software control and consumes no edited trial.
    noop_rows = run_validation(validation, adapter, "N", generation, validation_root / "unchanged", search_identity)["rows"]
    atomic_json(output / "noop-validation.json", noop_rows)
    if [r["raw_output"] for r in noop_rows] != [r["raw_output"] for r in base_rows]:
        raise ValueError("Unedited/no-op generation changed during candidate search")
    checkpoint = output / "checkpoint"
    recipe = chosen["recipe"]
    if checkpoint.exists():
        manifest = read_json(edit_manifest_path(checkpoint))
        if (manifest["base"].get("checkpoint_hashes") != adapter.metadata["checkpoint_hashes"]
                or manifest["discovery_sha256"] != identity["discovery_sha256"] or manifest["direction_sha256"] != digest(recipe["direction"])
                or manifest["strength"] != recipe["strength"]
                or [(t["name"], t["original_sha256"], t["edited_sha256"]) for t in manifest["tensors"]]
                != [(recipe["tensor"], chosen["edit"]["original_sha256"], chosen["edit"]["edited_sha256"])]) or any(
                    file_hash(checkpoint / name) != expected for name, expected in manifest["checkpoint_hashes"].items()):
            raise ValueError("Existing exported checkpoint differs from the resumed selection")
    else:
        manifest = export_hf_projection(adapter, [recipe["tensor"]], recipe["direction"], recipe["strength"],
                                        checkpoint, identity["discovery_sha256"])
    edited_config = dict(config, model_id=str(checkpoint.resolve()), local_files_only=True)
    atomic_json(output / "edited-model.json", edited_config)
    requests = [render(c, cond, strong=True, **generation)
                for c in validation for cond in protocol["conditions"]]
    atomic_json(output / "reload-requests.json", [asdict(r) for r in requests])
    reference = read_json(output / (chosen["candidate_id"]+"-validation.json"))
    atomic_json(output / "reload-reference.json", [{"request_id": r["request_id"], "text": r["raw_output"]} for r in reference])
    reload_inputs = {name: file_hash(output / name) for name in ("edited-model.json", "reload-requests.json", "reload-reference.json")}
    selection = {"status": "selected", "candidate_id": chosen["candidate_id"], "selected_at": utc_now(),
                 "selection_rule": "Validation-only maximum misleading BA, neutral/correct guardrails, smaller norm tie break.",
                 "identity": identity, "original_config": config, "edited_config": edited_config, "protocol": protocol,
                 "development_family_ids": sorted(set(discovery["family_ids"]) | {c.family_id for c in cases}),
                 "development_source_ids": sorted({c.source_id for c in cases}),
                 "development_code_hashes": sorted({c.source_sha256 for c in cases}),
                 "development_template_hashes": sorted({source_template_identity(c.code) for c in cases}),
                 "random_control_recipes": [t["recipe"] for t in registry.state["trials"] if t.get("role") == "random_control"],
                 "validation_cases": [asdict(c) for c in validation], "edit_manifest_sha256": file_hash(edit_manifest_path(checkpoint)),
                 "checkpoint_hashes": manifest["checkpoint_hashes"], "trials_sha256": file_hash(registry.path),
                 "original_checkpoint_hashes": adapter.metadata["checkpoint_hashes"],
                 "reload_inputs": reload_inputs, "reload_verified": False}
    atomic_json(output / "selection.json", selection)
    return selection


def verify_reload(search: Path):
    selection = read_json(search / "selection.json")
    if selection["status"] != "selected":
        raise ValueError("A selected exported candidate is required")
    checkpoint = Path(selection["edited_config"]["model_id"])
    if (file_hash(edit_manifest_path(checkpoint)) != selection["edit_manifest_sha256"]
            or any(file_hash(checkpoint / name) != expected for name, expected in selection["checkpoint_hashes"].items())
            or any(file_hash(search / name) != expected for name, expected in selection.get("reload_inputs", {}).items())):
        raise ValueError("The selected checkpoint or its reload inputs changed")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-m", "security_wsp.reload_probe", "--kind", "hf",
                             "--checkpoint", str((search / "edited-model.json").resolve()),
                             "--requests", str((search / "reload-requests.json").resolve()),
                             "--output", str((search / "reload-outputs.json").resolve())], env=env, timeout=7200)
    if result.returncode:
        raise ValueError("Fresh-process checkpoint reload failed")
    reference, actual = read_json(search / "reload-reference.json"), read_json(search / "reload-outputs.json")
    passed = len(reference) == len(actual) and all(a["request_id"] == b["request_id"] and a["text"] == b["text"] for a, b in zip(reference, actual))
    record = {"passed": passed, "fresh_process": True, "hooks_required": False, "checked_generations": len(reference),
              "checkpoint_manifest_sha256": selection["edit_manifest_sha256"], "checked_at": utc_now()}
    atomic_json(search / "reload-verification.json", record)
    if not passed:
        raise ValueError("Exported/reloaded generations differ from the selected in-memory candidate")
    return record


def freeze_candidate(search: Path, final_dataset_sha256: str, exposure_note: str, final_review_sha256: str,
                     coding_dataset_sha256=None):
    commitments = {"dataset": final_dataset_sha256, "review": final_review_sha256}
    if coding_dataset_sha256 is not None:
        commitments["coding dataset"] = coding_dataset_sha256
    for name, value in commitments.items():
        if not re.fullmatch(r"[a-fA-F0-9]{64}", value):
            raise ValueError(f"Supply the evaluator's final {name} FILE SHA-256 before opening final data")
    if not exposure_note.strip():
        raise ValueError("Describe how the external evaluator kept final cases out of edit selection")
    selection = read_json(search / "selection.json")
    reload = read_json(search / "reload-verification.json")
    if selection["status"] != "selected" or not reload["passed"]:
        raise ValueError("Selection and fresh-process behavioral reload must pass before freeze")
    if reload["checkpoint_manifest_sha256"] != selection["edit_manifest_sha256"] or selection["identity"]["source"] != source_fingerprint():
        raise ValueError("The reload check or source does not match this selection")
    checkpoint = Path(selection["edited_config"]["model_id"])
    if file_hash(edit_manifest_path(checkpoint)) != selection["edit_manifest_sha256"]:
        raise ValueError("Selected edit manifest changed")
    for name, expected in selection["checkpoint_hashes"].items():
        if file_hash(checkpoint / name) != expected:
            raise ValueError("Selected checkpoint changed")
    frozen_path = search / "frozen.json"
    if frozen_path.exists():
        raise ValueError("A frozen comparison is immutable. Use it as-is; do not refreeze after viewing final results.")
    frozen = dict(selection, final_dataset_sha256=final_dataset_sha256.lower(), final_review_sha256=final_review_sha256.lower(),
                  evaluator_exposure_note=exposure_note,
                  frozen_at=utc_now(), reload_verified=True, reload_sha256=file_hash(search / "reload-verification.json"),
                  prompt_version=PROMPT_VERSION, source_sha256=source_fingerprint())
    if "trials_sha256" in selection:
        if file_hash(search / "trials.json") != selection["trials_sha256"]:
            raise ValueError("Validation trial ledger changed after selection")
        frozen["trials"] = read_json(search / "trials.json")["trials"]
    if coding_dataset_sha256 is not None:
        frozen["coding_dataset_sha256"] = coding_dataset_sha256.lower()
    atomic_json(frozen_path, frozen)
    return frozen


def final_evaluate(frozen_path: Path, dataset_path: Path, output: Path, review_path: Path, coding_dataset=None):
    frozen = read_json(frozen_path)
    if file_hash(dataset_path) != frozen["final_dataset_sha256"]:
        raise ValueError("Final dataset differs from the evaluator's pre-freeze commitment")
    if file_hash(review_path) != frozen.get("final_review_sha256"):
        raise ValueError("Independent review differs from the evaluator's pre-freeze commitment")
    coding_hash = frozen.get("coding_dataset_sha256")
    if (coding_hash or coding_dataset) and (not coding_dataset or file_hash(coding_dataset) != coding_hash):
        raise ValueError("Coding dataset differs from the evaluator's pre-freeze commitment")
    coding_cases = load_coding_cases(coding_dataset) if coding_dataset else None
    if source_fingerprint() != frozen["source_sha256"] or frozen["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Source or prompt changed after freeze")
    from .final_review import final_cases, verify_independent_review
    cases = final_cases(dataset_path)
    if set(frozen["development_family_ids"]) & {c.family_id for c in cases} or set(frozen["development_source_ids"]) & {c.source_id for c in cases}:
        raise ValueError("Final/development family or source leakage")
    if set(frozen["development_code_hashes"]) & {c.source_sha256 for c in cases}:
        raise ValueError("Final set repeats development source code")
    if set(frozen["development_template_hashes"]) & {source_template_identity(c.code) for c in cases}:
        raise ValueError("Final set repeats a development source template")
    final_review = verify_independent_review(dataset_path, review_path, cases)
    checkpoint = Path(frozen["edited_config"]["model_id"])
    if checkpoint_identity(frozen["original_config"]) != frozen["identity"]["config"]:
        raise ValueError("Frozen original checkpoint identity changed")
    if file_hash(edit_manifest_path(checkpoint)) != frozen["edit_manifest_sha256"]:
        raise ValueError("Frozen edit manifest changed")
    for name, expected in frozen["checkpoint_hashes"].items():
        if file_hash(checkpoint / name) != expected:
            raise ValueError("Frozen checkpoint changed")
    # The output directory prevents accidental repeated final looks.
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest["status"] == "completed":
            raise ValueError("Final comparison already completed. Report the frozen result; no retuning on this set.")
    return run_evaluation(cases, frozen["protocol"], frozen["original_config"], output,
                          frozen["edited_config"], stage="final_evaluation", freeze=frozen,
                          final_review=final_review, coding_cases=coding_cases)

