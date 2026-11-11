"""Command-line entry points for local demonstration and gated research runs."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

from .artifacts import atomic_json, file_hash, read_json, verify_run, environment
from .data import build_cases, load_dataset, run_oracle, split_by_family, validate_cases, write_dataset


def build_parser():
    parser = argparse.ArgumentParser(prog="aawden", description="Evidence before authority. Reproducible security-review robustness experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Run the explicitly labeled, non-LLM toy pipeline")
    p.add_argument("--output", type=Path, default=Path("runs/aawden-demo"))
    p.add_argument("--protocol", type=Path, default=Path("configs/protocol.json"))
    p.add_argument("--core-only", action="store_true", help="Do not request AuditKIT scoring")
    p = sub.add_parser("serve", help="Open the local evidence dashboard")
    p.add_argument("--run", type=Path, default=Path("runs/aawden-demo"))
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("build-data", help="Write 50 executable, original benchmark cases")
    p.add_argument("--output", type=Path, default=Path("datasets/screen.json"))
    p.add_argument("--exploratory-splits", action="store_true")
    p.add_argument("--fixtures", type=Path, default=Path("fixtures/generated"))
    p = sub.add_parser("validate-data")
    p.add_argument("--dataset", type=Path, default=Path("datasets/screen.json"))
    p.add_argument("--schema-only", action="store_true", help="Validate imported data without executing it")
    p = sub.add_parser("doctor", help="Inspect runtime readiness without downloading a model")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p = sub.add_parser("baseline", help="Run original standard/strong prompts on a local Cohere checkpoint")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--dataset", type=Path, default=Path("datasets/screen.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/protocol.json"))
    p.add_argument("--output", type=Path, default=Path("runs/baseline"))
    p = sub.add_parser("neutral-study", help="Select an evidence-first prompt and test it on fresh Aya fixtures")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/neutral-study.json"))
    p.add_argument("--development", type=Path, default=Path("datasets/screen.json"))
    p.add_argument("--confirmation", type=Path, default=Path("datasets/neutral-confirmation.json"))
    p.add_argument("--output", type=Path, default=Path("runs/aya-neutral-study"))
    p = sub.add_parser("adapter-review", help="Select review instructions for a frozen Aya adapter")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/adapter-review.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/aya-adapter.safetensors"))
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("mental-review", help="Generate execution notes and a strict review with a frozen Aya adapter")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/mental-review.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/aya-adapter.safetensors"))
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select-prompt", help="Choose among prespecified stronger prompts on validation cases")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--dataset", type=Path, default=Path("datasets/exploratory.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/protocol.json"))
    p.add_argument("--output", type=Path, default=Path("runs/prompt-selection"))
    p = sub.add_parser("parity", help="Require actual reference/CircuitKIT numerical agreement")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--dataset", type=Path, default=Path("datasets/screen.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/protocol.json"))
    p.add_argument("--output", type=Path, default=Path("runs/parity.json"))
    p.add_argument("--sequential", action="store_true", help="Keep one model on the GPU at a time")
    p = sub.add_parser("discover", help="Discovery-only residual contrasts and causal patch controls")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--dataset", type=Path, default=Path("datasets/exploratory.json"))
    p.add_argument("--baseline", type=Path, default=Path("runs/baseline"))
    p.add_argument("--parity", type=Path, default=Path("runs/parity.json"))
    p.add_argument("--output", type=Path, default=Path("runs/discovery.json"))
    p.add_argument("--sequential", action="store_true", help="Keep the reference model in host memory")
    p = sub.add_parser("evaluate-candidates", help="Bounded validation search; export only the winner")
    p.add_argument("--config", type=Path, default=Path("configs/model.json"))
    p.add_argument("--dataset", type=Path, default=Path("datasets/exploratory.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/protocol.json"))
    p.add_argument("--discovery", type=Path, default=Path("runs/discovery.json"))
    p.add_argument("--parity", type=Path, default=Path("runs/parity.json"))
    p.add_argument("--output", type=Path, default=Path("runs/candidates"))
    p = sub.add_parser("verify-reload")
    p.add_argument("--search", type=Path, default=Path("runs/candidates"))
    p = sub.add_parser("freeze", help="Freeze a candidate against an external evaluator's dataset commitment")
    p.add_argument("--search", type=Path, default=Path("runs/candidates"))
    p.add_argument("--final-dataset-sha256", required=True)
    p.add_argument("--final-review-sha256", required=True)
    p.add_argument("--coding-dataset-sha256")
    p.add_argument("--exposure-note", required=True, help="Describe evaluator separation and any final-set exposure")
    p = sub.add_parser("review-packet", help="Prepare a blinded label review packet")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development", action="store_true")
    p = sub.add_parser("verify-final-review", aliases=["verify-label-review"], help="Validate independent review and print commitments")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--development", action="store_true")
    p = sub.add_parser("final-evaluate")
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--coding-dataset", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/final"))
    p = sub.add_parser("validate-coding-data", help="Check the coding regression dataset and print its commitment")
    p.add_argument("--dataset", type=Path, required=True)
    for name in ("grounding-packet", "score-grounding"):
        p = sub.add_parser(name, help="Prepare or score a blinded review of the final explanations")
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "score-grounding":
            p.add_argument("--review", type=Path, required=True)
    for name in ("verify", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, default=Path("runs/aawden-demo"))
        if name == "report":
            p.add_argument("--grounding-review", type=Path, help="Include verified independent explanation scores")
    p = sub.add_parser("package", help="Create a local submission bundle; does not upload or submit")
    p.add_argument("--run", type=Path, default=Path("runs/aawden-demo"))
    p.add_argument("--output", type=Path, default=Path("dist/aawden-submission.zip"))
    for name in ("audit-adapter", "adapter-review-packet"):
        p = sub.add_parser(name, help="Verify saved Aya answers or prepare independent review without GPU inference")
        p.add_argument("--run", type=Path, default=Path("runs/aya-beam-20261107/adapter-v7/study/confirmation"))
        p.add_argument("--dataset", type=Path, default=Path("datasets/neutral-confirmation-v5.json"))
        p.add_argument("--selection", type=Path, default=Path("configs/selected-review.json"))
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("score-adapter-explanations", help="Validate an independent review of saved Aya explanations")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return parser


def doctor(config):
    from .sponsors import toolkit_status
    env = environment()
    versions = env["packages"]
    transformer_version = versions["transformers"]["version"] if versions["transformers"] else None
    tl_version = versions["transformer-lens"]["version"] if versions["transformer-lens"] else None
    runtime_ready = bool(transformer_version and transformer_version.startswith("5.") and int(transformer_version.split(".")[1]) >= 15 and tl_version == "3.8.0")
    model_dir = Path(config["model_id"])
    token_variable = next((name for name in ("HF_TOKEN", "AAWDEN_AYA_HF_TOKEN") if os.environ.get(name)), None)
    return {"core_ready": True, "research_runtime_ready": runtime_ready, "environment": env, "toolkits": toolkit_status(),
            "model": {"id": config["model_id"], "revision": config["revision"], "local_checkpoint": model_dir.is_dir(),
                      "credential_in_environment": token_variable is not None, "credential_variable": token_variable,
                      "model_access_verified": False},
            "gates": {"real_weight_parity": "not run by doctor", "cohere_baseline": "not run by doctor",
                      "protected_final_set": "must be supplied by a separate evaluator"},
            "next_step": "Run demo now. Use an isolated compatible research environment and authorized checkpoint for the Cohere study."}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo
            summary = run_demo(args.output, read_json(args.protocol), use_auditkit=not args.core_only)
            result = {"run": str(args.output.resolve()), "evidence_kind": summary["evidence_kind"],
                      "claim_status": summary["claim_status"], "report": str((args.output / "report.md").resolve()),
                      "auditkit": summary["integrations"]["AuditKIT"]["status"]}
        elif args.command == "serve":
            from .server import serve
            serve(args.run, args.port)
            return 0
        elif args.command == "build-data":
            cases = split_by_family(build_cases()) if args.exploratory_splits else build_cases()
            result = write_dataset(args.output, cases)
            args.fixtures.mkdir(parents=True, exist_ok=True)
            for case in cases:
                (args.fixtures / (case.case_id+".py")).write_text(case.code, encoding="utf-8")
            oracle = [run_oracle(c) for c in cases]
            atomic_json(args.fixtures / "oracle-results.json", oracle)
            result["oracle_passed"] = sum(r["passed"] for r in oracle)
            if result["oracle_passed"] != len(cases):
                raise ValueError("An executable oracle failed")
        elif args.command == "validate-data":
            cases = load_dataset(args.dataset)
            result = validate_cases(cases)
            if not args.schema_only:
                oracles = [run_oracle(c) for c in cases]
                result["oracle_passed"] = sum(r["passed"] for r in oracles)
                if not all(r["passed"] for r in oracles):
                    raise ValueError("Executable oracle failed")
        elif args.command == "doctor":
            result = doctor(read_json(args.config))
        elif args.command == "baseline":
            from .workflow import run_evaluation
            result = run_evaluation(load_dataset(args.dataset), read_json(args.protocol), read_json(args.config), args.output)
        elif args.command == "mental-review":
            from .mental_review import run_mental_review
            result = run_mental_review(read_json(args.config), read_json(args.protocol), args.checkpoint,
                                       args.previous, args.output)
        elif args.command == "adapter-review":
            from .adapter_review import run_adapter_review
            result = run_adapter_review(read_json(args.config), read_json(args.protocol), args.checkpoint,
                                        args.previous, args.output)
        elif args.command == "neutral-study":
            from .neutral_study import run_neutral_study
            result = run_neutral_study(read_json(args.config), read_json(args.protocol), args.development, args.confirmation, args.output)
        elif args.command == "select-prompt":
            from .workflow import select_strong_prompt
            result = select_strong_prompt(load_dataset(args.dataset), read_json(args.protocol), read_json(args.config), args.output)
        elif args.command == "parity":
            from .workflow import require_research_runtime
            require_research_runtime()
            from .models import HuggingFaceAdapter
            from .parity import load_instrumented, run_parity, prepare_unedited_reload, verify_unedited_reload
            import gc
            import torch
            cases = load_dataset(args.dataset)
            adapter = HuggingFaceAdapter(read_json(args.config))
            instrumented = None if args.sequential else load_instrumented(adapter)
            result = run_parity(adapter, instrumented, cases, read_json(args.protocol)["parity"])
            del instrumented
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if result["passed"]:
                directory = args.output.resolve().with_suffix(".reference-roundtrip")
                prepare_unedited_reload(adapter, cases, directory)
                del adapter
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result["unedited_reload"] = verify_unedited_reload(directory)
                result["passed"] = result["unedited_reload"]["passed"]
            atomic_json(args.output, result)
            if not result["passed"]:
                print(json.dumps({"error": "Real-weight parity failed", "artifact": str(args.output)}, indent=2))
                return 1
        elif args.command == "discover":
            from .artifacts import digest
            from .workflow import require_research_runtime, checkpoint_identity
            require_research_runtime()
            verify_run(args.baseline)
            baseline = read_json(args.baseline / "summary.json")
            if baseline["evidence_kind"] != "cohere_local_checkpoint" or not baseline["screen"]["h1_screen_passed"]:
                raise ValueError("The actual Cohere baseline did not establish the prespecified failure. Stop this pilot.")
            b = baseline["metrics"]["B"]
            if b["neutral"]["balanced_accuracy"]-b["incorrect_suggestion"]["balanced_accuracy"] < .05:
                raise ValueError("Strong prompting leaves insufficient room for the prespecified gain. Report the cheaper baseline.")
            config, gate = read_json(args.config), read_json(args.parity)
            baseline_manifest = read_json(args.baseline / "manifest.json")
            if baseline_manifest["identity"]["original"] != checkpoint_identity(config):
                raise ValueError("The failure screen belongs to a different model configuration")
            if (not gate.get("real_weights_executed") or not gate.get("passed")
                    or not gate.get("unedited_reload", {}).get("passed") or gate["model_config_sha256"] != digest(config)):
                raise ValueError("Matching, passing real-weight parity is required")
            from .models import HuggingFaceAdapter
            from .parity import load_instrumented, offload_reference
            from .discovery import discover_directions
            adapter = HuggingFaceAdapter(config)
            weight_hashes = adapter.metadata["checkpoint_hashes"]
            if weight_hashes != gate["model"].get("checkpoint_hashes") or weight_hashes != baseline["models"]["A"].get("checkpoint_hashes"):
                raise ValueError("Discovery checkpoint weights differ from parity or baseline")
            if args.sequential:
                offload_reference(adapter)
            instrumented = load_instrumented(adapter)
            cases = [c for c in load_dataset(args.dataset) if c.split == "discovery"]
            result = discover_directions(adapter, instrumented, cases)
            atomic_json(args.output, result)
        elif args.command == "evaluate-candidates":
            from .workflow import evaluate_candidates
            result = evaluate_candidates(load_dataset(args.dataset), read_json(args.protocol), read_json(args.config),
                                         args.discovery, args.parity, args.output)
        elif args.command == "verify-reload":
            from .workflow import verify_reload
            result = verify_reload(args.search)
        elif args.command == "freeze":
            from .workflow import freeze_candidate
            result = freeze_candidate(args.search, args.final_dataset_sha256, args.exposure_note, args.final_review_sha256,
                                      args.coding_dataset_sha256)
        elif args.command == "review-packet":
            from .final_review import prepare_review_packet
            result = prepare_review_packet(args.dataset, args.output, development=args.development)
        elif args.command in ("verify-final-review", "verify-label-review"):
            from .final_review import verify_independent_review
            result = verify_independent_review(args.dataset, args.review, development=args.development)
        elif args.command == "final-evaluate":
            from .workflow import final_evaluate
            result = final_evaluate(args.freeze_manifest, args.dataset, args.output, args.review, args.coding_dataset)
        elif args.command == "validate-coding-data":
            from .regressions import load_coding_cases
            result = {"cases": len(load_coding_cases(args.dataset)), "dataset_sha256": file_hash(args.dataset)}
        elif args.command == "grounding-packet":
            from .regressions import write_grounding_packet
            result = write_grounding_packet(args.run, args.output)
        elif args.command == "score-grounding":
            from .regressions import score_grounding
            result = score_grounding(args.run, args.review, args.output)
        elif args.command == "verify":
            result = verify_run(args.run)
        elif args.command == "report":
            if args.grounding_review:
                from .regressions import score_grounding
                from .reporting import render_report
                scores = score_grounding(args.run, args.grounding_review)
                summary = read_json(args.run / "summary.json")
                summary["grounding_review"] = scores
                summary["unmeasured_endpoints"] = [item for item in summary["unmeasured_endpoints"]
                                                    if item != "human-reviewed explanation grounding"]
                print(render_report(summary))
            else:
                verify_run(args.run)
                print((args.run / "report.md").read_text(encoding="utf-8"))
            return 0
        elif args.command == "audit-adapter":
            from .adapter_evidence import audit_adapter
            result = audit_adapter(args.run, args.dataset, args.selection)
            atomic_json(args.output, result)
        elif args.command == "adapter-review-packet":
            from .adapter_evidence import prepare_adapter_handoff
            result = prepare_adapter_handoff(args.run, args.dataset, args.selection, args.output)
        elif args.command == "score-adapter-explanations":
            from .adapter_evidence import score_adapter_explanations
            result = score_adapter_explanations(args.packet, args.review)
            atomic_json(args.output, result)
        elif args.command == "package":
            from .submission import package_submission
            result = package_submission(args.run, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed per-example artifacts are retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Aawden stopped: {exc}", file=sys.stderr)
        return 1

