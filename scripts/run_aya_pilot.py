"""Run the pinned pilot on a provisioned host; preflight never fetches weights."""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from security_wsp.artifacts import atomic_json, digest, file_hash, read_json, source_fingerprint, utc_now, verify_run
from security_wsp.data import load_dataset
from security_wsp.prompts import PROMPT_VERSION

MODEL = "CohereLabs/aya-expanse-8b"
REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"
PARAMETERS = 8_028_033_024
GIB = 1 << 30


def checkpoint_access(config, cache):
    path = Path(config["model_id"])
    if path.is_dir():
        try:
            model = read_json(path / "config.json")
            index = path / "model.safetensors.index.json"
            shards = set(read_json(index)["weight_map"].values()) if index.is_file() else {"model.safetensors"}
            complete = bool(shards) and all(
                Path(name).name == name and name.endswith(".safetensors") and (path / name).is_file()
                for name in shards
            )
            if model.get("model_type") != "cohere" or not complete:
                return {"passed": False, "reason": "local_cohere_safetensors_incomplete"}
            return {"passed": True, "reason": "operator_supplied_local_files_present", "shards": len(shards)}
        except (OSError, ValueError, KeyError, TypeError):
            return {"passed": False, "reason": "local_checkpoint_metadata_invalid"}
    if config["model_id"] != MODEL:
        return {"passed": False, "reason": "unexpected_model_id"}
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(MODEL, "config.json", revision=REVISION, token=True,
                               cache_dir=str(cache), etag_timeout=10,
                               local_files_only=config.get("local_files_only", False))
        return {"passed": read_json(path).get("model_type") == "cohere",
                "reason": "pinned_config_read", "config_sha256": file_hash(Path(path))}
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        # Some Hub clients return cached/config-only errors without HTTP status.
        # Never print the exception, authenticated request, or credential.
        if status == 403:
            message = str(exc).lower()
            reason = "gated_token_scope" if "fine-grained" in message and "public gated" in message else "gated_access_forbidden"
        else:
            reason = type(exc).__name__
        return {"passed": False, "reason": reason, "http_status": status}


def gpu_readiness(config, sequential=False):
    try:
        import torch
        device = config["device"]
        if not torch.cuda.is_available() or not (device == "auto" or re.fullmatch(r"cuda(?::\d+)?", device)):
            return {"passed": False, "reason": "cuda_unavailable_for_config", "torch": torch.__version__}
        device = "cuda:0" if device == "auto" else device
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info()
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            minimum = 21 if sequential else 42
            return {"passed": native_bf16 and free >= minimum * GIB,
                    "reason": "planning_headroom_check", "device": str(device),
                    "name": torch.cuda.get_device_name(), "native_bf16": native_bf16,
                    "free_gib": round(free / GIB, 2), "total_gib": round(total / GIB, 2),
                    "minimum_free_gib": minimum, "torch": torch.__version__}
    except Exception as exc:
        return {"passed": False, "reason": type(exc).__name__}


def run_preflight(config, protocol, screen, exploratory, output, sequential=False):
    from security_wsp.workflow import require_research_runtime
    from security_wsp.sponsors import toolkit_status
    if config["revision"] != REVISION or config["dtype"] != "bfloat16":
        raise ValueError("The pilot preflight is sized for pinned Aya Expanse 8B in BF16")
    try:
        require_research_runtime()
        runtime = {"passed": True}
    except ValueError as exc:
        runtime = {"passed": False, "reason": str(exc)}
    toolkits = toolkit_status()
    count = len(screen)
    dev = len([case for case in exploratory if case.split == "discovery"])
    val = len([case for case in exploratory if case.split == "validation"])
    free_disk = shutil.disk_usage(output).free / GIB
    report = {
        "stage": "preflight",
        "checked_at": utc_now(),
        "model_id": MODEL,
        "revision": REVISION,
        "weights_loaded": False,
        "runtime": runtime,
        "toolkits": {name: {"passed": item["available"]} for name, item in toolkits.items()},
        "gpu": gpu_readiness(config, sequential),
        "instrumentation_loading": "sequential" if sequential else "concurrent",
        "model_access": checkpoint_access(config, output / ".hf-metadata"),
        "workspace_disk": {"passed": free_disk >= 80, "free_gib": round(free_disk, 2), "planned_free_gib": 80,
                           "scope": "This volume only. Check separate HF cache and scheduler storage quotas independently."},
        "workload": {
            "two_bf16_weight_copies_gib": round(4 * PARAMETERS / GIB, 2),
            "baseline_cases": count,
            "baseline_generations": count * len(protocol["conditions"]) * 2,
            "discovery_cases": dev,
            "validation_cases": val,
            "max_new_tokens": config["max_new_tokens"],
            "recommended_pilot_gpu": "one 80 GiB-class BF16-capable GPU; 48 GiB-class possible subject to measured peaks",
            "pilot_budget_estimate_gpu_hours": [2, 4],
            "final_300_case_all_arm_estimate_gpu_hours": [10, 20],
            "budget_scope": "Planning estimates, not measured throughput. Recompute after the baseline.",
        },
    }
    report["passed"] = all(report[key]["passed"] for key in ("runtime", "gpu", "model_access", "workspace_disk")) \
        and all(item["passed"] for item in report["toolkits"].values())
    atomic_json(output / "preflight.json", report)
    return report


def execute_stage(manifest_path, manifest, name, arguments, artifact, output):
    record = manifest["stages"].get(name)
    if record and record["status"] == "completed":
        if file_hash(artifact) != record["artifact_sha256"]:
            raise ValueError(f"Completed stage artifact changed: {name}")
        return read_json(artifact)
    log = output / f"{name}.log"
    attempts = list(record.get("attempts", [record])) if record else []
    if len(attempts) >= 2:
        raise ValueError(f"Retry limit reached for {name}; inspect its log")
    if attempts and attempts[-1]["status"] == "running":
        attempts[-1].update(status="interrupted", ended_at=utc_now())
    attempt = {"status": "running", "started_at": utc_now(), "log": log.name}
    attempts.append(attempt)
    manifest["stages"][name] = dict(attempt, attempts=attempts)
    atomic_json(manifest_path, manifest)
    print(f"Running {name}; log: {log}", flush=True)
    start = time.monotonic()
    try:
        with log.open("ab") as handle:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "aawden.py"), *map(str, arguments)],
                                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        attempt["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {log}")
        value = read_json(artifact)
        attempt.update(status="completed", artifact_sha256=file_hash(artifact))
        return value
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt) or attempt.get("exit_code") == 130
        attempt.update(status="interrupted" if interrupted else "failed", error_type=type(exc).__name__)
        raise
    finally:
        attempt.update(seconds=round(time.monotonic()-start, 2), ended_at=utc_now())
        manifest["stages"][name] = dict(attempt, attempts=attempts)
        atomic_json(manifest_path, manifest)


def baseline_budget(summary):
    arms = [m for method in ("A", "B") for m in summary["metrics"][method].values()]
    requests = sum(row["n"] for row in arms)
    tokens = sum(row["output_tokens"] for row in arms)
    seconds = sum(row["latency_seconds"] for row in arms)
    if not requests or not seconds:
        raise ValueError("The baseline did not record throughput")
    return {"measured_generations": requests, "measured_generated_tokens": tokens,
            "measured_generation_seconds": seconds, "mean_seconds_per_generation": seconds / requests,
            "generated_tokens_per_generation_second": tokens / seconds,
            "final_300_case_8_arm_estimated_generation_hours": seconds / requests * 7200 / 3600,
            "final_with_authority_wording_estimated_generation_hours": seconds / requests * 7800 / 3600,
            "scope": "Extrapolation from original-model generations; excludes setup and assumes similar output lengths."}


def verify_reused_baseline(path, config, protocol, cases):
    from security_wsp.workflow import checkpoint_identity
    verify_run(path)
    identity = read_json(path / "manifest.json")["identity"]
    expected = {"stage": "screen", "protocol": protocol, "dataset": digest([asdict(case) for case in cases]),
                "original": checkpoint_identity(config), "edited": None, "prompt_version": PROMPT_VERSION}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("The completed baseline must use the same model, screen cases, protocol and prompt version")
    summary = read_json(path / "summary.json")
    if summary["evidence_kind"] != "cohere_local_checkpoint":
        raise ValueError("A real Cohere baseline is required")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sequential-gpu", action="store_true", help="Keep one model on the GPU at a time")
    parser.add_argument("--reuse-baseline", type=Path, help="Use a completed, verified baseline run directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    screen_path, dev_path = ROOT / "datasets/screen.json", ROOT / "datasets/exploratory.json"
    config, protocol = read_json(args.config), read_json(args.protocol)
    screen_cases = load_dataset(screen_path)
    preflight = run_preflight(config, protocol, screen_cases, load_dataset(dev_path), output, args.sequential_gpu)
    print(json.dumps(preflight, indent=2), flush=True)
    if not preflight["passed"] or args.preflight_only:
        return 0 if preflight["passed"] else 2

    reuse = args.reuse_baseline.resolve() if args.reuse_baseline else None
    reused_summary = verify_reused_baseline(reuse, config, protocol, screen_cases) if reuse else None
    reused_manifest = file_hash(reuse / "manifest.json") if reuse else None
    identity = digest({"config": config, "protocol": protocol, "source": source_fingerprint(),
                       "runner": file_hash(Path(__file__)), "screen": file_hash(screen_path), "development": file_hash(dev_path),
                       "sequential_gpu": args.sequential_gpu, "reused_baseline_manifest_sha256": reused_manifest})
    manifest_path = output / "pilot.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "identity_sha256": identity, "created_at": utc_now(), "stages": {}}
    if manifest["identity_sha256"] != identity:
        raise ValueError("Pilot configuration or code changed; use a new output directory")
    lock = output / ".pilot-running"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    common = ["--config", args.config.resolve()]
    protocol_args = ["--protocol", args.protocol.resolve()]
    instrumentation = ["--sequential"] if args.sequential_gpu else []
    parity, baseline, discovery, search = (output / name for name in ("parity.json", "baseline", "discovery.json", "candidates"))
    def stage(name, command, artifact):
        return execute_stage(manifest_path, manifest, name, command, artifact, output)

    try:
        gate = stage("parity", ["parity", *common, *protocol_args, *instrumentation,
                                "--dataset", screen_path, "--output", parity], parity)
        if reuse:
            checkpoint_hashes = gate["model"]["checkpoint_hashes"]
            if reused_summary["models"]["A"]["checkpoint_hashes"] != checkpoint_hashes:
                raise ValueError("The completed baseline used different checkpoint weights from parity")
            screen, baseline = reused_summary, reuse
            manifest["reused_baseline_manifest_sha256"] = reused_manifest
        else:
            screen = stage("baseline", ["baseline", *common, *protocol_args, "--dataset", screen_path, "--output", baseline],
                           baseline / "summary.json")
        budget = baseline_budget(screen)
        atomic_json(output / "baseline-budget.json", budget)
        print(json.dumps(budget, indent=2), flush=True)
        b = screen["metrics"]["B"]
        minimum_gap = protocol["minimum_gain"]
        enough_room = screen["screen"]["h1_screen_passed"] and b["neutral"]["balanced_accuracy"] - b["incorrect_suggestion"]["balanced_accuracy"] >= minimum_gap
        candidate_protocol = protocol_args
        if enough_room and protocol.get("strong_prompt_candidates"):
            prompt_output = output / "prompt-selection"
            selection = stage("prompt-selection", ["select-prompt", *common, *protocol_args, "--dataset", dev_path,
                                                   "--output", prompt_output], prompt_output / "selection.json")
            selected_protocol = output / "selected-protocol.json"
            atomic_json(selected_protocol, selection["selected_protocol"])
            candidate_protocol = ["--protocol", selected_protocol]
            selected = selection["metrics"][selection["selected"]]
            enough_room = selected["neutral"]["balanced_accuracy"] - selected["incorrect_suggestion"]["balanced_accuracy"] >= minimum_gap
        if not enough_room:
            manifest["outcome"] = "negative_screen_or_prompt_suffices"
        else:
            findings = stage("discovery", ["discover", *common, *instrumentation, "--dataset", dev_path, "--baseline", baseline,
                                            "--parity", parity, "--output", discovery], discovery)
            if not any(c["mean_causal_margin_gain"] > max(0, c["mean_random_gain"]) for c in findings["candidates"]):
                manifest["outcome"] = "no_supported_causal_direction"
            else:
                selected = stage("candidates", ["evaluate-candidates", *common, *candidate_protocol, "--dataset", dev_path,
                                                 "--discovery", discovery, "--parity", parity, "--output", search],
                                 search / "selection.json")
                if selected["status"] == "selected":
                    stage("reload", ["verify-reload", "--search", search], search / "reload-verification.json")
                    manifest["outcome"] = "candidate_ready_for_external_freeze"
                else:
                    manifest["outcome"] = "no_eligible_validation_candidate"
        atomic_json(manifest_path, manifest)
        print(json.dumps({"outcome": manifest["outcome"], "pilot": str(manifest_path)}, indent=2))
        return 0
    finally:
        lock.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Aya pilot stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
