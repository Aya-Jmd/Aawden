"""Run the frozen Aya review protocol on executable security fixtures."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import file_hash, read_json, verify_run
from security_wsp.calibrated_review import VERSION as CALIBRATION_VERSION, calibrated_request, calibrated_study
from security_wsp.consensus_review import VERSION as CONSENSUS_VERSION, consensus_study, verify_consensus
from security_wsp.evaluation import score_generation
from security_wsp.models import GenerationResult, HuggingFaceAdapter
from security_wsp.rigorous_review import CHECK_VERSION, checked_dataset, draft_requests, error_report, selected_request, study
from security_wsp.workflow import require_research_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/rigorous-review.json")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model-adapted.json")
    parser.add_argument("--verify", action="store_true", help="Reparse saved answers without inference")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if args.verify:
        print(verify_confirmation(args.output, protocol))
        return
    config = read_json(args.model_config)
    require_research_runtime()
    runner = {CALIBRATION_VERSION: calibrated_study, CONSENSUS_VERSION: consensus_study}.get(protocol["prompt_version"], study)
    result = runner(ROOT, HuggingFaceAdapter(config), protocol, args.output)
    if result["confirmation"] is None:
        print(result["reason"], flush=True)
    else:
        print("Strict confirmation:", result["confirmation"]["overall"], flush=True)


def verify_confirmation(output, protocol):
    if protocol["prompt_version"] == CONSENSUS_VERSION:
        return verify_consensus(ROOT, output, protocol)
    stage = output / "confirmation"
    integrity = verify_run(stage)
    frozen = read_json(output / "frozen.json")
    outcome = read_json(output / "outcome.json")
    model = read_json(stage / "summary.json")["model"]
    adapter_hash = frozen["adapter_sha256"]
    if frozen["protocol"] != protocol or model["adapter_sha256"] != adapter_hash:
        raise ValueError("Saved protocol or adapter differs")
    if protocol["prompt_version"] == CALIBRATION_VERSION and model["verdict_calibration"] != frozen["verdict_calibration"]:
        raise ValueError("Saved decoding calibration differs")
    if adapter_hash != protocol["adapter_sha256"]:
        training = read_json(output / "training.json")
        if (training["protocol"] != protocol["training"] or
                training["parent_adapter_sha256"] != protocol["adapter_sha256"] or
                training["adapter_sha256"] != adapter_hash or
                file_hash(output / "adapter.safetensors") != adapter_hash):
            raise ValueError("Refined checkpoint provenance differs")
    variant = frozen["selected_prompt"]
    if variant != outcome["selected_prompt"]:
        raise ValueError("Frozen prompt and reported prompt differ")
    cases = checked_dataset(ROOT, protocol["confirmation"])
    rows = read_json(stage / "results.json")
    if len(rows) != len(cases) or {row["case_id"] for row in rows} != {case.case_id for case in cases}:
        raise ValueError("Saved answers differ from the registered confirmation cases")
    by_id = {row["case_id"]: row for row in rows}
    make_request = draft_requests(output / "confirmation_drafts") if protocol["prompt_version"] == CHECK_VERSION else None
    if protocol["prompt_version"] == CALIBRATION_VERSION:
        make_request = lambda case, _: calibrated_request(case, frozen["source_method"])
    for case in cases:
        row = by_id[case.case_id]
        routed = variant[case.target_category] if isinstance(variant, dict) else variant
        request = make_request(case, routed) if make_request else selected_request(case, variant)
        result = GenerationResult(request.request_id, row["raw_output"], row["input_tokens"], row["output_tokens"],
                                  row["finish_reason"], row["generation_error"], row["latency_seconds"])
        method = "category_route" if isinstance(variant, dict) else variant
        rescored = score_generation(case, request, result, method, prompt_version=protocol["prompt_version"])
        if dict(rescored, storage_id=row["storage_id"]) != row:
            raise ValueError(f"Raw answer or prompt differs for {case.case_id}")
    report = error_report(rows)
    if report != outcome["confirmation"] or outcome["target_reached"] != (report["overall"]["accuracy"] == 1.0):
        raise ValueError("Saved aggregate differs from the rescored answers")
    return dict(integrity, cases=len(cases), accuracy=report["overall"]["accuracy"],
                errors=report["errors"], target_reached=outcome["target_reached"])


if __name__ == "__main__":
    main()
