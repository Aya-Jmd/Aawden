"""Reproduce the frozen best neutral-review pipeline. Uses a local compatible GPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from security_wsp.artifacts import file_hash, read_json, source_fingerprint
from security_wsp.data import load_dataset, validate_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("configs/selected-review.json"))
    parser.add_argument("--dataset", type=Path, help="Defaults to the now-exposed 20-case confirmation set")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = read_json(args.selection)
    config = read_json(selection["model_config"])
    if config["revision"] != selection["base_revision"] or config["lora_adapter"]["sha256"] != selection["adapter_sha256"]:
        raise ValueError("The selected model identity changed")
    if file_hash(Path(config["lora_adapter"]["checkpoint"])) != selection["adapter_sha256"]:
        raise ValueError("The saved adapter bytes changed")
    dataset = args.dataset or Path(selection["confirmation_file"])
    if args.dataset is None and file_hash(dataset) != selection["confirmation_file_sha256"]:
        raise ValueError("The reproduction dataset changed")
    cases = load_dataset(dataset)
    validate_cases(cases)
    if args.output.exists():
        raise ValueError("Use a new output directory")
    from security_wsp.workflow import require_research_runtime
    from security_wsp.models import HuggingFaceAdapter
    from security_wsp.adapter_review import evaluate
    require_research_runtime()
    adapter = HuggingFaceAdapter(config)
    identity = {"selection_sha256": file_hash(args.selection), "dataset_sha256": file_hash(dataset),
                "source": source_fingerprint(), "scope": "Neutral reproduction; independent final review not asserted"}
    result = evaluate(cases, adapter, ("trace",), args.output, identity)
    print("Strict neutral balanced accuracy:", result["metrics"]["trace"]["balanced_accuracy"])


if __name__ == "__main__":
    main()
