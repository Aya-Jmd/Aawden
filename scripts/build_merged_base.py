"""Build the protocol-v3 route (b) edit base: Aya Expanse 8B with the retrained v7 LoRA merged dense.

Writes a model config that the rest of the pipeline consumes unchanged:
  * model ids / revision / decoding copied from configs/model.json
  * lora_adapter spec pointing at the retrained checkpoint, with its exact sha256 and the pinned
    base shard hashes from configs/model-adapted.json (load_adapter refuses a different base)
  * merge_lora=true, so HuggingFaceAdapter folds the update into plain weights and tensor names stay
    model.layers.N.<proj>.weight for the sparse patch format.

Usage:
  python scripts/build_merged_base.py --adapter runs/wsp-v3/retrain-chain/step-v7/ckpt/v7.safetensors \
      --output configs/model-merged-v7.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import atomic_json, file_hash, read_json  # noqa: E402

BASE_CONFIG = ROOT / "configs" / "model.json"
ADAPTED_CONFIG = ROOT / "configs" / "model-adapted.json"


def build(adapter: Path, rank=16, alpha=32, target_modules=None, adapted_config=ADAPTED_CONFIG):
    adapted = read_json(adapted_config)
    reference = adapted["lora_adapter"]
    if not adapter.is_file():
        raise SystemExit(f"Refused: adapter checkpoint missing: {adapter}")
    spec = {
        # Always repo-relative: an absolute path from the build machine does not exist in a job
        # container, and load_adapter hashes the file before any generation runs.
        "checkpoint": str(Path(adapter).resolve().relative_to(ROOT)) if Path(adapter).resolve().is_relative_to(ROOT) else str(adapter),
        "sha256": file_hash(adapter),
        "rank": int(rank),
        "alpha": int(alpha),
        "target_modules": list(target_modules or reference["target_modules"]),
        "base_checkpoint_hashes": dict(reference["base_checkpoint_hashes"]),
    }
    config = read_json(BASE_CONFIG)
    config.update(lora_adapter=spec, merge_lora=True,
                  base_route="b_merged_adapter",
                  merged_base_note=("Protocol v3 route (b): the edit base is the retrained LoRA merged "
                                    "into dense weights, so a sparse patch applies to plain tensor names."))
    config.pop("adapted_from", None)
    config["adapted_from"] = {"adapter_sha256": spec["sha256"], "rank": spec["rank"], "alpha": spec["alpha"]}
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "model-merged-v7.json")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--adapted-config", type=Path, default=ADAPTED_CONFIG)
    args = parser.parse_args(argv)
    adapter = args.adapter if args.adapter.is_absolute() else ROOT / args.adapter
    config = build(adapter, args.rank, args.alpha, adapted_config=args.adapted_config)
    atomic_json(args.output, config)
    print(json.dumps({"output": str(args.output), "adapter_sha256": config["lora_adapter"]["sha256"],
                      "rank": config["lora_adapter"]["rank"], "merge_lora": config["merge_lora"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
