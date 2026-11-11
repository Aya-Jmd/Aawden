"""Write a copy of a LoRA adapter with its update scaled by a constant.

A LoRA update is B @ A * (alpha / r), so multiplying every lora_B tensor by `scale` multiplies the
whole update by `scale` and nothing else. scale=1 reproduces the adapter; scale=0 is the base model.
Used for the round-2 strength sweep (docs/FIX_PROTOCOL.md, addendum R5). No GPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def scale_state(state: dict, scale: float) -> tuple[dict, int]:
    out, touched = {}, 0
    for name, tensor in state.items():
        if "lora_B" in name:
            out[name] = (tensor.float() * scale).to(tensor.dtype)
            touched += 1
        else:
            out[name] = tensor
    return out, touched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from safetensors.torch import load_file, save_file
    state = load_file(str(args.adapter / "adapter_model.safetensors"))
    scaled, touched = scale_state(state, args.scale)
    if not touched:
        raise SystemExit("no lora_B tensors found; refusing to write an unscaled copy")
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(scaled, str(args.output / "adapter_model.safetensors"))
    shutil.copy(args.adapter / "adapter_config.json", args.output / "adapter_config.json")
    (args.output / "scale.json").write_text(json.dumps(
        {"source": str(args.adapter), "scale": args.scale, "lora_B_tensors_scaled": touched}, indent=1) + "\n")
    print(f"[scale] {touched} lora_B tensors x {args.scale} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
