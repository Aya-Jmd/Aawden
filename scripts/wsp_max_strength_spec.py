"""Build the maximum-strength edit the protocol budget allows, from measured opinion-attending heads.

Why this exists
    Three independent patches (astra-01, astra-02, fable-01) left misleading balanced accuracy at
    exactly 0.000. A null from a gentle edit is ambiguous: it can mean the hypothesis is wrong, or
    that the edit was too small. The forced-choice margins settle which. Under a misleading reviewer
    the correct answer sits a mean of 15.26 nats on the wrong side (all 12 measured cases negative,
    best case -7.37), while the strongest single causal component measured +0.604 nats. A patch that
    nudges by a fraction of a nat cannot cross a 15-nat gap however well it is targeted.

    So this builds the loudest edit the protocol permits: scale to zero every output column of the
    highest opinion-attending heads, i.e. delete those heads' contribution to the residual stream
    entirely. If the verdict still does not move, the budget is the binding constraint and no sparse
    edit at 0.05% of weights can close the gap -- which is a quantitative result, not a failed guess.

Selection rule (fixed before measurement, taken from the attention artifact, not from behaviour)
    Take heads in descending relative opinion mass across all measured layers, skipping a layer once
    it would push the covered-scalar total over the cap. Zero the whole 128-column block. No column is
    chosen by weight norm, and no discovery score is consulted: this is the attention evidence at
    maximum amplitude.

Run with AAWDEN_AYA_HF_TOKEN or HF_TOKEN set. Reads only public checkpoint bytes; no forward pass,
no dataset, no gradients.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/aya-expanse-8b"
REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
HEAD_DIM = 128
HIDDEN = 4096
# 0.05% of the model, the protocol cap. Kept as an explicit number so the selection loop can stop
# before it, rather than discovering the refusal at apply time.
FRACTION_LIMIT = 0.0005
TOTAL_PARAMETERS = 8028033024


def metadata(filename: str, token: str) -> dict:
    response = requests.get(BASE + filename, headers={"Authorization": f"Bearer {token}"}, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"Metadata {filename}: HTTP {response.status_code}")
    return response.json()


def byte_range(filename: str, start: int, end: int, token: str) -> bytes:
    response = requests.get(
        BASE + filename, params={"download": "true", "slice": f"{start}-{end}"},
        headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}",
                 "Accept-Encoding": "identity"}, timeout=120, stream=True)
    with response:
        if response.status_code != 206:
            raise RuntimeError(f"Range request: expected HTTP 206, got {response.status_code}")
        if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise RuntimeError("Server returned the wrong byte range")
        data = response.content
    if len(data) != end - start + 1:
        raise RuntimeError("Truncated checkpoint byte range")
    return data


def load_tensor(tensor_name: str, index: dict, token: str) -> torch.Tensor:
    """The pinned checkpoint stores F16; the harness loads BF16, so round through that dtype."""
    shard = index["weight_map"][tensor_name]
    header_size = struct.unpack("<Q", byte_range(shard, 0, 7, token))[0]
    if not 0 < header_size < 10_000_000:
        raise RuntimeError("Unexpected safetensors header size")
    header = json.loads(byte_range(shard, 8, 7 + header_size, token))
    entry = header[tensor_name]
    if entry["dtype"] not in ("F16", "BF16") or entry["shape"] != [HIDDEN, HIDDEN]:
        raise RuntimeError(f"Unexpected metadata for {tensor_name}")
    begin, end = entry["data_offsets"]
    if end - begin != HIDDEN * HIDDEN * 2:
        raise RuntimeError(f"Unexpected byte length for {tensor_name}")
    raw = byte_range(shard, 8 + header_size + begin, 8 + header_size + end - 1, token)
    stored = {"F16": torch.float16, "BF16": torch.bfloat16}[entry["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=stored).reshape(HIDDEN, HIDDEN).to(torch.bfloat16).clone()


def select_heads(attention: dict, cap: int) -> tuple[list[tuple[int, int]], list[dict]]:
    """Highest relative opinion mass first, stopping before the scalar cap. Returns (heads, trace)."""
    ranked = sorted(attention["entries"], key=lambda e: (-e["mean_relative_opinion_mass"], e["layer"], e["head"]))
    chosen, trace, used = [], [], 0
    seen_layers = {}
    for entry in ranked:
        layer, head = int(entry["layer"]), int(entry["head"])
        cost = HEAD_DIM * HIDDEN  # a complete head block: 128 columns x 4096 rows
        if used + cost > cap:
            trace.append({"layer": layer, "head": head, "status": "skipped",
                          "reason": f"would take covered scalars to {used + cost}, above the {cap} cap"})
            continue
        if (layer, head) in chosen:
            continue
        chosen.append((layer, head))
        used += cost
        seen_layers[layer] = seen_layers.get(layer, 0) + 1
        trace.append({"layer": layer, "head": head, "status": "chosen",
                      "relative_opinion_mass": entry["mean_relative_opinion_mass"],
                      "p_value": entry["relative_p_value"], "ci95": entry["relative_ci95"],
                      "covered_scalars_running_total": used})
    return chosen, trace


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, default=ROOT / "runs" / "wsp-v3" / "wsp-dump3" / "workbench")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "wsp-v3" / "proposals" / "max-strength-01.json")
    parser.add_argument("--report", type=Path, default=ROOT / "runs" / "wsp-v3" / "proposals" / "max-strength-01_build.json")
    # 0.0 ablates the head; 0.5 halves it; 1.0 would be the unedited weights. The same head set at
    # several amplitudes turns one data point into a dose-response curve, which is what distinguishes
    # "the heads carry the behaviour but the edit was too small" from "the heads do not carry it".
    parser.add_argument("--scale", type=float, default=0.0)
    args = parser.parse_args(argv)

    token = os.environ.get("AAWDEN_AYA_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import get_token
        token = get_token()
    if not token:
        raise SystemExit("Set AAWDEN_AYA_HF_TOKEN or HF_TOKEN to a token with gated access to the model")

    sites = json.loads((args.workbench / "sites.json").read_text())
    attention = json.loads((args.workbench / "attention.json").read_text())
    protocol = json.loads((args.workbench / "protocol.json").read_text())
    total = int(sites["total_parameters"])
    if total != TOTAL_PARAMETERS:
        raise SystemExit(f"Refused: workbench total_parameters {total} is not the pinned {TOTAL_PARAMETERS}")
    fraction = min(FRACTION_LIMIT, float(protocol["limits"]["max_changed_fraction"]))
    cap = int(fraction * total)

    heads, trace = select_heads(attention, cap)
    if not heads:
        raise SystemExit("Refused: no head fits the budget")
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    index = metadata("model.safetensors.index.json", token)
    edits, report_tensors = [], []
    for layer in sorted(by_layer):
        tensor_name = f"model.layers.{layer}.self_attn.o_proj.weight"
        columns = [j for head in sorted(by_layer[layer]) for j in range(head * HEAD_DIM, (head + 1) * HEAD_DIM)]
        if columns != sorted(set(columns)):
            raise SystemExit("Refused: head column blocks overlap or are unsorted")
        weight = load_tensor(tensor_name, index, token)
        original = weight[:, columns].clone()
        # scale 0.0 deletes the head's write into the residual stream; scale s multiplies it.
        # bf16 rounding is part of what is measured, so the reported changed count reflects stored bytes.
        replacement = (original.float() * args.scale).to(weight.dtype)
        edits.append({"tensor_name": tensor_name, "axis": 1, "indices": columns,
                      "values": replacement.float().T.tolist()})
        report_tensors.append({
            "tensor_name": tensor_name, "heads": sorted(by_layer[layer]), "columns": len(columns),
            "covered_scalars": int(original.numel()),
            "changed_scalars": int((replacement != original).sum()),
            "original_slice_l2": float(original.double().norm()),
            "weights_loaded_from_pinned_revision": True})

    covered = sum(t["covered_scalars"] for t in report_tensors)
    if covered > cap:
        raise SystemExit(f"Refused: {covered} covered scalars exceeds the {cap} cap")
    if len(edits) != len({e['tensor_name'] for e in edits}):
        raise SystemExit("Refused: more than one edit for a tensor")

    rationale = (
        "Maximum-strength control, not a hypothesis-driven proposal. Three independent sparse patches "
        "(astra-01: 64 high-norm columns, one head; astra-02: full blocks of the two strongest layer-18 "
        "opinion heads at half gain; fable-01: a closed-form 'not'-lens ablation over four heads) all "
        "left misleading balanced accuracy at exactly 0.000, with neutral moving at most +0.100 and the "
        "correct-opinion condition unchanged or improved. The measured forced-choice margins say why: "
        "under a misleading reviewer the correct answer sits a mean of 15.26 nats on the wrong side "
        "(all 12 cases negative; the best case still needs 7.37 nats), while the strongest single causal "
        "component measured +0.604 nats. This edit therefore asks the opposite question from the earlier "
        "rounds: not 'is this the right component' but 'is ANY edit of this size able to move the "
        f"verdict'. It sets to zero the entire output projection of the {len(heads)} highest "
        "relative-opinion-mass heads across "
        f"{len(by_layer)} layers ({covered} covered scalars, {covered / total * 100:.4f}% of the model, "
        f"{covered / cap * 100:.1f}% of the {fraction * 100:.2f}% cap), chosen from the attention "
        "artifact alone in descending opinion mass with no behavioural input. Deleting a head's write is "
        "far outside the amplitude of a calibrated patch and is expected to damage neutral and "
        "correct-opinion behaviour; that cost is the point, because a null under maximum strength is "
        "evidence about the budget rather than about targeting. Falsification: if misleading discovery "
        "balanced accuracy stays at 0.000 while neutral drops, the deference is not carried in the "
        "writing of these attention heads at any amplitude this protocol allows, and the honest "
        "conclusion is that a sparse weight patch at 0.05% of parameters cannot close a 15-nat gap."
    )
    spec = {"schema_version": 1, "rationale": rationale, "edits": edits}
    schema = json.loads((args.workbench / "schema.json").read_text())["json_schema"]
    import jsonschema
    jsonschema.Draft202012Validator(schema).validate(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, separators=(",", ":"), allow_nan=False) + "\n")

    report = {
        "spec": str(args.output), "schema_validation": "PASS",
        "heads_chosen": len(heads), "heads": heads, "layers": sorted(by_layer),
        "covered_scalars": covered, "cap_scalars": cap, "fraction_of_model": covered / total,
        "fraction_of_cap": covered / cap,
        "tensors": report_tensors, "selection_trace": trace,
        "forward_calls": 0, "behavioral_evaluation": "NOT RUN",
        "note": "Maximum-strength ablation. Any gain here is not evidence of a good edit; a null is evidence about the budget."}
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("heads_chosen", "heads", "layers", "covered_scalars",
                                             "cap_scalars", "fraction_of_cap")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
