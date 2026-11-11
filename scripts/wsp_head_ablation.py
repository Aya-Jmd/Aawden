"""Sparse weight edit on a model that HAS the skill: ablate opinion-attending heads, with random controls.

Why this exists
    The Aya-8B weight edits could not work for a reason found later: Aya-8B is at chance on plain code
    review with no reviewer present, so there is no competent answer for an edit to uncover. The right
    test bed is a model that solves cases unaided and is still talked out of them. Qwen2.5-14B-Instruct
    is that model (solves 68-86% of the plain set unaided, a wrong reviewer flips 52-76% of those).

Design (fixed before measurement)
    1. SELECT on the trick set (wsp-discovery): under `incorrect_suggestion`, measure per-head attention
       mass from the final prompt position to the reviewer-opinion span, relative to the span's share of
       the prompt. Rank heads by the mean of that ratio.
    2. EDIT: zero the o_proj input columns of the top-k heads, k = the largest number of whole heads
       inside the protocol budget (0.05% of parameters). This deletes those heads' write into the
       residual stream. Same budget rule as the Aya study.
    3. CONTROLS: three random head sets of the same size (seeds 17/29/43), excluding the chosen heads.
    4. EVALUATE on the plain set (plain-pooled), which shares no family with the selection set: neutral
       competence (solved unaided), flip rate under a wrong reviewer on solved cases, and correct-
       reviewer accuracy, on all three answer instruments.

    Success = the chosen-head edit lowers the flip rate more than every random control while neutral
    solved stays within 5 cases of the unedited model. Anything else is reported as it comes out.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from security_wsp.artifacts import atomic_json, file_hash, source_fingerprint  # noqa: E402
from security_wsp.data import load_dataset  # noqa: E402

FRACTION_LIMIT = 0.0005
OPINION_MARKER = "Reviewer assessment (opinion only):"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def opinion_attention(model, tokenizer, cases, cross, prefill: str) -> dict:
    """Mean relative attention mass on the opinion span, per (layer, head), at the last prompt token."""
    import torch
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    total = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    used = 0
    for case in cases:
        text, _ = cross.render_for(tokenizer, case, "incorrect_suggestion", prefill)
        start = text.find(OPINION_MARKER)
        if start < 0:
            continue
        end = text.find("\n", start)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        span = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not span:
            continue
        ids = enc["input_ids"].to(model.device)
        with torch.inference_mode():
            out = model(input_ids=ids, output_attentions=True, use_cache=False)
        share = len(span) / ids.shape[1]
        for layer, att in enumerate(out.attentions):
            mass = att[0, :, -1, span].float().sum(-1).double().cpu()  # [heads]
            total[layer] += mass / share
        used += 1
        del out
    mean = total / max(used, 1)
    return {"cases": used, "mean_relative_mass": mean.tolist()}


def apply_ablation(model, heads, head_dim: int, originals: dict) -> int:
    import torch
    changed = 0
    with torch.no_grad():
        for layer, head in heads:
            weight = model.model.layers[layer].self_attn.o_proj.weight
            if layer not in originals:
                originals[layer] = weight.detach().clone()
            cols = slice(head * head_dim, (head + 1) * head_dim)
            changed += int((weight[:, cols] != 0).sum())
            weight[:, cols] = 0
    return changed


def restore(model, originals: dict) -> None:
    import torch
    with torch.no_grad():
        for layer, saved in originals.items():
            model.model.layers[layer].self_attn.o_proj.weight.copy_(saved)
    originals.clear()


def measure(model, tokenizer, cases, specs, context_limit, cross, flip) -> dict:
    table = {}
    for condition in ("neutral", "correct_suggestion", "incorrect_suggestion"):
        table[condition], _ = cross.margin_table(model, tokenizer, cases, condition, specs, context_limit,
                                                 cross.load_localize())
    full = {"conditions": table}
    return {"by_instrument": {name: flip.analyse(full, name) for name in specs},
            "balanced_accuracy_json": cross.balanced_accuracy(table, cases)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--revision", default="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
    parser.add_argument("--adapter-id", default=None,
                        help="LoRA merged into the base before selection/ablation, so o_proj columns "
                             "are the adapted model's own weights (used for the distilled Aya arm)")
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--select-dataset", type=Path, default=ROOT / "datasets" / "wsp-discovery.json")
    parser.add_argument("--eval-dataset", type=Path, default=ROOT / "datasets" / "plain-pooled.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cross = _load("_cross", ROOT / "scripts" / "wsp_cross_model.py")
    flip = _load("_flip", ROOT / "scripts" / "wsp_flip_analysis.py")
    localize = cross.load_localize()
    specs = localize.instrument_specs()
    prefill = specs["json_prefill"]["suffix"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True,
        device_map="auto", attn_implementation="eager")
    if args.adapter_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_id, revision=args.adapter_revision)
        model = model.merge_and_unload()
    model.eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    total_params = sum(p.numel() for p in model.parameters())
    per_head = cfg.hidden_size * head_dim
    k = max(1, int(FRACTION_LIMIT * total_params) // per_head)
    print(f"[ablate] {total_params} params, {per_head} scalars/head, budget {k} heads", flush=True)

    select_cases = sorted(load_dataset(args.select_dataset), key=lambda c: c.case_id)
    eval_cases = sorted(load_dataset(args.eval_dataset), key=lambda c: c.case_id)
    if args.limit:
        select_cases, eval_cases = select_cases[: args.limit], eval_cases[: args.limit]
    select_families = {c.family_id for c in select_cases}
    assert not select_families & {c.family_id for c in eval_cases}, "selection and evaluation share a family"

    print("[ablate] measuring opinion attention on the selection set", flush=True)
    attention = opinion_attention(model, tokenizer, select_cases, cross, prefill)
    mass = attention["mean_relative_mass"]
    ranked = sorted(((mass[l][h], l, h) for l in range(cfg.num_hidden_layers)
                     for h in range(cfg.num_attention_heads)), reverse=True)
    chosen = [(l, h) for _, l, h in ranked[:k]]
    print(f"[ablate] chosen heads {chosen}", flush=True)

    all_heads = [(l, h) for l in range(cfg.num_hidden_layers) for h in range(cfg.num_attention_heads)]
    pool = [x for x in all_heads if x not in set(chosen)]
    arms = {"unedited": [], "opinion_heads": chosen}
    for seed in (int(s) for s in args.seeds.split(",")):
        arms[f"random_seed{seed}"] = sorted(random.Random(seed).sample(pool, k))

    report = {"model_id": args.model_id, "revision": args.revision,
              "adapter_id": args.adapter_id, "adapter_revision": args.adapter_revision,
              "total_parameters": total_params,
              "budget_fraction": FRACTION_LIMIT, "heads_per_arm": k, "scalars_per_head": per_head,
              "select_dataset_sha256": file_hash(args.select_dataset),
              "eval_dataset_sha256": file_hash(args.eval_dataset),
              "source_fingerprint": source_fingerprint(),
              "top_heads": [{"layer": l, "head": h, "relative_mass": m} for m, l, h in ranked[:3 * k]],
              "attention_cases": attention["cases"], "arms": {}}
    originals: dict = {}
    for name, heads in arms.items():
        changed = apply_ablation(model, heads, head_dim, originals)
        print(f"[ablate] arm {name}: {len(heads)} heads, {changed} scalars zeroed", flush=True)
        result = measure(model, tokenizer, eval_cases, specs, args.context_limit, cross, flip)
        restore(model, originals)
        report["arms"][name] = {"heads": heads, "changed_scalars": changed,
                                "fraction_of_model": changed / total_params, **result}
        brief = {i: (r["all"]["solved_unaided"], r["all"]["flip_rate"])
                 for i, r in result["by_instrument"].items()}
        print(f"[ablate] {name}: (solved, flip) by instrument {brief}", flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "ablation.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
