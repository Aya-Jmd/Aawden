"""Direction edit with no gradients: ablate the reviewer-opinion direction (Arm 2, docs/FIX_PROTOCOL.md).

Why this exists
    Zeroing opinion-attending heads on Qwen2.5-14B was a clean null. A head is one object; the place a
    model stores "what the reviewer said" may instead be a DIRECTION in the residual stream spread over
    many heads. This tests that, without gradients, on a model that solves the task unaided.

Design (fixed in docs/FIX_PROTOCOL.md before measurement)
    1. OPINION DIRECTION, selection set only. For every case, one reviewer condition asserts
       "vulnerable" and the other asserts "clean", on byte-identical code. Their residual difference at
       the answer position, averaged over cases, is the opinion direction. The code cancels by
       construction: it is the same in both prompts.
    2. GUARD. The code-truth direction (neutral prompts; mean vulnerable minus mean clean) is projected
       OUT of the opinion direction, so the edit cannot delete the code signal it is meant to protect.
       The cosine between the two is reported per layer: near-parallel means the opinion IS the readout
       and no single-direction edit can separate them.
    3. EDIT. The guarded direction r is removed from everything WRITTEN into the residual stream: the
       outputs of the embedding, of every o_proj and of every down_proj, x <- x - (x.r) r. Since those
       maps are linear, removing r from their outputs is the same function as orthogonalising their
       weights against r -- a weight edit. (Hooking only decoder-layer outputs would NOT be equivalent:
       inside a layer the MLP reads the attention write before such a hook runs.) The equivalence is CHECKED at the end, not assumed: the chosen arm is re-run
       with the weights actually edited and the verdicts compared.
    4. LAYER CHOICE on the selection set only: each candidate source layer's direction is scored by
       flip rate on the selection set, subject to solved-unaided staying within 3 of unedited.
    5. CONTROLS: three random directions (seeds 17/29/43), guarded the same way, and the UNGUARDED
       opinion direction, to show what the guard buys.
    6. EVALUATE once on the plain set (family-disjoint from selection), all three instruments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from security_wsp.artifacts import atomic_json, file_hash, source_fingerprint  # noqa: E402
from security_wsp.data import CLEAN, VULNERABLE, load_dataset  # noqa: E402


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- pure maths (unit-tested)
def unit(v):
    return v / v.norm().clamp_min(1e-12)


def guard(direction, protect):
    """Remove `protect`'s component from `direction`, then normalise."""
    p = unit(protect)
    return unit(direction - (direction @ p) * p)


def ablate(h, r):
    """h - (h.r) r over the last dimension, computed in fp32 and cast back."""
    hf = h.float()
    return (hf - (hf @ r)[..., None] * r).to(h.dtype)


# ---------------------------------------------------------------- residual capture
def answer_residuals(model, tokenizer, cases, condition, cross, prefill):
    """Residual at the last prompt token for every decoder layer output -> [cases, layers, hidden]."""
    import torch
    rows = []
    for case in cases:
        text, _ = cross.render_for(tokenizer, case, condition, prefill)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        with torch.inference_mode():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        # hidden_states[0] is the embedding, [1..n] the decoder layer outputs. The last one has the
        # final norm applied on some architectures, so it is excluded from the candidate grid below.
        rows.append(torch.stack([h[0, -1].float().cpu() for h in out.hidden_states[1:]]))
        del out
    return torch.stack(rows)


def asserted(case, condition):
    if condition == "correct_suggestion":
        return case.oracle_verdict
    return CLEAN if case.oracle_verdict == VULNERABLE else VULNERABLE


# ---------------------------------------------------------------- hooks
class DirectionAblation:
    def __init__(self, model, r):
        self.model, self.r, self.handles = model, r, []

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            return (ablate(output[0], self.r.to(output[0].device)),) + tuple(output[1:])
        return ablate(output, self.r.to(output.device))

    def __enter__(self):
        inner = self.model.model
        self.handles.append(inner.embed_tokens.register_forward_hook(self._hook))
        for layer in inner.layers:
            self.handles.append(layer.self_attn.o_proj.register_forward_hook(self._hook))
            self.handles.append(layer.mlp.down_proj.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def orthogonalise_weights(model, r):
    """The weight-space form of the same edit. Returns CPU copies for exact restoration."""
    import torch
    saved = {}
    inner = model.model
    head = getattr(model, "lm_head", None)
    if head is not None and head.weight.data_ptr() == inner.embed_tokens.weight.data_ptr():
        # Tied embeddings (Cohere/Aya): editing the embedding in place would ALSO edit the output head,
        # which the hooked edit never touches. Untie first so the two forms stay the same function.
        head.weight = torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)
        saved["untied_lm_head"] = True
    with torch.no_grad():
        emb = inner.embed_tokens.weight                     # [vocab, hidden]: rows write the residual
        saved["embed"] = emb.detach().cpu().clone()
        rr = r.to(emb.device)
        emb.copy_(ablate(emb, rr))
        for i, layer in enumerate(inner.layers):
            for name, lin in (("o", layer.self_attn.o_proj), ("down", layer.mlp.down_proj)):
                w = lin.weight                              # [hidden_out, in]: columns write the residual
                saved[f"{name}{i}"] = w.detach().cpu().clone()
                rd = r.to(w.device, torch.float32)
                wf = w.float()
                w.copy_((wf - rd[:, None] * (rd @ wf)[None, :]).to(w.dtype))
                if getattr(lin, "bias", None) is not None:
                    b = lin.bias
                    saved[f"{name}{i}.bias"] = b.detach().cpu().clone()
                    b.copy_(ablate(b, rd))
    return saved


def restore_weights(model, saved):
    import torch
    inner = model.model
    with torch.no_grad():
        inner.embed_tokens.weight.copy_(saved["embed"])
        for i, layer in enumerate(inner.layers):
            for name, lin in (("o", layer.self_attn.o_proj), ("down", layer.mlp.down_proj)):
                lin.weight.copy_(saved[f"{name}{i}"])
                if f"{name}{i}.bias" in saved:
                    lin.bias.copy_(saved[f"{name}{i}.bias"])


# ---------------------------------------------------------------- measurement
def measure(model, tokenizer, cases, specs, context_limit, cross, flip, localize) -> dict:
    table = {}
    for condition in ("neutral", "correct_suggestion", "incorrect_suggestion"):
        table[condition], _ = cross.margin_table(model, tokenizer, cases, condition, specs, context_limit,
                                                 localize)
    full = {"conditions": table}
    return {"table": full, "by_instrument": {name: flip.analyse(full, name) for name in specs}}


def verdicts(table: dict, instrument: str) -> dict:
    out = {}
    for condition, block in table["conditions"].items():
        spec = block[instrument]
        for cid, e in spec["cases"].items():
            out[(cid, condition)] = spec["verdict_by_continuation"][max(e["scores"], key=e["scores"].get)]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--revision", default="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
    parser.add_argument("--select-dataset", type=Path, default=ROOT / "datasets" / "wsp-discovery.json")
    parser.add_argument("--eval-dataset", type=Path, default=ROOT / "datasets" / "plain-pooled.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--layer-step", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cross = _load("_cross", ROOT / "scripts" / "wsp_cross_model.py")
    flip = _load("_flip", ROOT / "scripts" / "wsp_flip_analysis.py")
    localize = cross.load_localize()
    specs = localize.instrument_specs()
    prefill = specs["json_prefill"]["suffix"]
    json_only = {"json_prefill": specs["json_prefill"]}

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True,
        device_map="auto" if torch.cuda.device_count() > 1 else None)
    if torch.cuda.device_count() <= 1:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_layers = model.config.num_hidden_layers

    select = sorted(load_dataset(args.select_dataset), key=lambda c: c.case_id)
    evaluate = sorted(load_dataset(args.eval_dataset), key=lambda c: c.case_id)
    if args.limit:
        select, evaluate = select[: args.limit], evaluate[: args.limit]
    assert not {c.family_id for c in select} & {c.family_id for c in evaluate}, \
        "selection and evaluation share a family"

    # ---- 1-2: directions on the selection set
    print(f"[dir] capturing residuals on {len(select)} selection cases", flush=True)
    res = {c: answer_residuals(model, tokenizer, select, c, cross, prefill)
           for c in ("neutral", "correct_suggestion", "incorrect_suggestion")}
    says_vuln = torch.stack([res["correct_suggestion"][i] if asserted(case, "correct_suggestion") == VULNERABLE
                             else res["incorrect_suggestion"][i] for i, case in enumerate(select)])
    says_clean = torch.stack([res["incorrect_suggestion"][i] if asserted(case, "correct_suggestion") == VULNERABLE
                              else res["correct_suggestion"][i] for i, case in enumerate(select)])
    opinion = (says_vuln - says_clean).mean(0)                       # [layers, hidden]
    vuln_mask = torch.tensor([c.oracle_verdict == VULNERABLE for c in select])
    code = res["neutral"][vuln_mask].mean(0) - res["neutral"][~vuln_mask].mean(0)
    cosine = [float(unit(opinion[l]) @ unit(code[l])) for l in range(n_layers)]
    print(f"[dir] cos(opinion, code) by layer: {[round(c, 2) for c in cosine]}", flush=True)

    # ---- 4: choose the source layer on the selection set
    unedited_sel = measure(model, tokenizer, select, json_only, args.context_limit, cross, flip, localize)
    base = unedited_sel["by_instrument"]["json_prefill"]["all"]
    print(f"[dir] selection unedited: solved {base['solved_unaided']} flip {base['flip_rate']}", flush=True)
    grid = list(range(max(1, n_layers // 5), n_layers - 1, args.layer_step))
    candidates = []
    for layer in grid:
        r = guard(opinion[layer], code[layer]).to(torch.float32)
        with DirectionAblation(model, r.to(model.device)):
            res_l = measure(model, tokenizer, select, json_only, args.context_limit, cross, flip, localize)
        a = res_l["by_instrument"]["json_prefill"]["all"]
        candidates.append({"layer": layer, "solved_unaided": a["solved_unaided"], "flip_rate": a["flip_rate"],
                           "cos_opinion_code": cosine[layer]})
        print(f"[dir] layer {layer}: solved {a['solved_unaided']} flip {a['flip_rate']}", flush=True)
    eligible = [c for c in candidates
                if c["solved_unaided"] >= base["solved_unaided"] - 3 and c["flip_rate"] is not None]
    chosen = min(eligible or candidates,
                 key=lambda c: (c["flip_rate"] if c["flip_rate"] is not None else 9, -c["solved_unaided"]))
    L = chosen["layer"]
    print(f"[dir] chosen source layer {L} (eligible {len(eligible)}/{len(candidates)})", flush=True)

    # ---- 5: arms
    hidden = opinion.shape[1]
    arms = {"unedited": None,
            "opinion_guarded": guard(opinion[L], code[L]),
            "opinion_unguarded": unit(opinion[L])}
    for seed in (int(s) for s in args.seeds.split(",")):
        g = torch.Generator().manual_seed(seed)
        arms[f"random_seed{seed}"] = guard(torch.randn(hidden, generator=g), code[L])

    report = {"model_id": args.model_id, "revision": args.revision, "layers": n_layers,
              "select_dataset_sha256": file_hash(args.select_dataset),
              "eval_dataset_sha256": file_hash(args.eval_dataset),
              "source_fingerprint": source_fingerprint(),
              "cos_opinion_code_by_layer": cosine, "selection_unedited": base,
              "selection_candidates": candidates, "chosen_layer": L,
              "eligible_candidates": len(eligible), "arms": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, r in arms.items():
        if r is None:
            result = measure(model, tokenizer, evaluate, specs, args.context_limit, cross, flip, localize)
        else:
            with DirectionAblation(model, r.to(torch.float32).to(model.device)):
                result = measure(model, tokenizer, evaluate, specs, args.context_limit, cross, flip, localize)
        tables[name] = result["table"]
        atomic_json(args.output / f"table-{name}.json", result["table"])
        report["arms"][name] = {"by_instrument": result["by_instrument"],
                                "cos_with_code": None if r is None else float(unit(r) @ unit(code[L])),
                                "cos_with_opinion": None if r is None else float(unit(r) @ unit(opinion[L]))}
        brief = {i: (v["all"]["solved_unaided"], v["all"]["flip_rate"]) for i, v in result["by_instrument"].items()}
        print(f"[dir] {name}: (solved, flip) {brief}", flush=True)
        atomic_json(args.output / "direction.json", report)

    # ---- 3: the weight-space form must reproduce the hooked arm
    r = arms["opinion_guarded"].to(torch.float32)
    saved = orthogonalise_weights(model, r)
    weight_result = measure(model, tokenizer, evaluate, json_only, args.context_limit, cross, flip, localize)
    restore_weights(model, saved)
    hooked_v = verdicts(tables["opinion_guarded"], "json_prefill")
    weight_v = verdicts(weight_result["table"], "json_prefill")
    agree = sum(hooked_v[k] == weight_v[k] for k in hooked_v)
    report["weight_edit_check"] = {
        "edited_tensors": len([k for k in saved if not k.endswith(".bias")]),
        "verdict_agreement": agree, "verdict_cells": len(hooked_v),
        "by_instrument": weight_result["by_instrument"]}
    print(f"[dir] weight-space edit reproduces hooked verdicts on {agree}/{len(hooked_v)} cells", flush=True)
    torch.save({"direction": r.cpu(), "layer": L, "model_id": args.model_id, "revision": args.revision},
               args.output / "direction.pt")
    atomic_json(args.output / "direction.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
