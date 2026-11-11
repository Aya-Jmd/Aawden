"""LoRA SFT of a student on teacher traces -- a hand-written loop, because the cluster has no trl.

Why this exists
    Distillation is the only way to move competence across models whose architectures do not line up
    (Aya 8B: 32x4096; Aya 32B: 40x8192; no tensor is copyable). The Aya-8B weight edits in this
    project were uninterpretable because the base model is at chance on code review -- there was no
    competent behaviour for an edit to uncover. This trains the behaviour in from teacher traces, so
    the deference question can finally be asked of a small model that has the skill.

What is trained
    Loss is computed on COMPLETION tokens only: the prompt is masked to -100. Training on prompt
    tokens would teach the student to imitate question formats rather than answers.

    LoRA only (no full fine-tune) so the result is an adapter that can be merged, shipped, and
    measured with `wsp_cross_model.py --adapter-id`, and so a null result stays auditable.

Reporting
    Writes a loss curve and a manifest (trace file hashes, hyperparameters, peak memory). A run that
    silently diverges must be visible as such, so every N steps it records the raw mean loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def final_verdict(text: str) -> str | None:
    import re
    found = re.findall(r'"verdict"\s*:\s*"([a-z_]+)"', text)
    return found[-1] if found else None


def load_traces(paths: list[Path], limit: int | None) -> tuple[list[dict], dict]:
    """Keep only traces the oracle agrees with (rejection sampling).

    The teacher is wrong on a sizeable share of review cases; training on those would teach the
    student wrong answers. Traces with no oracle (code corpus) are kept as-is.
    """
    rows, stats = [], {"read": 0, "kept": 0, "dropped_wrong": 0, "dropped_no_verdict": 0}
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats["read"] += 1
            if row.get("oracle"):
                verdict = final_verdict(row["completion"])
                if verdict is None:
                    stats["dropped_no_verdict"] += 1
                    continue
                if verdict != row["oracle"]:
                    stats["dropped_wrong"] += 1
                    continue
            rows.append(row)
    if limit:
        rows = rows[:limit]
    stats["kept"] = len(rows)
    return rows, stats


def build_example(tok, row: dict, max_len: int):
    """Return input_ids and labels with the prompt masked.

    A row with a `prefill` (the anti-deference arm) is tokenised exactly as the evaluator scores it:
    `wsp_localize.continuation_scores` tokenises prefix+continuation as ONE string and takes the tokens
    past the prefix. Tokenising the two halves separately could split the boundary differently and
    train tokens the evaluator never scores. Such rows get no EOS: the evaluator stops at the verdict.
    """
    # A row may carry a whole conversation (`messages`, round 3 prior-answer rows); otherwise it is the
    # usual system + user pair.
    messages = row.get("messages") or [{"role": "system", "content": row["system"]},
                                       {"role": "user", "content": row["prompt"]}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if row.get("prefill") is not None:
        prefix = prompt_text + row["prefill"]
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        full = tok(prefix + row["completion"], add_special_tokens=False)["input_ids"]
        if full[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"{row.get('key')}: completion is not a token-level extension of the prefix")
        ids = full[:max_len]
        labels = ([-100] * len(prefix_ids) + full[len(prefix_ids):])[:max_len]
        return ids, labels
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tok(row["completion"], add_special_tokens=False)["input_ids"]
    completion_ids.append(tok.eos_token_id)
    ids = (prompt_ids + completion_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
    return ids, labels


def collate(batch, pad_id: int):
    import torch
    width = max(len(b[0]) for b in batch)
    input_ids, labels, attn = [], [], []
    for ids, lbl in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lbl + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="CohereLabs/aya-expanse-8b")
    parser.add_argument("--revision", default="5062468bf9bc0c6035fd64e06274333ec127d980")
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="adapter output directory")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17, help="data order and LoRA initialisation")
    parser.add_argument("--device-map", default=None,
                        help="'auto' to shard a base that does not fit one card (e.g. Aya Expanse 32B)")
    parser.add_argument("--max-gpu-gib", type=int, default=None,
                        help="per-card cap with --device-map auto, so shards balance instead of filling card 0")
    parser.add_argument("--init-adapter", type=Path, default=None,
                        help="continue training this saved adapter instead of starting a fresh one "
                             "(round 6 curriculum: learn the task first, then the reviewer conditions)")
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, filter_stats = load_traces(args.traces, args.limit)
    print(f"[sft] oracle filter: {filter_stats}", flush=True)
    if not rows:
        print("[sft] no traces", flush=True)
        return 1
    print(f"[sft] {len(rows)} traces from {len(args.traces)} file(s)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True,
        device_map=args.device_map,
        max_memory=({i: f"{args.max_gpu_gib}GiB" for i in range(torch.cuda.device_count())}
                    if args.device_map and args.max_gpu_gib else None))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.device_map is None:
        model.to("cuda")

    cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    torch.manual_seed(args.seed)   # LoRA A-matrix init draws from the global generator
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    examples = []
    skipped = 0
    for r in rows:
        ids, lbl = build_example(tok, r, args.max_len)
        floor = 1 if r.get("prefill") is not None else 4   # a prefill row's target is one verdict
        if sum(1 for x in lbl if x != -100) < floor:        # completion too short/truncated to learn from
            skipped += 1
            continue
        examples.append((ids, lbl))
    print(f"[sft] usable {len(examples)} (skipped {skipped})", flush=True)
    if not examples:
        return 1

    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * args.warmup_ratio))
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    import random
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    model.train()
    losses, t0, step, accum = [], time.time(), 0, 0
    batch = []
    for epoch in range(max(1, int(math.ceil(args.epochs)))):
        rng.shuffle(order)
        for idx in order:
            if step >= total_steps:
                break
            batch.append(examples[idx])
            if len(batch) < args.batch_size:
                continue
            ids, attn, labels = collate(batch, tok.pad_token_id)
            ids, attn, labels = ids.cuda(), attn.cuda(), labels.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            losses.append(float(out.loss.detach()))
            batch.clear()
            accum += 1
            if accum >= args.grad_accum:
                for g in optim.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                if step % args.log_every == 0:
                    recent = sum(losses[-args.log_every * args.grad_accum:]) / len(
                        losses[-args.log_every * args.grad_accum:])
                    print(f"[sft] step {step}/{total_steps} loss {recent:.4f} "
                          f"lr {lr_at(step):.2e} ({(time.time()-t0)/60:.1f} min)", flush=True)
                    args.output.mkdir(parents=True, exist_ok=True)
                    (args.output / "loss.json").write_text(json.dumps(
                        {"steps": step, "total_steps": total_steps, "losses": losses}) + "\n")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tok.save_pretrained(str(args.output))
    meta = {"base": args.base, "revision": args.revision, "traces": [str(p) for p in args.traces],
            "trace_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.traces},
            "examples": len(examples), "skipped": skipped, "oracle_filter": filter_stats, "epochs": args.epochs, "lr": args.lr,
            "seed": args.seed, "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum, "max_len": args.max_len,
            "steps": step, "final_loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4),
            "peak_cuda_gb": round(max(torch.cuda.max_memory_allocated(i)
                                      for i in range(torch.cuda.device_count())) / 1e9, 2),
            "device_map": args.device_map,
            "init_adapter": str(args.init_adapter) if args.init_adapter else None,
            "elapsed_minutes": round((time.time() - t0) / 60, 2)}
    (args.output / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[sft] done: {step} steps, final loss {meta['final_loss']}, "
          f"peak {meta['peak_cuda_gb']} GB -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
