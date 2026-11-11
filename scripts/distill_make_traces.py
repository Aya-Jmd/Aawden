"""Generate distillation traces from a strong teacher model.

Why this exists
    Copying weights between differently-shaped models is impossible (Aya 8B: 32 layers x 4096;
    Aya 32B: 40 layers x 8192; nothing lines up). Distillation on *data* is the only route that
    crosses that gap, and it is the route the competence problem needs: Aya 8B is at chance on code
    review, so it has no correct behaviour for a weight edit to uncover. This generates the teacher's
    reasoning traces so the student can learn the behaviour from the outside.

Two corpora
    --corpus review    teacher answers a security-review question: reasoning, then a JSON verdict.
                       Used to teach the task the safety study depends on.
    --corpus code      teacher writes a solution to a task statement. Used to teach raw coding.

    Both write JSONL with `prompt` / `completion` / provenance so the SFT script needs no join.

Contamination rule
    Training prompts come from a `--dataset` the caller guarantees is family-disjoint from whatever
    will be used for evaluation. This script asserts that the trace file records the source hash, so
    an accidental overlap is visible after the fact rather than silent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVIEW_SYSTEM = (
    "You are a security code reviewer. Read the code carefully and decide whether it is vulnerable. "
    "Work through the data flow before you conclude: where does untrusted input enter, and can it "
    "reach a dangerous operation without being constrained? State the reasoning, then give the "
    'verdict as JSON: {"verdict": "vulnerable"} or {"verdict": "not_vulnerable_for_target"}.'
)

CODE_SYSTEM = (
    "You are an expert Python programmer. Write correct, complete, runnable code for the task. "
    "Return only the code in one markdown block."
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def review_prompt(case: dict) -> str:
    return (f"Code to review ({case.get('language', 'python')}):\n\n```\n{case['code']}\n```\n\n"
            f"Threat model: {case['threat_model']}\n\nIs this code vulnerable? Reason first, "
            f"then give the verdict as JSON.")


def code_prompt(row: dict) -> str:
    return row["prompt"] if row["prompt"].strip() else row["task_id"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="CohereLabs/North-Mini-Code-1.0")
    parser.add_argument("--revision", default="d11e61a842617a22dc328552fa5bb86231ee4f37")
    parser.add_argument("--corpus", default="review", choices=["review", "code"])
    parser.add_argument("--dataset", type=Path, required=True,
                        help="our JSON fixtures (review corpus) -- must be eval-disjoint")
    parser.add_argument("--benchmark", default=None, choices=["humaneval", "mbpp"],
                        help="code corpus source: a public benchmark's TRAIN side")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="prompts per forward; batch amortises weight reads on a sharded teacher")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- build the prompt list
    items: list[dict] = []
    if args.corpus == "review":
        raw = json.loads(args.dataset.read_text())
        cases = raw["cases"] if isinstance(raw, dict) else raw
        for c in cases:
            items.append({"key": c["case_id"], "family": c.get("family_id"),
                          "prompt": review_prompt(c), "system": REVIEW_SYSTEM,
                          "oracle": c["oracle_verdict"]})
    else:
        from datasets import load_dataset
        name = "openai/openai_humaneval" if args.benchmark == "humaneval" else "google-research-datasets/mbpp"
        split = args.split
        ds = load_dataset(name, split=split)
        for i, r in enumerate(ds):
            prompt = r["prompt"] if args.benchmark == "humaneval" else r["text"]
            items.append({"key": f"{args.benchmark}/{i}", "family": args.benchmark,
                          "prompt": prompt, "system": CODE_SYSTEM, "oracle": None})
    if args.limit:
        items = items[: args.limit]
    print(f"[trace] {args.corpus}: {len(items)} prompts from "
          f"{args.dataset if args.corpus=='review' else args.benchmark}", flush=True)

    # ---- load teacher
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True,
        device_map="auto")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.open("w")
    t0 = time.time()
    kept = 0
    # Left padding so generation can be batched; the strip length per row keeps the completion clean.
    tok.padding_side = "left"
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        texts = []
        for item in chunk:
            messages = [{"role": "system", "content": item["system"]},
                        {"role": "user", "content": item["prompt"]}]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 pad_token_id=tok.pad_token_id)
        prompt_width = enc["input_ids"].shape[1]
        for item, row in zip(chunk, gen):
            completion = tok.decode(row[prompt_width:], skip_special_tokens=True)
            record = {"key": item["key"], "family": item["family"], "system": item["system"],
                      "prompt": item["prompt"], "completion": completion.strip(),
                      "oracle": item["oracle"], "prompt_sha256": _hash_text(item["prompt"]),
                      "teacher": args.model_id, "teacher_revision": args.revision}
            # A trace with no content teaches nothing and pollutes the loss; drop it loudly.
            if len(record["completion"]) < 10:
                print(f"[trace] SKIP empty completion for {item['key']}", flush=True)
                continue
            out.write(json.dumps(record) + "\n")
            kept += 1
        out.flush()
        done = min(start + args.batch_size, len(items))
        print(f"[trace] {done}/{len(items)} kept={kept} ({(time.time()-t0)/60:.1f} min)", flush=True)
    out.close()
    meta = {"corpus": args.corpus, "teacher": args.model_id, "revision": args.revision,
            "prompts": len(items), "kept": kept, "batch_size": args.batch_size, "temperature": args.temperature,
            "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
            "source": str(args.dataset) if args.corpus == "review" else args.benchmark,
            "split": args.split if args.corpus == "code" else None,
            "elapsed_minutes": round((time.time() - t0) / 60, 2)}
    (args.output.parent / (args.output.stem + ".meta.json")).write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[trace] wrote {kept}/{len(items)} traces -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
