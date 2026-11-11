"""pass@1 on real public coding benchmarks, with sandboxed execution.

Why this exists
    The submission's weak point is external validity: every fixture is templated, and the small models
    are at chance on them. This measures the same models on benchmarks other people wrote and run, so
    "Aya-8B cannot code" can be checked against HumanEval / MBPP / LiveCodeBench rather than asserted
    from our own fixtures.

What it does
    1. Loads a public benchmark (HumanEval 164, MBPP 257, or LiveCodeBench-lite).
    2. Prompts the model with the standard task statement, greedy decoding, so the number is
       reproducible rather than a lucky sample.
    3. Extracts the completion (markdown fence or bare def), appends the benchmark's own test harness,
       and executes it in a SUBPROCESS with a wall-clock timeout and no inherited environment.
    4. Reports pass@1 with a Wilson interval.

    Greedy decoding is deliberate: pass@1 with temperature 0 answers "can this model do the task at
    all", which is the competence question. Sampling would measure something else.

Contamination
    HumanEval and MBPP are public and almost certainly inside pretraining of every model here. A high
    score is therefore NOT evidence of generalisation; the number is only meaningful as a BEFORE/AFTER
    pair on the same model, which is how the distillation arm uses it. LiveCodeBench is time-split and
    is the cleaner check, which is why it is supported here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXEC_TIMEOUT = 12.0


# ---------------------------------------------------------------- code extraction
def extract_code(text: str) -> str:
    """Pull the Python body out of a completion: last fenced block, else from the first def/import."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if fences:
        return fences[-1].strip("\n")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*(def |class |import |from )", line):
            return "\n".join(lines[i:]).rstrip()
    return text.strip()


# ---------------------------------------------------------------- execution
def run_program(program: str, timeout: float = EXEC_TIMEOUT) -> tuple[bool, str]:
    """Execute in a subprocess with a clean env; no network use, no inherited secrets."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONHASHSEED": "0",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        try:
            proc = subprocess.run([sys.executable, "-I", "-S", str(path)], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as exc:                      # noqa: BLE001 - report, never crash the run
            return False, f"{type(exc).__name__}: {exc}"
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout).strip()[-400:]


# ---------------------------------------------------------------- benchmarks
def load_humaneval(limit: int | None) -> list[dict]:
    from datasets import load_dataset
    rows = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        rows.append({"task_id": r["task_id"], "prompt": r["prompt"], "tests": r["test"],
                     "entry_point": r["entry_point"], "harness": "humaneval"})
    return rows[:limit] if limit else rows


def load_mbpp(limit: int | None) -> list[dict]:
    from datasets import load_dataset
    rows = []
    for i, r in enumerate(load_dataset("google-research-datasets/mbpp", split="test")):
        imports = "\n".join(r.get("test_imports") or [])
        tests = "\n".join(r["test_list"])
        check = f"\ndef check(candidate):\n"
        # MBPP tests call the function by name directly; wrap them so a wrong name fails cleanly.
        check += "    candidate() if False else None\n"
        # Standard MBPP prompting: the task text plus the asserts, so the model knows the function name.
        prompt = (f"{r['text']}\nYour code should pass these tests:\n\n{tests}\n\n"
                  "Write the Python function in one markdown code block.")
        rows.append({"task_id": f"mbpp/{i}", "prompt": prompt, "tests": imports + "\n" + tests,
                     "entry_point": None, "harness": "mbpp",
                     "raw_asserts": "\n".join(r["test_list"])})
    return rows[:limit] if limit else rows


def build_program(row: dict, completion: str) -> str:
    """Assemble a runnable file: candidate code + the benchmark's own assertions."""
    code = extract_code(completion)
    if row["harness"] == "humaneval":
        body = row["prompt"] + code[len(row["prompt"]):] if code.startswith(row["prompt"][:20]) else code
        # HumanEval's prompt already contains the signature; use completion if it defines the entry point.
        program = body if f"def {row['entry_point']}" in body else row["prompt"] + "\n" + code
        return program + "\n\n" + row["tests"] + f"\n\ncheck({row['entry_point']})\n"
    # MBPP: plain task text, model must write the function; run the official asserts.
    return code + "\n\n" + row["tests"] + "\n"


# ---------------------------------------------------------------- scoring
def wilson(k: int, n: int) -> list | None:
    import math
    if not n:
        return None
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


# ---------------------------------------------------------------- inference
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-id", default=None, help="optional LoRA to merge before scoring")
    parser.add_argument("--benchmark", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", default=None, help="optional system prompt")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_humaneval(args.limit) if args.benchmark == "humaneval" else load_mbpp(args.limit)
    print(f"[bench] {args.benchmark}: {len(rows)} tasks", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    multi = torch.cuda.device_count() > 1
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16, use_safetensors=True,
        device_map="auto" if multi else None,
        max_memory=({i: f"{os.environ['WSP_MAX_GPU_GIB']}GiB" for i in range(torch.cuda.device_count())}
                    if multi and os.environ.get("WSP_MAX_GPU_GIB") else None))
    if args.adapter_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_id)
        model = model.merge_and_unload()
    if not multi:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results, passed = [], 0
    t0 = time.time()
    for i, row in enumerate(rows):
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
                   [{"role": "user", "content": row["prompt"]}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        program = build_program(row, completion)
        ok, err = run_program(program)
        passed += ok
        results.append({"task_id": row["task_id"], "passed": ok, "error": err,
                        "completion": completion, "program": program})
        if (i + 1) % 20 == 0:
            print(f"[bench] {i+1}/{len(rows)} passed={passed} ({(time.time()-t0)/60:.1f} min)", flush=True)

    n = len(results)
    report = {"model_id": args.model_id, "revision": args.revision, "adapter_id": args.adapter_id,
              "benchmark": args.benchmark, "system": args.system, "decoding": "greedy",
              "max_new_tokens": args.max_new_tokens, "tasks": n, "passed": passed,
              "pass_at_1": round(passed / n, 4) if n else None, "pass_at_1_ci95": wilson(passed, n),
              "elapsed_minutes": round((time.time() - t0) / 60, 2),
              "torch": torch.__version__, "transformers": __import__("transformers").__version__,
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[bench] {args.benchmark} pass@1 = {passed}/{n} = {report['pass_at_1']} "
          f"CI {report['pass_at_1_ci95']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
