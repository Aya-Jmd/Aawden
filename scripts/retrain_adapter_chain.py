#!/usr/bin/env python3
"""Retrain the Aawden Aya LoRA lineage v5 -> v6 -> v7 from configs/retrain-v7.json.

The v5, v6 and v7 adapter files were lost, so this rebuilds them from the recorded recipes.
The target v7 depends on v6, which depends on v5. Planning and every refusal check use only
the standard library; torch is imported only when a stage actually trains.

scripts/train_aya_adapter.py cannot be called for this: its non-refine path hard-codes the v6
values, and its refine path refuses any resume file whose hash is not the original v6 hash.
This script therefore reuses that module's prompt, target and request helpers and runs the same
training loop, step for step, with per-stage settings from the config.

Examples (GPU host; the v7 reproduction snapshot must be extracted to SNAPSHOT first):
  python scripts/retrain_adapter_chain.py --dry-run --source-root SNAPSHOT
  python scripts/retrain_adapter_chain.py --stage v5 --output checkpoints/retrained --source-root SNAPSHOT
  python scripts/retrain_adapter_chain.py --stage v6 --output checkpoints/retrained --source-root SNAPSHOT
  python scripts/retrain_adapter_chain.py --stage v7 --output checkpoints/retrained --source-root SNAPSHOT
  python scripts/retrain_adapter_chain.py --shortcut --output checkpoints/retrained --source-root SNAPSHOT
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "retrain-v7.json"
SOURCE_SUFFIXES = {".py", ".html", ".css", ".js"}
LINEAGE = ("v5", "v6", "v7")
DEFAULT_OUTPUT = "checkpoints/retrained"


class Refused(Exception):
    """A check failed before any GPU work started."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(source_root: Path) -> tuple[str, int]:
    """Same algorithm as security_wsp.artifacts.source_fingerprint, computed without importing it."""
    root = source_root / "src" / "security_wsp"
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    return hashlib.sha256(canonical(files)).hexdigest(), len(files)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_facts(source_root: Path, config: dict) -> dict:
    facts = {}
    for name, spec in sorted(config["datasets"].items()):
        path = source_root / "datasets" / name
        if not path.is_file():
            facts[name] = {"present": False, "matches": False, "count": None, "sha256": None}
            continue
        digest = sha256_file(path)
        count = len(json.loads(path.read_text(encoding="utf-8"))["cases"])
        facts[name] = {"present": True, "sha256": digest, "count": count,
                       "matches": digest == spec["sha256"] and count == spec["count"]}
    return facts


def stage_specs(config: dict, shortcut: bool) -> list[dict]:
    return [config["shortcut"]["stage"]] if shortcut else list(config["stages"])


def estimate(config: dict, spec: dict, facts: dict) -> dict:
    gpu = config["gpu_estimate"]
    examples = len(config["conditions"]) * sum(facts[name]["count"] or 0 for name in spec["training_sets"])
    steps = examples * spec["epochs"]
    evaluates = spec.get("validation") is not None
    eval_cases = 0
    if evaluates:
        eval_cases = (facts[spec["validation"]]["count"] or 0) + (facts[spec["confirmation"]]["count"] or 0)
    seconds = gpu["setup_seconds"] + steps * gpu["step_seconds"] + gpu["post_seconds"] + eval_cases * gpu["eval_seconds_per_case"]
    return {"examples_per_epoch": examples, "steps": steps, "evaluates": evaluates,
            "estimated_job_seconds": round(seconds)}


def build_plan(config: dict, source_root: Path, *, shortcut: bool, selected: list[str], output: str) -> dict:
    facts = dataset_facts(source_root, config)
    fingerprint, files = source_fingerprint(source_root)
    expected = config["source"]["fingerprint"]
    cap = config["gpu_estimate"]["timeout_cap_seconds"]
    problems = []
    if fingerprint != expected:
        problems.append("source")
    for name, fact in sorted(facts.items()):
        if not fact["matches"]:
            problems.append(f"dataset:{name}")
    stages = []
    for spec in stage_specs(config, shortcut):
        if not shortcut and spec["name"] not in selected:
            continue
        info = estimate(config, spec, facts)
        if info["estimated_job_seconds"] > cap:
            problems.append(f"job_over_cap:{spec['name']}")
        stages.append({
            "name": spec["name"],
            "resume_from": spec["resume_from"],
            "training_sets": spec["training_sets"],
            "epochs": spec["epochs"],
            "learning_rate": spec["learning_rate"],
            "verdict_weight": spec["verdict_weight"],
            "trace_prompt": spec["trace_prompt"],
            "validation": spec.get("validation"),
            "confirmation": spec.get("confirmation"),
            "steps": info["steps"],
            "examples_per_epoch": info["examples_per_epoch"],
            "evaluates": info["evaluates"],
            "estimated_job_seconds": info["estimated_job_seconds"],
            "recorded_adapter_sha256": (spec.get("recorded") or {}).get("adapter_sha256"),
            "outputs": {"adapter": f"{spec['name']}.safetensors", "record": f"{spec['name']}.json"},
        })
    total = sum(s["estimated_job_seconds"] for s in stages)
    if len(stages) > 1 and total > cap:
        problems.append("all_stages_in_one_process_over_cap")
    return {
        "target": config["target"],
        "mode": "shortcut" if shortcut else "lineage",
        "deviation": shortcut,
        "output": output,
        "gpu": config["gpu_estimate"]["gpu"],
        "source_fingerprint": {"expected": expected, "actual": fingerprint, "files": files,
                               "matches": fingerprint == expected},
        "datasets": facts,
        "stages": stages,
        "estimated_job_seconds_total": total,
        "problems": problems,
    }


def stage_outputs(out: Path, name: str) -> tuple[Path, Path, Path]:
    return out / f"{name}.safetensors", out / f"{name}.json", out / f"{name}-eval"


def resume_source(stage: dict, first_selected: str, args, out: Path, produced: dict) -> Path | None:
    """Where a stage reads its starting adapter from. None means the base model (fresh LoRA)."""
    if not stage["resume_from"]:
        return None
    if stage["resume_from"] in produced:
        return produced[stage["resume_from"]]
    if stage["name"] == first_selected and args.resume_from:
        return Path(args.resume_from)
    return out / f"{stage['resume_from']}.safetensors"


def check_run(plan: dict, args, out: Path) -> None:
    """Refuse before any GPU work: plan problems, overwrites and resume files that disagree with their record."""
    accepted = set()
    if args.allow_source_mismatch:
        accepted.add("source")
    if args.allow_long_run:
        accepted.add("all_stages_in_one_process_over_cap")
    refusals = [f"plan problem: {p}" for p in plan["problems"] if p not in accepted]
    produced: dict[str, Path] = {}
    first = plan["stages"][0]["name"] if plan["stages"] else None
    for stage in plan["stages"]:
        checkpoint, record, evaluation = stage_outputs(out, stage["name"])
        for path in (checkpoint, record, evaluation):
            if path.exists():
                refusals.append(f"exists, refusing to overwrite: {path}")
        source = resume_source(stage, first, args, out, produced)
        if source is not None and stage["resume_from"] not in produced:
            if not source.is_file():
                refusals.append(f"resume file missing for {stage['name']}: {source}")
            elif source.with_suffix(".json").is_file():
                recorded = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))["adapter_sha256"]
                if sha256_file(source) != recorded:
                    refusals.append(f"resume file differs from its own record: {source}")
        produced[stage["name"]] = checkpoint
    if refusals:
        raise Refused("; ".join(refusals))


def load_training_module(source_root: Path):
    path = source_root / "scripts" / "train_aya_adapter.py"
    spec = importlib.util.spec_from_file_location("train_aya_adapter_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_examples(adapter, tm, cases, conditions, trace, clean):
    """Same example construction as scripts/train_aya_adapter.py."""
    examples = []
    for case in cases:
        for condition in conditions:
            request = tm.training_request(case, condition, trace=trace)
            prefix = adapter.chat_prefix(request.messages)
            inputs = adapter._tokens(prefix + tm.target(case) + adapter.tokenizer.eos_token).cpu()
            start = adapter._tokens(prefix).shape[1]
            labels = inputs.clone()
            labels[:, :start] = -100
            verdict_end = adapter._tokens(prefix + '{"verdict":' + json.dumps(case.oracle_verdict)).shape[1]
            opposite = "vulnerable" if case.oracle_verdict == clean else clean
            alternative = adapter._tokens(prefix + tm.target(replace(case, oracle_verdict=opposite))).cpu()
            shared = min(inputs.shape[1], alternative.shape[1])
            verdict_start = (inputs[0, :shared] != alternative[0, :shared]).nonzero()[0].item()
            examples.append((inputs, labels, verdict_start, verdict_end))
    return examples


def run_chain(plan: dict, config: dict, args, source_root: Path, out: Path) -> int:
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from safetensors.torch import load_file, save_file
    import security_wsp
    if not Path(security_wsp.__file__).resolve().is_relative_to((source_root / "src").resolve()):
        raise Refused("security_wsp was imported from outside the --source-root snapshot")
    from security_wsp.adaptation import attach
    from security_wsp.artifacts import file_hash, source_fingerprint as package_fingerprint
    from security_wsp.confirmation_data import confirmation_cases
    from security_wsp.confirmation_v2 import confirmation_cases_v2
    from security_wsp.data import CLEAN, load_dataset, run_oracle
    from security_wsp.models import HuggingFaceAdapter

    tm = load_training_module(source_root)
    sets = {name: load_dataset(source_root / "datasets" / name) for name in config["datasets"]}
    if [asdict(c) for c in confirmation_cases()] != [asdict(c) for c in sets["neutral-confirmation.json"]]:
        raise Refused("confirmation_cases() differs from datasets/neutral-confirmation.json")
    if [asdict(c) for c in confirmation_cases_v2()] != [asdict(c) for c in sets["neutral-confirmation-v2.json"]]:
        raise Refused("confirmation_cases_v2() differs from datasets/neutral-confirmation-v2.json")

    model_config = json.loads((source_root / config["model"]["config"]).read_text(encoding="utf-8"))
    adapter = HuggingFaceAdapter(model_config)
    if adapter.metadata["checkpoint_hashes"] != config["model"]["base_checkpoint_hashes"]:
        raise Refused("base checkpoint hashes differ from configs/retrain-v7.json")
    rank, targets = config["lora"]["rank"], list(config["lora"]["target_modules"])
    model = adapter.model
    chain_sha = sha256_file(Path(__file__).resolve())
    train_sha = sha256_file(source_root / "scripts" / "train_aya_adapter.py")
    out.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}
    first = plan["stages"][0]["name"]

    for stage in plan["stages"]:
        name = stage["name"]
        checkpoint, record_path, eval_dir = stage_outputs(out, name)
        source = resume_source(stage, first, args, out, produced)
        if source is not None and not source.is_file():
            raise Refused(f"resume file missing: {source}")
        resume_sha = sha256_file(source) if source else None
        train_cases = [case for set_name in stage["training_sets"] for case in sets[set_name]]
        if not all(run_oracle(case)["passed"] for case in train_cases):
            raise Refused("a training case failed its executable oracle")
        if stage["validation"]:
            if {c.family_id for c in train_cases} & {c.family_id for c in sets[stage["validation"]]}:
                raise Refused("training and validation families overlap")

        torch.manual_seed(config["seed"])
        params = attach(model, rank, targets)
        if source is not None:
            tensors = load_file(source, device=adapter.device)
            if set(tensors) != set(params) or any(tensors[n].shape != v.shape for n, v in params.items()):
                raise Refused("resume tensors do not match the configured LoRA architecture")
            with torch.no_grad():
                for n, v in params.items():
                    v.copy_(tensors[n])
            del tensors
            print(f"[{name}] resumed from {source.name} sha256={resume_sha}", flush=True)

        examples = build_examples(adapter, tm, train_cases, config["conditions"], stage["trace_prompt"], CLEAN)
        print(f"[{name}] {len(examples)} examples/epoch, {stage['steps']} steps planned", flush=True)
        optimizer = torch.optim.AdamW(list(params.values()), lr=stage["learning_rate"], weight_decay=0)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        rng = random.Random(config["seed"])
        accumulation = config["optimizer"]["accumulation"]
        clip = config["optimizer"]["grad_clip_norm"]
        history, step = [], 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(stage["epochs"]):
            rng.shuffle(examples)
            losses = []
            for inputs, labels, verdict_start, verdict_end in examples:
                logits = model(input_ids=inputs.to(adapter.device), use_cache=False).logits
                per_token = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().transpose(1, 2), labels[:, 1:].to(adapter.device),
                    ignore_index=-100, reduction="none")
                weights = (labels[:, 1:] != -100).to(adapter.device, torch.float32)
                weights[:, verdict_start - 1:verdict_end - 1] *= stage["verdict_weight"]
                loss = (per_token * weights).sum() / weights.sum()
                (loss / accumulation).backward()
                losses.append(loss.item())
                step += 1
                if step % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(list(params.values()), clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 25 == 0:
                    print(f"[{name}] step {step} mean recent loss {sum(losses[-25:]) / len(losses[-25:]):.6f}", flush=True)
            history.append({"epoch": epoch + 1, "mean_loss": sum(losses) / len(losses)})
            print(f"[{name}] epoch {epoch + 1} done, mean loss {history[-1]['mean_loss']:.6f}", flush=True)
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        for parameter in params.values():
            parameter.requires_grad_(False)

        tmp = out / f".{name}.tmp.safetensors"
        save_file({n: v.detach().cpu().contiguous() for n, v in params.items()}, tmp)
        os.link(tmp, checkpoint)  # raises FileExistsError if the checkpoint already exists
        tmp.unlink()
        adapter_sha = sha256_file(checkpoint)
        produced[name] = checkpoint
        trainable = sum(p.numel() for p in params.values())

        protocol = {"rank": rank, "alpha": config["lora"]["alpha"], "epochs": stage["epochs"],
                    "lr": stage["learning_rate"], "accumulation": accumulation,
                    "verdict_weight": stage["verdict_weight"], "seed": config["seed"],
                    "target_modules": targets, "conditions": config["conditions"],
                    "trace_prompt": stage["trace_prompt"], "resume_sha256": resume_sha,
                    "training_sets": {n: {"sha256": sha256_file(source_root / "datasets" / n)}
                                      for n in stage["training_sets"]}}
        adapter.metadata = dict(adapter.metadata, adaptation="LoRA", adapter_sha256=adapter_sha,
                                training_protocol=protocol)
        evaluation = None
        if stage["evaluates"]:
            evaluation = evaluate_stage(tm, adapter, stage, sets, eval_dir, protocol, package_fingerprint)
        record = {
            "target": config["target"], "stage": name, "deviation": plan["deviation"],
            "adapter_file": checkpoint.name, "adapter_sha256": adapter_sha,
            "adapter_bytes": checkpoint.stat().st_size, "trainable_parameters": trainable,
            "resume_from": str(source) if source else None, "resume_sha256": resume_sha,
            "protocol": protocol, "steps": step, "examples_per_epoch": len(examples),
            "history": history, "source_fingerprint": plan["source_fingerprint"]["actual"],
            "chain_script_sha256": chain_sha, "train_script_sha256": train_sha,
            "reference": {"recorded_adapter_sha256": stage["recorded_adapter_sha256"],
                          "expected_to_match": False,
                          "note": "Retraining is not byte-identical by design."},
            "evaluation": evaluation,
        }
        with record_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[{name}] wrote {checkpoint} sha256={adapter_sha}", flush=True)
    return 0


def evaluate_stage(tm, adapter, stage, sets, eval_dir, protocol, package_fingerprint):
    """Only v7 has a recorded validation gate. The 75% threshold and fresh-set rule are preserved."""
    identity = {"config": adapter.config, "protocol": protocol, "source": package_fingerprint(),
                "model": adapter.metadata}
    development = tm.evaluate(adapter, sets[stage["validation"]], eval_dir / "development", identity,
                              trace=stage["trace_prompt"])
    outcome = {"development_set": stage["validation"],
               "development_balanced_accuracy": development["metrics"]["balanced_accuracy"],
               "threshold_exceeded": False}
    if development["threshold_exceeded"]:
        confirmation = tm.evaluate(adapter, sets[stage["confirmation"]], eval_dir / "confirmation", identity,
                                   trace=stage["trace_prompt"])
        outcome.update(confirmation_set=stage["confirmation"],
                       confirmation_balanced_accuracy=confirmation["metrics"]["balanced_accuracy"],
                       threshold_exceeded=confirmation["threshold_exceeded"])
    (eval_dir / "outcome.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outcome


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="v7", choices=["v7"],
                        help="only the v7 lineage is implemented; v8-v10 use rigorous_training and are not covered")
    parser.add_argument("--stage", choices=[*LINEAGE, "all"], default="all",
                        help="one lineage stage per Beam job; 'all' runs v5, v6, v7 in one process")
    parser.add_argument("--shortcut", action="store_true",
                        help="DEVIATION: train the v7 recipe from the base model in one stage, skipping v5 and v6")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="directory for <stage>.safetensors and <stage>.json (default %(default)s)")
    parser.add_argument("--resume-from", default=None,
                        help="adapter .safetensors for the first selected stage when it is not in --output")
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="extracted v7 reproduction snapshot (must match the recorded source fingerprint)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-source-mismatch", action="store_true", help="run even if the source fingerprint differs")
    parser.add_argument("--allow-long-run", action="store_true",
                        help="run all selected stages in one process even if the total exceeds the job cap")
    parser.add_argument("--dry-run", action="store_true", help="print the plan as JSON; no torch import, no writes")
    args = parser.parse_args(argv)
    if args.shortcut and args.stage != "all":
        parser.error("--shortcut runs its own single stage; do not pass --stage")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    source_root = args.source_root.resolve()
    shortcut = args.shortcut
    selected = [] if shortcut else (list(LINEAGE) if args.stage == "all" else [args.stage])
    plan = build_plan(config, source_root, shortcut=shortcut, selected=selected, output=args.output)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    out = Path(args.output)
    try:
        check_run(plan, args, out)
        return run_chain(plan, config, args, source_root, out)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
