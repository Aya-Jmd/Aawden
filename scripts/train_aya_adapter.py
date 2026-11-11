"""Bounded LoRA competence study with one frozen fresh-case evaluation."""
from __future__ import annotations
import argparse
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from safetensors.torch import load_file, save_file
from security_wsp.artifacts import RunStore, atomic_json, digest, file_hash, read_json, source_fingerprint
from security_wsp.data import CLEAN, load_dataset, run_oracle, validate_cases
from security_wsp.evaluation import evaluate_requests
from security_wsp.models import HuggingFaceAdapter
from security_wsp.prompts import render
from security_wsp.sponsors import audit_records
from security_wsp.statistics import metrics
from security_wsp.adaptation import attach
from security_wsp.confirmation_data import confirmation_cases
from security_wsp.confirmation_v2 import confirmation_cases_v2


def target(case):
    safe = case.oracle_verdict == CLEAN
    if case.target_category == "sql_injection":
        reason = "The input remains a literal through binding or SQLite quote escaping." if safe else "Untrusted input can change the SQL expression at execution."
    else:
        reason = "The requested record is returned only when the stated policy permits access." if safe else "A request can return another actor's record without the required policy check."
    return json.dumps({"verdict": case.oracle_verdict, "category": case.target_category,
                       "evidence_lines": list(range(2, len(case.code.splitlines()) + 1)),
                       "explanation": reason, "confidence": .9}, separators=(",", ":"))


def training_request(case, condition, trace=False):
    request = render(case, condition, max_new_tokens=192, numbered_source=True)
    if trace:
        from security_wsp.evidence_prompts import TRACE
        messages = [dict(message) for message in request.messages]
        messages[0]["content"] += " " + TRACE[case.target_category]
        request = replace(request, messages=messages)
    return request


def evaluate(adapter, cases, output, identity, trace=False):
    store = RunStore(output, dict(identity, dataset=digest([asdict(case) for case in cases])),
                     stage="adapter_competence", evidence_kind="cohere_local_checkpoint")
    store.start()
    try:
        store.bind_model("adapted", adapter.metadata)
        rows = evaluate_requests(((case, training_request(case, "neutral", trace))
                                  for case in cases), adapter, "adapted", store=store)
        summary = {"metrics": metrics(rows), "dataset": validate_cases(cases),
                   "threshold_exceeded": metrics(rows)["balanced_accuracy"] > .75,
                   "model": adapter.metadata, "scope": "Trained LoRA adapter. Single-author feasibility cases."}
        summary["AuditKIT"] = audit_records(rows, output / "auditkit.json")
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "results.json", rows)
        store.finish([row["storage_id"] for row in rows], ["summary.json", "results.json", "auditkit.json"])
        print(output.name, summary["metrics"]["balanced_accuracy"], flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        store.fail(error, interrupted=isinstance(error, KeyboardInterrupt))
        raise


def main(output, resume=None, refine=False):
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT / "configs/model.json")
    protocol = {"rank": 16, "alpha": 32, "epochs": 6, "lr": .0002, "accumulation": 2, "verdict_weight": 50,
                "seed": 42, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "conditions": ["neutral", "incorrect_suggestion"],
                "confirmation_sha256": file_hash(ROOT / "datasets/neutral-confirmation-v3.json"),
                "study_version": "adapter-v6-focused-verdict", "resume_sha256": file_hash(resume) if resume else None}
    protocol["training_script_sha256"] = file_hash(Path(__file__))
    protocol["training_files"] = {name: file_hash(ROOT / "datasets" / name) for name in
                                  ("screen.json", "neutral-confirmation.json", "neutral-confirmation-v2.json")}
    if refine:
        if not resume or file_hash(resume) != read_json(ROOT / "configs/adapter-review.json")["adapter"]["sha256"]:
            raise ValueError("Refinement requires the frozen v6 adapter")
        protocol.update(epochs=3, lr=.0001, verdict_weight=10, study_version="adapter-v7-trace-refinement",
                        confirmation_sha256=file_hash(ROOT / "datasets/neutral-confirmation-v5.json"),
                        validation_sha256=file_hash(ROOT / "datasets/neutral-confirmation-v4.json"),
                        added_training_sha256=file_hash(ROOT / "datasets/neutral-confirmation-v3.json"),
                        validation_used_for_training=False, trace_prompt=True)
    atomic_json(output / "protocol.json", protocol)
    cases = load_dataset(ROOT / "datasets/screen.json") + list(confirmation_cases()) + list(confirmation_cases_v2())
    if refine:
        cases += load_dataset(ROOT / "datasets/neutral-confirmation-v3.json")
    if not all(run_oracle(case)["passed"] for case in cases):
        raise ValueError("Training case executable oracle failed")
    adapter = HuggingFaceAdapter(config)
    if refine and adapter.metadata["checkpoint_hashes"] != read_json(ROOT / "configs/adapter-review.json")["adapter"]["base_checkpoint_hashes"]:
        raise ValueError("Refinement base weights changed")
    torch.manual_seed(protocol["seed"])
    params = attach(adapter.model, protocol["rank"], protocol["target_modules"])
    if resume:
        tensors = load_file(resume, device=adapter.device)
        if set(tensors) != set(params) or any(tensors[name].shape != value.shape for name, value in params.items()):
            raise ValueError("Resume checkpoint differs from the configured LoRA architecture")
        with torch.no_grad():
            for name, value in params.items():
                value.copy_(tensors[name])
        del tensors
        print("Resumed adapter", protocol["resume_sha256"], flush=True)
    model = adapter.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(list(params.values()), lr=protocol["lr"], weight_decay=0)
    examples = []
    for case in cases:
        for condition in protocol["conditions"]:
            request = training_request(case, condition, trace=refine)
            prefix = adapter.chat_prefix(request.messages)
            inputs = adapter._tokens(prefix + target(case) + adapter.tokenizer.eos_token).cpu()
            start = adapter._tokens(prefix).shape[1]
            labels = inputs.clone()
            labels[:, :start] = -100
            verdict_end = adapter._tokens(prefix + '{"verdict":' + json.dumps(case.oracle_verdict)).shape[1]
            opposite = "vulnerable" if case.oracle_verdict == CLEAN else CLEAN
            alternative = adapter._tokens(prefix + target(replace(case, oracle_verdict=opposite))).cpu()
            shared = min(inputs.shape[1], alternative.shape[1])
            verdict_start = (inputs[0, :shared] != alternative[0, :shared]).nonzero()[0].item()
            examples.append((inputs, labels, verdict_start, verdict_end))
    print("Training", len(examples), "examples and", sum(p.numel() for p in params.values()), "parameters", flush=True)
    rng = random.Random(protocol["seed"])
    history = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(protocol["epochs"]):
        rng.shuffle(examples)
        losses = []
        for inputs, labels, verdict_start, verdict_end in examples:
            logits = model(input_ids=inputs.to(adapter.device), use_cache=False).logits
            losses_by_token = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                                                               labels[:, 1:].to(adapter.device), ignore_index=-100,
                                                               reduction="none")
            weights = (labels[:, 1:] != -100).to(adapter.device, torch.float32)
            weights[:, verdict_start - 1:verdict_end - 1] *= protocol["verdict_weight"]
            loss = (losses_by_token * weights).sum() / weights.sum()
            (loss / protocol["accumulation"]).backward()
            losses.append(loss.item())
            step += 1
            if step % protocol["accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(list(params.values()), 1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 25 == 0:
                print("Training step", step, "mean recent loss", sum(losses[-25:]) / len(losses[-25:]), flush=True)
        history.append({"epoch": epoch + 1, "mean_loss": sum(losses) / len(losses)})
        save_file({name: value.detach().cpu().contiguous() for name, value in params.items()},
                  output / f"adapter-epoch-{epoch + 1}.safetensors")
        atomic_json(output / "training-progress.json", {"history": history, "steps": step})
    del optimizer, loss, logits, losses_by_token, weights
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    for parameter in params.values():
        parameter.requires_grad_(False)
    checkpoint = output / "adapter.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in params.items()}, checkpoint)
    atomic_json(output / "training.json", {"history": history, "steps": step, "trainable_parameters": sum(p.numel() for p in params.values())})
    saved = load_file(checkpoint, device=adapter.device)
    with torch.no_grad():
        for name, value in params.items():
            value.copy_(saved[name])
    del saved
    adapter.metadata = dict(adapter.metadata, adaptation="LoRA", adapter_sha256=file_hash(checkpoint), training_protocol=protocol)
    identity = {"config": config, "protocol": protocol, "source": source_fingerprint(), "model": adapter.metadata}
    validation_cases = load_dataset(ROOT / "datasets/neutral-confirmation-v4.json") if refine else cases
    if refine:
        if file_hash(ROOT / "datasets/neutral-confirmation-v4.json") != protocol["validation_sha256"]:
            raise ValueError("Validation data changed")
        if {case.family_id for case in cases} & {case.family_id for case in validation_cases}:
            raise ValueError("Training and validation families overlap")
    development = evaluate(adapter, validation_cases, output / "development", identity, trace=refine)
    outcome = {"development_balanced_accuracy": development["metrics"]["balanced_accuracy"], "threshold_exceeded": False}
    if development["threshold_exceeded"]:
        freeze = {"adapter_sha256": file_hash(checkpoint), "protocol_sha256": file_hash(output / "protocol.json"),
                  "source": source_fingerprint(), "confirmation_sha256": protocol["confirmation_sha256"]}
        atomic_json(output / "frozen.json", freeze)
        fresh_path = ROOT / ("datasets/neutral-confirmation-v5.json" if refine else "datasets/neutral-confirmation-v3.json")
        if file_hash(fresh_path) != freeze["confirmation_sha256"]:
            raise ValueError("Fresh confirmation file differs from the frozen commitment")
        fresh = load_dataset(fresh_path)
        if not all(run_oracle(case)["passed"] for case in fresh):
            raise ValueError("Fresh confirmation executable oracle failed")
        confirmation = evaluate(adapter, fresh, output / "confirmation", identity, trace=refine)
        outcome.update(confirmation_balanced_accuracy=confirmation["metrics"]["balanced_accuracy"],
                       threshold_exceeded=confirmation["threshold_exceeded"])
    atomic_json(output / "outcome.json", outcome)
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--refine", action="store_true", help="Bounded v7 refinement with untrained v4 validation and v5 confirmation")
    args = parser.parse_args()
    main(args.output, args.resume_checkpoint, args.refine)
