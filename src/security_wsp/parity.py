"""Real-weight reference / CircuitKIT equivalence gate. Never passes on a skip."""
from __future__ import annotations
import gc
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from .artifacts import atomic_json, digest, file_hash, read_json, utc_now
from .prompts import render
from .sponsors import enable_vendor


def load_instrumented(adapter):
    enable_vendor()
    import circuitkit as ck  # Must precede the first transformer_lens import.
    config = adapter.config
    model = ck.load_model(config["model_id"], dtype=config["dtype"], device=adapter.device,
                          hf_model=adapter.model, tokenizer=adapter.tokenizer, revision=config["revision"],
                          local_files_only=config.get("local_files_only", False),
                          fold_ln=False, center_writing_weights=False, center_unembed=False, fold_value_biases=False)
    # Aawden patches whole attention and MLP outputs. Per-head projections create
    # a large broadcast tensor and are only needed for head-level interventions.
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    model.set_ungroup_grouped_query_attention(False)
    return model.eval()


def offload_reference(adapter):
    adapter.model.to("cpu")
    if adapter.device.startswith("cuda"):
        adapter.torch.cuda.empty_cache()


def parity_queries(adapter, cases):
    import torch
    examples = [cases[i] for i in sorted({0, len(cases)//2, len(cases)-1})]
    for case in examples:
        for condition in ("neutral", "incorrect_suggestion"):
            prefix = adapter.chat_prefix(render(case, condition).messages)
            tokens = adapter._tokens(prefix).cpu()
            for batch_size in (1, 2):
                yield tokens.repeat(batch_size, 1), None, {"case_id": case.case_id, "condition": condition,
                                                          "batch_size": batch_size, "input_tokens": tokens.shape[1]}
            for continuation in (' {"verdict":"vulnerable"}', ' {"verdict":"not_vulnerable_for_target"}'):
                full = adapter._tokens(prefix+continuation).cpu()
                n = tokens.shape[1]
                if full.shape[1] <= n or not torch.equal(full[:, :n], tokens):
                    raise ValueError("Parity continuation changes token boundary")
                yield full, n, {"case_id": case.case_id, "continuation": continuation, "tokens": full.shape[1]-n}


def parity_score(logits, ids, start):
    if start is None:
        return logits[:, -1].float().log_softmax(-1)
    # Keep every continuation position and leave the prefix logits out of the reduction.
    logprobs = logits[:, start-1:-1].float().log_softmax(-1)
    return logprobs.gather(-1, ids[:, start:].unsqueeze(-1)).sum()


def run_parity(adapter, instrumented, cases, tolerances):
    import torch
    queries = list(parity_queries(adapter, cases))
    references = []
    with torch.inference_mode():
        for ids, start, _ in queries:
            device_ids = ids.to(adapter.device)
            logits = adapter.model(input_ids=device_ids).logits
            references.append(parity_score(logits, device_ids, start).cpu())
    del logits
    sequential = instrumented is None
    if sequential:
        offload_reference(adapter)
        instrumented = load_instrumented(adapter)
    results = []
    with torch.inference_mode():
        for (ids, start, record), reference in zip(queries, references, strict=True):
            device_ids = ids.to(adapter.device)
            observed = parity_score(instrumented(device_ids), device_ids, start)
            if start is None:
                reference = reference.to(observed.device)
                maxdiff = (reference-observed).abs().max().item()
                kl = (reference.exp() * (reference-observed)).sum(-1).max().item()
                argmax = bool(torch.equal(reference.argmax(-1), observed.argmax(-1)))
                passed = (maxdiff <= tolerances["max_logprob_difference"] and kl <= tolerances["max_kl"]
                          and (argmax or not tolerances["require_argmax_match"]))
                results.append(dict(record, max_logprob_difference=maxdiff, kl=max(0, kl), argmax_match=argmax, passed=passed))
            else:
                a, b = reference.item(), observed.item()
                results.append(dict(record, reference_logprob_sum=a, instrumented_logprob_sum=b,
                                    passed=abs(a-b) <= record["tokens"]*tolerances["max_logprob_difference"]))
    if sequential:
        del instrumented
        gc.collect()
        if adapter.device.startswith("cuda"):
            torch.cuda.empty_cache()
        adapter.model.to(adapter.device)
    return {"version": 1, "stage": "real_weight_parity", "status": "completed",
            "passed": all(row["passed"] for row in results), "checked_at": utc_now(), "tests": results,
            "model_config_sha256": digest(adapter.config), "model": adapter.metadata, "tolerances": tolerances,
            "instrumentation_loading": "sequential" if sequential else "concurrent",
            "scope": "Numerical conversion gate. Unedited export/reload and edited behavioral reload are separate required checks.",
            "real_weights_executed": True}


def prepare_unedited_reload(adapter, cases, directory):
    """Record deterministic predictions and serialize the unchanged HF reference."""
    examples = [cases[i] for i in sorted({0, len(cases)//2, len(cases)-1})]
    requests = [render(case, condition, max_new_tokens=adapter.config["max_new_tokens"], seed=adapter.config["seed"])
                for case in examples for condition in ("neutral", "incorrect_suggestion")]
    identity = digest({"config": adapter.config, "model": adapter.metadata, "requests": [asdict(r) for r in requests]})
    marker = directory / "reference.json"
    if directory.exists():
        if not marker.is_file() or read_json(marker)["identity_sha256"] != identity:
            raise ValueError("Existing reference export is partial or belongs to a different model; choose a new parity output")
        return
    predictions = [{"request_id": row.request_id, "text": row.text, "finish_reason": row.finish_reason}
                   for row in adapter.generate(requests)]
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".roundtrip-") as temporary:
        staging = Path(temporary) / "roundtrip"
        checkpoint = staging / "checkpoint"
        checkpoint.mkdir(parents=True)
        adapter.model.save_pretrained(checkpoint, safe_serialization=True)
        adapter.tokenizer.save_pretrained(checkpoint)
        config = dict(adapter.config, model_id=str((directory / "checkpoint").resolve()), local_files_only=True)
        atomic_json(staging / "config.json", config)
        atomic_json(staging / "requests.json", [asdict(r) for r in requests])
        files = {str(p.relative_to(staging)): file_hash(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
        files.update({name: file_hash(staging / name) for name in ("config.json", "requests.json")})
        atomic_json(staging / marker.name, {"identity_sha256": identity, "predictions": predictions, "files": files})
        staging.rename(directory)


def verify_unedited_reload(directory):
    """After freeing the reference model, reload the unchanged export in a new process."""
    reference = read_json(directory / "reference.json")
    for name, expected in reference["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError("Unedited roundtrip export or input changed")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "-m", "security_wsp.reload_probe", "--kind", "hf",
                    "--checkpoint", str(directory / "config.json"), "--requests", str(directory / "requests.json"),
                    "--output", str(directory / "actual.json")], env=env, check=True, timeout=7200)
    predictions = [{k: row[k] for k in ("request_id", "text", "finish_reason")} for row in read_json(directory / "actual.json")]
    return {"passed": predictions == reference["predictions"], "generations": len(predictions),
            "fresh_process": True, "runtime_hooks": False, "reference_sha256": file_hash(directory / "reference.json"),
            "actual_sha256": file_hash(directory / "actual.json")}

