"""Exercise the real toolkit APIs on a tiny RANDOMLY INITIALIZED architecture."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from security_wsp.artifacts import atomic_json, digest, read_json, utc_now
from security_wsp.data import build_cases
from security_wsp.discovery import discover_directions
from security_wsp.evaluation import evaluate_method
from security_wsp.interventions import export_hf_projection
from security_wsp.models import HuggingFaceAdapter
from security_wsp.parity import load_instrumented, run_parity, prepare_unedited_reload, verify_unedited_reload
from security_wsp.prompts import render
from security_wsp.sponsors import audit_records, enable_vendor
from security_wsp.workflow import temporary_projection


def build_random_model(path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import CohereConfig, CohereForCausalLM, PreTrainedTokenizerFast
    torch.manual_seed(5)
    vocabulary = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    specials = ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>")
    vocabulary.update({token: len(vocabulary)+i for i, token in enumerate(specials)})
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(list(specials))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token=specials[0],
                                        bos_token=specials[1], eos_token=specials[2])
    tokenizer.chat_template = ("{{ bos_token }}{% for message in messages %}"
                               "{{ '<' + message['role'] + '>' + message['content'] + eos_token }}"
                               "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}")
    config = CohereConfig(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=2048,
                          pad_token_id=vocabulary[specials[0]], bos_token_id=vocabulary[specials[1]],
                          eos_token_id=vocabulary[specials[2]])
    model = CohereForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return {"model_id": str(path), "revision": "randomly-initialized-seed-5", "device": "cpu", "dtype": "float32",
            "context_limit": 2048, "max_new_tokens": 4, "seed": 5, "local_files_only": True, "trust_remote_code": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    enable_vendor()
    import torch
    torch.set_num_threads(2)
    config = build_random_model(output / "random-source")
    adapter = HuggingFaceAdapter(config)
    instrumented = None if args.sequential else load_instrumented(adapter)
    cases = build_cases()[:2]
    tolerances = read_json(ROOT / "configs/protocol.json")["parity"]
    parity = run_parity(adapter, instrumented, cases, tolerances)
    parity.update(evidence_kind="randomly_initialized_architecture_smoke", real_weights_executed=False,
                  stage="synthetic_architecture_parity")
    atomic_json(output / "parity-smoke.json", parity)
    if not parity["passed"]:
        raise ValueError("Synthetic conversion did not pass; see parity-smoke.json")
    print("Synthetic Cohere architecture: conversion parity passed", flush=True)
    if instrumented is None:
        instrumented = load_instrumented(adapter)
    findings = discover_directions(adapter, instrumented, [replace(c, split="discovery") for c in cases], max_layers=1)
    findings["evidence_kind"] = parity["evidence_kind"]
    atomic_json(output / "discovery-smoke.json", findings)
    candidate = findings["candidates"][0]
    print("Synthetic Cohere architecture: residual hooks and causal controls ran", flush=True)
    del instrumented
    unedited = output / "unedited-roundtrip"
    prepare_unedited_reload(adapter, cases, unedited)
    unedited_result = verify_unedited_reload(unedited)
    atomic_json(output / "unedited-roundtrip.json", unedited_result)
    if not unedited_result["passed"]:
        raise ValueError("Synthetic unedited roundtrip changed deterministic output")
    requests = [render(c, "neutral", strong=True, max_new_tokens=4, seed=5) for c in cases]
    with temporary_projection(adapter, candidate["tensor_name"], candidate["direction"], .5):
        expected = [(result.request_id, result.text) for result in adapter.generate(requests)]
    manifest = export_hf_projection(adapter, [candidate["tensor_name"]], candidate["direction"], .5,
                                    output / "random-edited", digest(findings))
    edited_config = dict(config, model_id=str(output / "random-edited"))
    edited = HuggingFaceAdapter(edited_config)
    atomic_json(output / "edited-model.json", edited_config)
    atomic_json(output / "reload-requests.json", [asdict(request) for request in requests])
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-m", "security_wsp.reload_probe", "--kind", "hf",
                    "--checkpoint", str(output / "edited-model.json"), "--requests", str(output / "reload-requests.json"),
                    "--output", str(output / "reload-outputs.json")], env=env, check=True, timeout=240)
    observed = [(result["request_id"], result["text"]) for result in read_json(output / "reload-outputs.json")]
    if expected != observed:
        raise ValueError("Synthetic checkpoint reload changed deterministic output")
    rows = evaluate_method(cases, edited, "D", max_new_tokens=4, seed=5)
    audit = audit_records(rows, output / "auditkit-smoke.json")
    result = {"status": "completed", "evidence_kind": parity["evidence_kind"], "created_at": utc_now(),
              "parity_passed": True, "parity_checks": len(parity["tests"]), "discovery_completed": True,
              "safetensors_verified": True, "changed_values": sum(row["changed_entries"] for row in manifest["tensors"]),
              "unedited_roundtrip_passed": True,
              "fresh_process_reload_passed": True, "reload_generations": len(observed), "auditkit": audit,
              "aya_expanse_weights_loaded": False, "scientific_success": False}
    atomic_json(output / "smoke-summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
