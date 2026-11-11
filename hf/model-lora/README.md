---
license: cc-by-nc-4.0
library_name: peft
base_model: CohereLabs/aya-expanse-8b
language:
  - en
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - security-review
  - aya-expanse
  - research
datasets:
  - RishyanthReddy/aawden-authority-bench
---

# Aawden Aya security-review LoRA (v7)

**Status: research artifact. Single author. Small fixtures. Independent review pending. The 20-case check below is now exposed and is not an independent benchmark.**

This is a rank-16 LoRA adapter for the pinned base model `CohereLabs/aya-expanse-8b` at revision `5062468bf9bc0c6035fd64e06274333ec127d980`. It is the retrained v7 adapter from the Aawden project. It needs the original base model and the project's adapter loader. It is not a standalone model and it is not a demonstrated causal circuit repair.

**The base model weights are not included.** The adapter is a derivative of Aya Expanse 8B. Obtain the base model under its own terms: CC-BY-NC-4.0 and the [Cohere Labs Acceptable Use Policy](https://huggingface.co/CohereLabs/aya-expanse-8b). The MIT license of the Aawden code does not relicense the base model or this adapter.

## Training

- Rank 16, alpha 32, learning rate 0.0001, three epochs, verdict-token loss weight 10.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- 41,943,040 trainable parameters. Original base tensors stayed frozen.
- Training used 150 exposed, project-authored Python security fixtures with neutral and misleading reviewer opinions. The v7 run resumed a previously trained adapter.
- The exact fixture file-to-training mapping is recorded in the run logs. UNKNOWN - not checked for this card beyond the run directory.

## Files

- `aya-adapter-v7.safetensors`: adapter weights. 167,819,216 bytes, 448 FP32 tensors.
- `aawden-loader-config.json`: loader configuration that binds the adapter to the base shard hashes.

{{FILES_TABLE}}

Expected SHA-256 of `aya-adapter-v7.safetensors`: `e86f9475e3c8871ba3c60a63b10704954f66edcb791bd9631866a07d82a9022f`.

## Measured results

Balanced accuracy against executable oracle labels, from the saved run summaries listed below. The development and confirmation rows are separate splits.

{{RESULTS_TABLE}}

Summary files used for this card:

{{SUMMARY_FILES}}

Reported in the project: 90% strict neutral balanced accuracy on a 20-case, ten-family check that was frozen before it was run. Nine of ten vulnerable and nine of ten safe cases were correct, with no invalid JSON. The family-bootstrap 95% interval is 75 to 100%. That check is now exposed and must not be reused as an independent final benchmark.

## Intended use

Research on small Python SQLite and record-authorization examples under explicit threat models. Outputs require human verification before any security decision.

## Evidence and limits

- All fixtures share one author. Adaptive experiment selection, small sample size and related synthetic code patterns limit generalization.
- Not measured: advice robustness, ordinary coding preservation, independently reviewed explanations, dense export and reload equivalence, and independent final performance.
- Aya-to-CircuitKIT numerical parity missed its prespecified tolerance.

## Related

- Benchmark: `RishyanthReddy/aawden-authority-bench`. Run logs: `RishyanthReddy/aawden-run-logs`.
- Sibling artifacts: `RishyanthReddy/aawden-wsp-patch`, `RishyanthReddy/aawden-projection-patch`.
