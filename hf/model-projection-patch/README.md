---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
language:
  - en
pipeline_tag: text-generation
tags:
  - weight-space-programming
  - projection
  - sparse-patch
  - security-review
  - aya-expanse
  - research
datasets:
  - RishyanthReddy/aawden-authority-bench
---

# Aawden projection sparse patch for Aya Expanse 8B

**Status: research artifact. Single author. Small fixtures. Independent review pending.**

This repository holds a reversible output-projection patch for the pinned base model `CohereLabs/aya-expanse-8b` at revision `5062468bf9bc0c6035fd64e06274333ec127d980`. The edit projects a contrastive direction out of the residual-output projections (`self_attn.o_proj.weight` and `mlp.down_proj.weight`). It is stored as a sparse patch, so only the changed slices and their original values are present.

**The base model weights are not included.** This is a derivative of Aya Expanse 8B. Obtain the base model under its own terms: CC-BY-NC-4.0 and the [Cohere Labs Acceptable Use Policy](https://huggingface.co/CohereLabs/aya-expanse-8b). The MIT license of the Aawden code does not relicense the base model or this derivative.

## Contents

- `projection-patch.safetensors`: changed slices and original slices.
- `projection-patch.json`: manifest with per-tensor hashes, changed-scalar counts and the update norm.

Files:

{{FILES_TABLE}}

## Measured results

Balanced accuracy against executable oracle labels, from the saved run summaries listed below.

{{RESULTS_TABLE}}

Summary files used for this card:

{{SUMMARY_FILES}}

## How to apply

Apply only with the verified loader in `src/security_wsp/patches.py`. It checks the patch hash and each target tensor hash against the base, and `revert_patch` restores the original slices.

## Evidence and limits

- Fixtures are project-original, single-author, executable Python cases. They are small and synthetic in style.
- Which split chose the projection direction and strength is recorded in the patch manifest. UNKNOWN - not checked until this patch exists and its manifest is read.
- Advice robustness, general coding preservation, human-reviewed explanations and independent label review are not measured here.
- Outputs need human verification before any security decision.

## Related

- Benchmark: `RishyanthReddy/aawden-authority-bench`. Run logs: `RishyanthReddy/aawden-run-logs`.
- Sibling artifacts: `RishyanthReddy/aawden-wsp-patch`, `RishyanthReddy/aawden-aya-lora-v7`.
