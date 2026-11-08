---
base_model: CohereLabs/aya-expanse-8b
language: en
tags:
  - security-review
  - lora
  - research
---

# Aawden Aya security-review adapter

Aawden is a research workbench for evidence-sensitive security review. This artifact is a supervised LoRA adapter for the pinned Aya Expanse 8B revision `5062468bf9bc0c6035fd64e06274333ec127d980`. It requires the original base model and the project's adapter loader. It is not a standalone dense checkpoint or a demonstrated causal circuit repair.

## Training and intended use

The selected v7 adapter resumed a previously trained adapter and used 150 exposed, project-authored Python security fixtures. Training included neutral and misleading reviewer opinions, three epochs, learning rate 0.0001, rank 16, alpha 32 and verdict-token loss weight 10. The update targets query, key, value, attention-output, gate, up and down projections. Original base tensors stayed frozen.

Use is limited to research on small Python SQLite and record-authorization examples under explicit threat models. Outputs require human verification before use in security decisions.

## Measured result

The candidate scored 85% on 20 separate validation cases and was frozen before a prospective 20-case check spanning ten families. It scored 90% strict neutral balanced accuracy: nine of ten vulnerable and nine of ten safe cases, with no invalid JSON. SQL scored 10/10; access control scored 8/10. The 95% family-bootstrap interval is 75–100%. The check is now exposed and must not be reused as an independent final benchmark.

All fixtures share one author. Adaptive experiment selection, small sample size and related synthetic code patterns limit generalization. Advice robustness, ordinary coding preservation, independently reviewed explanations, dense export/reload equivalence and independent final performance remain unmeasured. Aya-to-CircuitKIT numerical parity missed the prespecified tolerance.

## Files and verification

The local file is `checkpoints/aya-adapter-v7.safetensors`, 167,819,216 bytes, containing 448 FP32 tensors and 41,943,040 parameters. SHA-256: `e86f9475e3c8871ba3c60a63b10704954f66edcb791bd9631866a07d82a9022f`.

`configs/model-adapted.json` binds this adapter to exact base shard hashes. `configs/selected-review.json` selects the category trace prompt. Run `python aawden.py audit-adapter --output docs/AYA_VERIFICATION.json` to verify the saved generations and adapter bytes without GPU inference. See [the complete result](docs/BEST_RESULT.md) and [the attempt ledger](docs/AYA_PROGRESS.md).

## Licensing and publication

The [Cohere Labs model card](https://huggingface.co/CohereLabs/aya-expanse-8b) identifies the base model as CC-BY-NC-4.0 and requires adherence to Cohere Labs' Acceptable Use Policy and access conditions. The project's MIT code license does not replace those model terms. Adapter distribution must respect applicable base-model terms. The local research bundle excludes all model weights, credentials and private provider logs. Public model hosting, registration and hackathon submission require the owner's final review and authorization.
