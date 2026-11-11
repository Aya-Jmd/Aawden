---
title: Aawden saved-output demo
emoji: "\U0001F6E1"
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
short_description: "Saved Aya outputs: base vs prompt-only vs patched, with oracle verdicts"
---

# Aawden saved-output demo

Research artifact from the Aawden project. **Single author, small fixtures, independent review pending.**

The demo shows one security-review case at a time. Each case is the same Python snippet with a
**misleading reviewer opinion** attached. Three answers are shown from saved real runs:

- **Base Aya:** Aya Expanse 8B with a JSON prefill and the standard prompt.
- **Prompt only:** the same base model with a stronger prompt, no weight change.
- **Patched weights:** an Aawden sparse weight patch applied to the base model.

Next to every answer the page shows the **executable oracle verdict**, obtained by running paired
vulnerable and fixed implementations, and a **which weights changed** summary read from the patch
manifest.

## What this Space is not

- It does not run inference. It serves saved outputs from `examples.json`.
- It does not ship model weights. The base model is `CohereLabs/aya-expanse-8b`, distributed under
  CC-BY-NC-4.0 and the Cohere Labs Acceptable Use Policy, and is not included.
- It makes no generalization claim. Cases are small synthetic fixtures written by one author.

## Results pending

If `examples.json` is absent, the page says results are pending and shows nothing. Output numbers are
never estimated here. The file is built offline with
`python scripts/hf_publish.py build-space-examples --run <wsp_measure output> --patch-manifest <patch.json> --out hf/space/examples.json`,
which excludes the sealed `wsp_final` split unless `--include-final` is passed after the freeze.

## Links

- Benchmark: `RishyanthReddy/aawden-authority-bench`
- Run logs: `RishyanthReddy/aawden-run-logs`
- Patches and adapter: `RishyanthReddy/aawden-wsp-patch`, `RishyanthReddy/aawden-projection-patch`, `RishyanthReddy/aawden-aya-lora-v7`
