---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
base_model_relation: merge
library_name: pytorch
language:
  - en
pipeline_tag: text-generation
tags:
  - weight-space-programming
  - sparse-patch
  - security-review
  - aya-expanse
  - research
datasets:
  - RishyanthReddy/aawden-authority-bench
---

# Aawden WSP sparse patch for Aya Expanse 8B

**Status: research artifact. Single author. Small fixtures. Four patches were built and measured;
all four are null on the headline, and the reason is a measured property of the base model rather than
of the patches: Aya Expanse 8B is at chance on this task with no reviewer present, so there is no
competent answer for an edit to uncover.** See "Measured results" below. The same sparse-edit idea,
repeated on Qwen2.5-14B — a model that does solve cases unaided — is also null against size-matched
random controls (see "The follow-up on a competent model").

This repository is meant to hold a sparse weight patch produced by Aawden's weight-space
programming (WSP) method: an LLM agent edits a small set of a model's weights directly, with no
gradients and no training loop, to stop it deferring to a wrong security reviewer.

## Licensing: this is a derivative, and no base weights ship here

The patch has no standalone model. It is a delta against Aya Expanse 8B, so it inherits that model's
terms and cannot be interpreted as a grant of anything more permissive.

- **License: `cc-by-nc-4.0`.** This is the base model's license, recorded in
  `configs/hf-publish.json` (`base_model.license`) and in the project's own
  [the model card](../../docs/MODEL_CARD_V3.md). It is **non-commercial**.
- **The base weights are not included and are not redistributed.** No tensor slices of the base
  model are stored. `configs/hf-publish.json` sets `base_model.weights_uploaded: false` and
  `forbid_base_slices: true` on this repo. The upload rules refuse weight files whose names match
  the base shard patterns (`^model-\d{5}-of-\d{5}\.safetensors$`, `^pytorch_model.*\.bin$`), so a
  full checkpoint cannot be published to this repo by accident.
- **You must obtain Aya yourself** from `CohereLabs/aya-expanse-8b` at the pinned revision
  `5062468bf9bc0c6035fd64e06274333ec127d980` (`configs/hf-publish.json`, `base_model.revision`;
  also `configs/protocol-v3.json`, `target_model.revision`), and accept the
  [Cohere Labs Acceptable Use Policy](https://huggingface.co/CohereLabs/aya-expanse-8b) on your own
  account. The patch is a derivative of that model and binds you to the same terms.
- **The project's MIT license covers only the Aawden code and fixtures.** See `LICENSE` at the
  repository root. It does not relicense the base model, and it does not relicense this derivative.

Because the license is non-commercial and the model is a derivative, do not ship this in a product,
a hosted service, or any commercial security tooling.

## What the file contains

Only the slices that changed. A patch is not a checkpoint and not a state dict: it holds no
untouched tensors, so it cannot be mistaken for a copy of the model.

The format is `aawden-patch-v1` (`src/security_wsp/patches.py`, constant `PATCH_FORMAT`). The layout
is fixed by `save_patch` and read back by `_load_verified`:

- **`wsp-patch.safetensors`** — for each patched tensor `NAME`, exactly three stored tensors, under
  these literal keys:
  - `NAME::values` — the edited slices, dtype and shape `(*, 1)` along the chosen axis;
  - `NAME::original` — the original slices at the same positions, kept so the patch can be reverted;
  - `NAME::indices` — `int64` positions of the changed slices along `NAME`'s axis, unique and
    strictly increasing.
  The file carries one metadata entry, `format = aawden-patch-v1`.
- **`wsp-patch.json`** — the manifest, written last so it only exists once the weights file is
  complete. It lists, per tensor: `name`, `shape`, `dtype`, `axis` (0 = rows, 1 = columns),
  `indices_count`, `changed_scalars`, `original_sha256` and `patched_sha256`; plus file-level
  `format`, `weights_file`, `weights_sha256`, `changed_scalars`, `total_parameters`,
  `percent_changed`, and an `extra` block.

Two design decisions worth knowing before you trust a number from this file:

- **Axis choice is a minimum, not a convention.** `build_patch` measures how many slices would
  change on axis 0 and on axis 1 and keeps whichever is smaller (`axis = min(candidates, key=...)`),
  with ties going to axis 0. So one tensor may be row-sparse and the next column-sparse.
- **Scalars are compared bitwise, not numerically.** `_differs` reinterprets the buffers as `uint8`
  and compares bytes, so `-0.0` vs `0.0` and NaN payloads count as changes. This makes the patch
  byte-exact and revertible, which is what the verification below depends on.

`build_patch` refuses to emit a patch that cannot reconstruct the edited tensor: after writing the
slices it rebuilds the tensor and compares hashes, raising `RuntimeError` if the sparse slices do
not reproduce the edit.

## How to apply

Use the project's loader. Do not hand-apply the slices. The entry point is `apply_patch` in
`src/security_wsp/patches.py`:

```python
from security_wsp.patches import apply_patch, revert_patch

manifest = apply_patch(model, "wsp-patch")   # in place; pass the path stem, not the filename
# ... run your evaluation ...
revert_patch(model, "wsp-patch")             # restores the original slices
```

Both functions take the **stem** (`path`), not `.safetensors`: `_paths()` appends `.safetensors` and
`.json` itself.

**Important — the edit target is not the bare base model.** Protocol v3 route (b) is in force: the
original Aya with the JSON prefill scored 0.400 neutral balanced accuracy on discovery, below the
0.75 threshold, so the edit base is Aya **with the retrained v7 LoRA adapter merged into dense
weights** (`configs/model-merged-v7.json`, `base_route: "b_merged_adapter"`;
`runs/wsp-v3/baseline-A2/decision.json`, `route: "b_merged_adapter"`). Load the merged base, then
apply the patch:

```python
import json
from security_wsp.models import HuggingFaceAdapter
from security_wsp.patches import apply_patch

with open("configs/model-merged-v7.json") as stream:   # HuggingFaceAdapter takes a parsed config dict
    config = json.load(stream)
adapter = HuggingFaceAdapter(config)   # loads Aya and folds v7 in (merge_lora: true)
apply_patch(adapter.model, "wsp-patch")   # stem only: it appends .safetensors and .json
```

A patch applied to the un-merged base will fail verification, which is the intended outcome — see
below.

## How to verify

`apply_patch` and `revert_patch` are the same routine (`_transfer`) run in opposite directions, and
all checking happens in two phases: **validate everything, then write anything.**

1. `format` in the manifest must equal `aawden-patch-v1`, else `ValueError`.
2. `sha256(wsp-patch.safetensors)` must equal the manifest's `weights_sha256`. A tampered or
   truncated weights file is rejected before a single tensor is read.
3. Every tensor name in the manifest must exist in the model exactly once; shape and dtype must
   match the manifest entry.
4. `axis` must be a valid dimension of the target, and the stored `indices` must be `int64`, 1-D,
   the manifest's `indices_count` long, unique and strictly increasing, and within range.
5. The stored `values` and `original` slices must have exactly the shape the axis implies and the
   same dtype as the target.
6. **The live tensor's hash must equal `original_sha256`** (applying) or `patched_sha256`
   (reverting). This is the wrong-base check: it refuses before writing.
7. A **probe**: the slices are written to a clone of the tensor and the clone's hash must equal the
   expected `patched_sha256` / `original_sha256`. If not, the operation is refused and nothing is
   written.
8. Only then are the real tensors written, and every tensor's hash is re-checked against the
   expected value. If **anything** fails mid-write — including an interrupt — `_restore` rebuilds all
   targets from pre-write snapshots and re-verifies each, so a half-applied patch cannot survive.

`apply_patch` returns the manifest dict, so you can check `changed_scalars` and `percent_changed`
against what you intended.

## Measured results

**Four sparse patches were built by two independent LLM programmers and scored on the 20-case
discovery subset. None moved the misleading score.** A maximum-strength ablation and its random
controls were also measured. Numbers below are from saved summaries; paths are repo-relative.

| arm | mechanism | scalars changed | misleading BA | neutral BA | correct-opinion BA |
|---|---|---:|---:|---:|---:|
| unedited baseline | — | 0 | 0.000 | 0.350 | 0.900 |
| astra-01 | 64 highest-norm columns, L18 head 25 | 262,144 (0.0033%) | 0.000 | 0.400 | 0.900 |
| astra-02 | full blocks L18 heads 8+30, x0.5 | 1,048,575 (0.0131%) | 0.000 | 0.450 | 0.900 |
| fable-01 | closed-form `not`-lens ablation, 4 heads | 2,027,865 (0.0253%) | 0.000 | 0.400 | 0.950 |
| astra-03 | two-column contrast channel, 7 heads | 3,670,016 (0.0457%) | 0.100 | 0.400 | 0.900 |
| max-strength | 7 complete heads zeroed | 3,670,016 (0.0457%) | 0.100 | 0.400 | 0.900 |

**Size- and norm-matched random controls scored 0.000 on all three seeds** (17, 29, 43), with update
norms matched to 1.2e-6 relative. So the attention heads matter; but note what the 0.100 is.

### Why the 0.100 is not a repair

On a balanced 10-vulnerable / 10-clean set, a model that *always* answers `vulnerable` scores exactly
0.500. The ablation reached 0.100 with **2 true positives and 0 true negatives** — it never correctly
cleared a safe file. Combined with the stronger prompt it reached 0.500 with 10 true positives and
**0 true negatives**, i.e. it became the constant predictor and lost the correct-opinion condition
(0.900 -> 0.500). That arm fails non-inferiority by 0.333 against a 0.02 tolerance.

### The measured reason

Holding the reviewer's opinion text fixed and varying only the code, the code moves the forced-choice
margin by **under 1 nat** (paired 95% intervals over 30 same-family pairs, three answer formats, six
tests). Changing only the opinion moves it **+10.7 nats** or **−19.7 nats**. With no opinion, forced
choice is at chance (0.517 balanced accuracy). Under a stated opinion the unedited model's generated
verdict copies it in **168 of 168** parseable answers. Because the model is at chance unaided, that
copying is guessing plus a hint, not deference over knowledge it has — which is why no edit could
uncover a correct answer.

The ablation's own effect follows the same pattern: it shifts the answer **+5.10 nats for clean code
and +4.97 for vulnerable code under the same opinion** (differential −0.13 ± 0.68). It cancels part of
the opinion and never touches the code. And that shift is **specific to the JSON prefill**: with
letter-coded answers the same edit leaves the model following the opinion in 240 of 240 comparisons,
and the verdict component of the shift is −0.5 nats, the remainder being a bias toward whichever answer
is labelled "A".

### The criterion that fails for every arm

A policy that reads the opinion and ignores the code cannot exceed
**J = BA(misleading) + BA(correct opinion) − 1 = 0**. Raw J is ≤ 0 in all eleven arm-by-split rows
measured. Counting parseable answers only, J is exactly 0.000 in five of six recomputable arms; the
exception is the `trace` prompt at +0.06 on both splits, which is one-sided toward the model's prior
and does not discriminate code (10/27 vs 7/29, p = 0.39).

**This repository therefore ships the patches as documented negative results**, not as a fix. The base
model's own figures, for context:

| Condition, arm `A_pf` (JSON prefill) | Discovery (60) | Validation (40) |
|---|---:|---:|
| Neutral (no reviewer opinion) | 0.400 | 0.425 |
| Correct reviewer opinion | 0.883 | 0.700 |
| **Misleading reviewer opinion** | **0.000** | **0.000** |

Balanced accuracy, strict parsing, invalid answers counted incorrect
(`runs/wsp-v3/baseline-A2/summary.json`).

### The follow-up on a competent model

Deference is only measurable on cases a model solves unaided. Measured that way (flip rate = share of
solved cases a wrong reviewer flips, three answer formats; `runs/wsp-v3/deference-summary-redacted.json`),
Aya Expanse 32B solves 90 / 85 / 85 of 129 plain cases and flips 0.94 / 1.00 / 0.91 of them;
Qwen2.5-14B solves 111 / 111 / 88 and flips 0.76 / 0.76 / 0.52. On Qwen2.5-14B, zeroing the 11 heads
that attend most to the opinion (7,208,960 scalars, 0.049%; chosen on the puzzle set, evaluated on the
disjoint plain set) gave flip rates 0.71 / 0.79 / 0.59 against 0.75 / 0.75 / 0.49 unedited and
0.71–0.77 / 0.58–0.78 / 0.49–0.66 for three size-matched random-head controls, and cost 13 solved
cases on the JSON format (`runs/wsp-v3/ablate-qwen25-14b/ablation.json`). That edit is not published
here; it is a measurement, and it is null.

### Machine-generated results at staging time

{{RESULTS_TABLE}}

Patch files are **not** included in this repository. The manifests and the specs that build them live
in the project repo under `runs/wsp-v3/proposals/` and `runs/wsp-v3/round-*/`. `configs/hf-publish.json`
marks this repo `available: false` until those sources are staged.

## Intended use

Research use only, and specifically: reproducing or extending a measurement of whether a
non-gradient weight edit can reduce a language model's deference to a wrong security reviewer,
on small Python SQLite and record-authorization cases under explicit threat models.

That is the whole intended scope. This is not a security tool, not a hardened classifier, and not a
drop-in replacement for the base model in any pipeline.

## Limitations

- **Measured, and null.** Six patches were produced and scored; none moved the misleading score above
  the size-matched random controls in a way that survived inspection. The 0.100 reached by the
  largest ablation is 2 true positives and 0 true negatives — not a repair, and its per-case
  margin shift is identical for both code classes.
- **The fixtures are single-author.** All cases share one author, who also ran the measurements.
  Independent label review is pending; a blinded packet exists at
  `dist/wsp-final-label-review/packet.json` (sha256 `47f15b1c…`, matching
  `configs/wsp-data.json`, `label_review_packet.packet_sha256`) but no reviewer has filled in the form.
- **Small samples.** Discovery is 60 cases and validation 40 (`configs/wsp-data.json`,
  `files.wsp_discovery.cases`, `files.wsp_validation.cases`). Intervals will be wide.
- **Not an independent benchmark.** Prompts and edits are selected on validation only
  (`configs/protocol-v3.json`, `selection.on: "validation only"`). Anything reported from validation
  is selection-contaminated by construction.
- **The sealed set is untouched.** `wsp_final` (120 cases, 60 families) is sealed
  (`configs/wsp-data.json`, `sealed`). It is opened once, after freeze, by
  `scripts/wsp_final.py`, and it is not in the public dataset unless the owner releases it
  afterwards.
- **Scope.** Only two vulnerability classes (`sql_injection`, `access_control`) in a narrow code
  style. Nothing here transfers to other languages, other classes, or real codebases by default.
- **Revert is only as good as the slices.** Reverting restores exactly the saved slices and verifies
  their hashes. It does not restore anything else you changed in the meantime.
- **Outputs need human verification** before any security decision. The base model, unedited, scored
  0.000 balanced accuracy under a misleading reviewer; a patch that improves that has not been
  shown to make the model safe to trust.

## Related

- Benchmark, oracles and per-answer results: `RishyanthReddy/aawden-authority-bench`.
- Run logs, manifests and decisions: `RishyanthReddy/aawden-run-logs`.
- Sibling artifacts, all requiring the same base model: `RishyanthReddy/aawden-projection-patch`,
  `RishyanthReddy/aawden-aya-lora-v7`.
- The registered base for this patch's edit route is the merged v7 base
  (`configs/model-merged-v7.json`), which requires the retrained adapter
  (`retrain-in/v7.safetensors`, sha256 `2e0fb9c0dcbd4a07441212cb5d3a8e9e98b0ba9e4b016db887adfd9797ec7689`).
- Protocol and frozen rules: `docs/PROTOCOL_V3.md`, `configs/protocol-v3.json`.
