# Phase 5: freeze and final evaluation (`scripts/wsp_final.py`)

Two subcommands: `freeze`, then `evaluate`. Nothing in this document is a measurement. Every number the
write-up cites comes from `runs/wsp-v3/final/report.md` and `runs/wsp-v3/final/summary.json`, both
generated from saved artifacts. Status of every command in this file: **not run** on real data; the
script is exercised offline by `tests/test_wsp_final.py` on a constructed plumbing double.

The executable values are in `configs/protocol-v3.json`; the reasoning is in `docs/PROTOCOL_V3.md`.
This document covers only the phase-5 contract.

## The exact command the evaluator runs

```bash
# 1. Freeze. No model is loaded. The sealed dataset is hashed, never parsed.
.venv-aawden/bin/python scripts/wsp_final.py freeze \
  --selection runs/wsp-v3/selection/selection.json \
  --review /evaluator/wsp-final-review.json \
  --regression datasets/wsp-coding.json \
  --regression datasets/wsp-general.json \
  --regression datasets/wsp-format.json \
  --output runs/wsp-v3/final

# 2. Verify the freeze and the repeated-look rule without running anything.
.venv-aawden/bin/python scripts/wsp_final.py check-freeze \
  --freeze-manifest runs/wsp-v3/final/freeze.json
.venv-aawden/bin/python scripts/wsp_final.py evaluate \
  --freeze-manifest runs/wsp-v3/final/freeze.json --check

# 3. The single final look.
.venv-aawden/bin/python scripts/wsp_final.py evaluate \
  --freeze-manifest runs/wsp-v3/final/freeze.json \
  --output runs/wsp-v3/final
```

Exit codes: `0` success, `2` refusal. A refusal prints `Refused: <reason>` to stderr and, before step
3's model work, writes nothing. `--allow-toy-core` is required twice for a `constructed_toy` run
(freeze and evaluate); it can never be used to dress a toy run as evidence.

## What `freeze` binds

`freeze` writes `runs/wsp-v3/final/freeze.json` and nothing else. It refuses if the file already
exists, if `evaluation.json`, `final_look.json`, `report.md` or `summary.json` is already present, and
if any bound file is missing.

| field | content |
| --- | --- |
| `manifest_sha256` | sha256 of the canonical manifest body **excluding this field**. It is written last and verified **first** on every load, so no field of the manifest (including `output_dir`) can be edited after freeze without refusal |
| `source_sha256` | `artifacts.source_fingerprint()` over `src/security_wsp` |
| `protocol_sha256` | sha256 of `configs/protocol-v3.json` |
| `prompt_identity` | `PROMPT_VERSION`, the JSON prefill, the conditions, the strong variants, the selected stronger prompt, every wording, and then the text that actually determines every prompt: `components` (sha256 of the standard and each strong system prompt, the prefill, the closing instruction, the source fences) and one measured suggestion string per `(condition, wording, verdict)` in `opinions` (appended after the source) and `source_comments` (written inside it, which is how `code_comment` carries its verdict) |
| `final_dataset` | path, sha256, committed case and family counts, and the sha256 of the commitments file |
| `label_review` | path, sha256, case count, reviewer id, and `review_form`: the committed form's sha256 and case count, or an explicit `unbound_reason` when the data commitments name no form |
| `regressions` | each file's sha256 and kind, plus `_commitments_sha256` |
| `selection` | the phase-4 selection's own sha256, method, patch binding, candidates, random controls, steering, LoRA spec |
| `patch` / `method` | the selected patch manifest and weights hashes, the changed-scalar count **re-derived from the patch file**, and the protocol fraction check applied to that derived count, or `method: "none"` |
| `adapt` | route b: the frozen adapter spec and its sha256, the merge strategy, and `merged_state_sha256: null` until the first evaluation records it |
| `final_look_rule` | the rule in prose, stored in the artifact, not only in code |
| `sealed_data_access` | states that the sealed file was hashed, not parsed, before the freeze |

A prompt hash cannot be computed before the sealed set opens, because every real prompt embeds its own
case's code and threat model. What *is* fixed at freeze time is all the text `render()` contributes
around a case, and that is what the freeze binds. `evaluate` re-checks every rendered request against
those components before it reaches a model, so drift in a system prompt, the prefill, an opinion
sentence, a source comment or the closing line, or a wording the freeze did not name, is refused.

Refusals: a second freeze; a `datasets/wsp-final.json` whose sha256 differs from
`configs/wsp-data.json`; a commitments file that does not mark `wsp_final` sealed; a review bound to
a different dataset sha256, missing coverage, a missing reviewer id or a false attestation; a
completed review whose case order does not follow the committed review form, or a form whose bytes
differ from the committed sha256; a regression set whose sha256 or case count differs from
`configs/wsp-regressions.json`; a selection with `status: "no_candidate"` (protocol v3: nothing is
frozen and that outcome is reported as-is); a patch whose **measured** size is above
`max_changed_fraction_of_weights`, or whose manifest understates the size the file actually contains;
an unknown candidate kind; a missing exposure note; a steering block that this script cannot apply (a
site other than the residual stream, a non-integer or negative layer, a zero or absent strength, a
zero or non-finite direction), or a steering direction bound with no steering method selected.


The review itself is a new file, so its own sha256 is bound by the freeze rather than committed in
advance; only the form it must follow is pre-committed.

## What `evaluate` does, in order

0. **Verify the manifest against itself.** `manifest_sha256` is recomputed over the canonical body
   and compared first, and `output_dir` must equal the directory the manifest actually sits in. A
   manifest that was edited, or copied into another directory, refuses before any other check reads
   a field from it.
1. Re-hash **everything** the freeze names: protocol, source fingerprint, sealed dataset (against
   both the freeze and the data commitments file), review, all three regression sets and their
   commitments file, the selection, the patch manifest and weights, every candidate and random-control
   patch, the steering direction and the LoRA spec. The **patch statistics are re-derived from the
   patch file** (`derive_patch_statistics`) and compared with both the manifest and the freeze record,
   and `max_changed_fraction_of_weights` is enforced again on the derived count. Any mismatch refuses
   before a model is loaded and before the sealed labels are parsed. A frozen steering selection is
   checked for applicability here.
2. Refuse a repeated final look (below).
3. **Consume the directory**: write `final_look.json` before any sealed byte is opened, then write
   `evaluation.json` with `status: "running"`, so a crash is visible and the directory is closed.
4. Only now: load the sealed cases, and run `final_review.verify_independent_review` over them. Its
   `ValueError` is caught and re-raised as a protocol refusal, so the CLI prints one line and exits 2
   instead of a traceback.
5. Load the frozen base. On route b it is the frozen LoRA **merged dense**, with the merged-state hash
   recorded; every edited arm is a copy of that merged state (`_reload_model`), never a fresh config
   load, so A/B and C/D share one base. Then run every arm below, checkpointing each answer through
   `RunStore`. A steering arm gets the frozen direction applied by a forward hook at the frozen layer
   and strength.

| arm | method id | prompt |
| --- | --- | --- |
| A | `A` | base, standard |
| B | `B` | base, the stronger prompt selected on validation |
| C | `C` | the frozen edit, standard |
| D | `D` | the frozen edit, stronger |
| each other comparator | `<kind>:C`, `<kind>:D` | standard and stronger |
| size-matched random patches | `random-17`, `random-29`, `random-43` | stronger |
| no-op | `noop` | stronger |

Conditions are `neutral`, `correct_suggestion`, `incorrect_suggestion` with wording `reviewer`, plus a
separate transfer block of the held-out wordings `authority`, `ciso`, `scanner`, `consensus`,
`code_comment`, each against `B` and `D` on the two opinion conditions (neutral ignores wording).
Every prompt uses the JSON prefill from `protocol["output_format"]["json_prefill"]`, applied
identically to every arm. Nothing is truncated; a prompt over the context budget raises.

The no-op arm applies no patch, and `summary.json` checks it against A on the primary condition
(`noop_check`, protocol v3 `controls.noop_must_equal_baseline`).

An arm whose patch was not frozen is **not** a refusal: it is recorded in `failures` and printed as
MISSING, so a missing comparator is visible in the report rather than silent.

Regressions run before and after the selected method over the three committed sets: coding through
`regressions.evaluate_coding` and `coding_summary` unchanged (arm names `original`/`edited`), general
scored by exact answer letter, format scored with the builder's own `verify()` loaded from
`scripts/build_wsp_regressions.py` so the verifiers cannot drift.

## What counts as a repeated final look

One output directory holds at most one final look, and "one look" means one **attempt**. `evaluate`
refuses when:

- `final_look.json` exists — a look was started here, in any status. That file is written before the
  sealed dataset is opened, so a process that dies during the first model call still leaves a
  directory that refuses a retry;
- `evaluation.json` exists — a look was attempted, whatever `status` it recorded (`running`,
  `failed`, `interrupted` and `completed` all refuse);
- `report.md` or `summary.json` exists without an `evaluation.json` — an unlogged look;
- `--output` differs from the freeze's own `output_dir`, or that recorded `output_dir` is not the
  directory the manifest sits in — a second look at the sealed set is a repeated final look however
  it is spelled, including a fresh directory;
- `freeze.json` already exists, so the comparison cannot be re-scoped after seeing results.

`--check` verifies the same directory a later `evaluate` would use, because both read the directory
`verify_freeze` re-derived from the manifest's location; `--output` may confirm it but never replace
it.

A crashed or failed look is inspected by hand. Recovering from one means a new freeze into a new
directory, which is a documented change of comparison, not a retry of the same one.

## Report and summary

`report.md` is generated from the summary dict only, written atomically (temp file, `fsync`,
`os.replace`), with one table per endpoint: the primary `D - B` on `incorrect_suggestion` with its
paired family-bootstrap 95% CI (10,000 replicates, resampled by `family_id`, seed 42), `C - A`,
non-inferiority on neutral and correct at the −0.02 margin, per-arm rates (balanced accuracy,
accuracy, false positive and negative rates, coverage, invalid answers, abstentions), flip rates,
every transfer wording, regression deltas, changed-weight counts and percent, forward-pass usage, and
every control including failures.

Invalid answers and abstentions are **two disjoint measurements**. An answer either parses — possibly
to the explicit abstention verdict — or it fails to parse; the parse is all-or-nothing, so no answer
is counted in both. The report's columns are labelled `invalid (unparsed)` and
`abstentions (parsed refusal)` for that reason, and `n = decisive + abstentions + invalid`.

A measurement that does not exist prints as `MISSING`. It is never omitted, never blank and never
zero. `tests/test_wsp_final.py` checks this by feeding the generator a deliberately incomplete
summary: a declared arm with no measurement keeps its row with eight MISSING cells, a dropped
transfer wording disappears with its heading, and numeric tokens in the report are checked against
the summary.

`summary.json` carries the same numbers plus the hashes of the freeze, protocol, source, sealed
dataset, review, regression sets, every patch, the selection and the run store, which is what the HF
cards fill in.

## Differences from the v2 helpers this reuses

`workflow.freeze_candidate` and `final_review.verify_independent_review` assume the v2 schema: a
checkpoint directory, `edit_manifest_path`, `checkpoint_hashes`, a `trials` ledger, a
`selection["edited_config"]`, and cases whose split is `final_test`. Protocol v3 changes all of that,
so phase 5 implements the equivalent and keeps the v2 *semantics*:

| v2 assumption | v3 reality | phase-5 handling |
| --- | --- | --- |
| edit is a whole checkpoint directory | edit is a sparse patch pair (`*.safetensors` + `*.json`) | binds and verifies `weights_sha256` and `manifest_sha256`; **re-derives** the changed-scalar count from the patch file at freeze and again at evaluate, and refuses a patch above the changed-fraction limit measured that way |
| `selection["edit_manifest_sha256"]` + reload check | phase-4 selection is a different, external artifact | binds the selection file's own sha256 and the strong-prompt choice; refuses an unknown schema version |
| coding dataset only | three regression sets, with `kind` per file | binds all three against `configs/wsp-regressions.json` |
| cases split `final_test` | cases split `wsp_final` | `verify_independent_review` is called with the explicit final case list, so its `final_test` split check is bypassed deliberately and the split is checked here instead; its `ValueError` is converted to a refusal |
| `PROMPT_VERSION` only | prefill, wordings and the selected strong variant | binds the sha256 of each fixed prompt component plus the measured suggestion text per `(condition, wording, verdict)`, and re-verifies every rendered request before generation |
| refreeze and repeat-look rejected in code | same | the rule is also written into `freeze.json` as `final_look_rule`, the manifest self-hashes, and the directory is consumed on disk before the sealed data is read |

`verify_independent_review` is called unchanged; only the case list passed to it comes from
`wsp_final` rather than `final_test`.

## Evidence kinds

`evidence_kind` is `real_model` (`cohere_local_checkpoint`) or `constructed_toy`. A toy freeze needs
`--allow-toy-core` at both steps, and its report opens with a line stating that it is a constructed
plumbing run and not evidence about Cohere. `tests/test_wsp_final.py` runs only the toy path, on a
fake final dataset written to `tmp_path` and labelled `FAKE TEST DATA - NOT THE SEALED WSP FINAL SET`.

## Route b and steering, and how they are tested without weights

Two pieces of the frozen comparison cannot be exercised offline with real weights: the LoRA merge
that gives every arm one base, and the activation steering a steering comparator applies. Both are
implemented behind injectable seams and tested with fakes, so the plumbing is proven even though the
numbers are not.

- **Merged base.** `build_adapt(frozen, *, config, adapter_factory, load, merge)` loads the frozen
  adapter, calls `merge_dense`, and records `merged_state_sha256` — a hash over every parameter and
  buffer name, dtype, shape and bytes. `_reload_model` copies **that** state, and refuses if the base
  has drifted from the hash it reports. `tests/test_wsp_final.py` drives it with a two-tensor fake
  model and asserts `merge` was called exactly once and that the recorded hash is the post-merge
  state, not the pre-merge one.
- **Steering.** `verify_steering_applicable` refuses at bind *and* at evaluate time any selection that
  cannot be applied: a site other than the residual stream, a non-integer or negative layer, a zero or
  absent strength, a zero or non-finite direction. `apply_steering` registers a forward hook on
  `model.model.layers[layer]` that adds `strength * direction` to that layer's output. The test builds
  a two-layer fake module, checks that the unhooked layer is unchanged and the hooked one moves by
  exactly `strength * direction`, and checks that a steering method with no frozen block is an
  `ArmUnavailable` gap (reported as MISSING) rather than a silently unsteered arm.

A prompt hash for a real final case cannot be bound before the sealed set opens, so the freeze binds
the text `render()` contributes and `evaluate` verifies each rendered request against it. That is the
strongest binding available under protocol v3's ordering, and it is checked before generation in
`tests/test_wsp_final.py` for every method × condition × wording, prefill included.

## Not done here

No GPU job is launched by this script and none should be started from this document. The real final
look needs the phase-4 selection to exist, an independent review to be completed, and an approved
evaluation host; until then `evaluate` refuses at the freeze manifest.
