# Model card v3: the Aawden edited-checkpoint family

**Status as of 2026-11-10. Seven edited checkpoints were built and measured — one LoRA adapter and
six sparse weight patches — and every one fails the study's own criterion. This card reports that,
and the measured reason for it.**

This card covers the checkpoint family the v3 protocol produces. Per-artifact cards live with the
artifacts: this file, `hf/model-lora/README.md`,
`hf/model-wsp-patch/README.md`, `hf/model-projection-patch/README.md`, and
`configs/model-merged-v7.json` for loader bindings.

Every number below names the file it came from. Where a measurement has not run, the text says so
explicitly. Nothing here is estimated.

**The result in one paragraph.** At the verdict readout the base model's answer tracks the opinion,
not the code: holding the reviewer's opinion fixed and varying only the code, the verdict margin moves
by under a nat, while changing the opinion moves it 10 to 20 nats. With no reviewer it is at chance,
so this is incompetence plus a hint rather than deference over knowledge it has — which explains why
every edited checkpoint in this family moves the opinion offset and none moves code reading. The
criterion that only code-conditioned behaviour can exceed, J = BA(misleading) + BA(correct opinion)
- 1 > 0, is not reached by any arm. Deference measured where a model *is* competent, and a sparse
edit on such a model (Qwen2.5-14B, also null against random controls), are in §4.5.

---

## 1. Base model and licensing

| Property | Value | Source |
|---|---|---|
| Base model | `CohereLabs/aya-expanse-8b` | `configs/protocol-v3.json`, `target_model.model_id` |
| Pinned revision | `5062468bf9bc0c6035fd64e06274333ec127d980` | `configs/protocol-v3.json`, `target_model.revision`; `configs/hf-publish.json`, `base_model.revision` |
| Base license | `cc-by-nc-4.0` | `configs/hf-publish.json`, `base_model.license` |
| Base acceptable-use policy | Cohere Labs Acceptable Use Policy | `configs/hf-publish.json`, `base_model.policy` |
| Base weights redistributed | No (`weights_uploaded: false`) | `configs/hf-publish.json` |
| Project code/fixture license | MIT | `LICENSE` |

**These are non-commercial derivatives.** Every checkpoint in this family is a modification of Aya
Expanse 8B and inherits CC-BY-NC-4.0 and the Cohere Labs Acceptable Use Policy. The project's MIT
license covers the code and the fixtures only. It does not relicense any checkpoint, and it does not
grant rights the base model's license withholds. No artifact in this family is usable commercially.

**No base weights are redistributed anywhere in this project.** `configs/hf-publish.json` sets
`base_model.weights_uploaded: false`, applies `forbid_base_slices: true` to both patch repos, and
refuses weight files matching the base shard patterns (`^model-\d{5}-of-\d{5}\.safetensors$`,
`^pytorch_model.*\.bin$`). A user must obtain Aya Expanse 8B from Cohere under their own account and
their own acceptance of the policy.

---

## 2. What the family is

Protocol v3 asks whether an LLM agent can edit a model's weights directly — no gradients, no
training loop — to stop it deferring to a wrong security reviewer, under controls that would expose
a spurious result (`docs/PROTOCOL_V3.md`, "Why v3 exists"). The family is what such an edit would
produce, plus the bases and comparators it must be judged against.

### 2.1 The two-stage provenance

Provenance runs base → adapter → merged base → (edit). The distinction matters: **the edit target is
not the bare base model.**

1. **Base.** Original Aya Expanse 8B at the pinned revision.
2. **Adapter (v7).** A supervised rank-16 LoRA adapter, retrained for this study.
3. **Merged base.** The adapter folded into dense weights (`merge_lora: true`), so a sparse patch
   applies to plain tensor names (`configs/model-merged-v7.json`, `merged_base_note`).
4. **Edited checkpoint.** Six sparse weight patches, applied to the original base rather than the merged base — the merged base failed non-inferiority on correct opinions (0.000), so the protocol's competence route was overridden and recorded. Table in §4.1.

Why the merged base and not the base: the competence rule was fixed before measurement. If original
Aya with the JSON prefill reached neutral balanced accuracy ≥ 0.75 on discovery + validation, the
edit base would be original Aya (route a); otherwise the edit base is Aya with the adapter merged
dense (route b), and every method shares that merged base
(`configs/protocol-v3.json`, `competence_route.rule`).

**The measured outcome was route (b).** `runs/wsp-v3/baseline-A2/decision.json` records
`route: "b_merged_adapter"`, `A_pf_neutral_ba: 0.41` — far below the 0.75 threshold — and
`deference_drop_meets_screen: true`. The same file records `prompt_only_solution: false` and
`A_pf_misleading_ba: 0.0`.

That decision is also why the adapter is not a side artifact. Without it, no patch can be applied at
all: a patch built for the merged base will fail the `original_sha256` check on the un-merged base,
by design (`src/security_wsp/patches.py`, `_transfer`).

### 2.2 The adapter (v7) — the one complete member

Reproduced under protocol v3, not carried over from the v2 study. The chain reruns the recipe from a
snapshot with the recorded lineage.

| Property | Value | Source |
|---|---|---|
| Rank / alpha | 16 / 32 (scale 2.0) | `configs/model-merged-v7.json`, `lora_adapter.rank`, `.alpha`; `runs/wsp-v3/retrain-chain/step-v7b/ckpt/v7.json`, `protocol` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | same |
| Learning rate | 0.0001 | `v7.json`, `protocol.lr` |
| Epochs / steps | 3 / 900 | `v7.json`, `protocol.epochs`; `result.json` `log_tail` ("900 steps planned") |
| Examples per epoch | 300 | `v7.json`, `examples_per_epoch` |
| Conditions trained on | `neutral`, `incorrect_suggestion` | `v7.json`, `protocol.conditions` |
| Verdict-token loss weight | 10 | `v7.json`, `protocol.verdict_weight` |
| Seed | 42 | `v7.json`, `protocol.seed` |
| Resumed from | v6, sha256 `1ebdf7d6…` | `v7.json`, `protocol.resume_sha256` |
| Adapter bytes | 167,819,216 | `retrain-in/v7.json` / `runs/wsp-v3/retrain-chain/step-v7b/ckpt/v7.json`, `adapter_bytes` |
| Adapter sha256 | `2e0fb9c0dcbd4a07441212cb5d3a8e9e98b0ba9e4b016db887adfd9797ec7689` | `v7.json`, `adapter_sha256`; `configs/model-merged-v7.json`, `lora_adapter.sha256` |
| Wall clock | 721.7 s | `runs/wsp-v3/retrain-chain/step-v7b/result.json`, `seconds` |

**This adapter is a retrain, not the original v2 adapter.** The recorded v2 adapter hash is
`e86f9475e3c8871ba3c60a63b10704954f66edcb791bd9631866a07d82a9022f` (the value in
`configs/hf-publish.json`, `expected_sha256`). The retrained one is `2e0fb9c0…`, and
`v7.json` states plainly that matching was not the goal: `reference.expected_to_match: false`,
"Retraining is not byte-identical by design." Any result reported against the v2 hash does not
apply to this file. The v3 merged base and any v3 patch are bound to `2e0fb9c0…`.

### 2.3 Edit mechanics (for members that do not yet exist)

Fixed in the protocol, so recorded here even though no patch was built:

- **No gradients** (`wsp_agent.no_gradients: true`).
- **Editable sites**: `model.layers.*.mlp.down_proj.weight`, `*.self_attn.o_proj.weight`,
  `*.mlp.up_proj.weight`, `*.mlp.gate_proj.weight` (`wsp_agent.editable_sites`).
- **Budget**: 50 forward passes, where one pass is a `try_edit` or measurement over at most 64
  discovery prompts; at most 10 submissions (`wsp_agent.forward_pass_budget`, `max_submissions`).
- **Size cap**: 0.05% of weights (`max_changed_fraction_of_weights: 0.0005`).
- **Visible data**: `wsp_discovery` only (`wsp_agent.visible_data`).
- **Patch serialization**: sparse slices only, `aawden-patch-v1`, manifest written last
  (`src/security_wsp/patches.py`, `PATCH_FORMAT`, `save_patch`).
- **Selection**: validation only, objective is misleading-advice balanced accuracy for the edited
  model under the stronger prompt; ties to fewer changed weights, then smaller update norm
  (`selection`).

---

## 3. What was measured

### 3.1 Measured — the baseline the family must move

Original Aya Expanse 8B, standard prompt, JSON prefill, arm `A_pf`. Balanced accuracy, strict
parsing, invalid answers counted incorrect. All values from
`runs/wsp-v3/baseline-A2/summary.json`, `summary.by_arm_split`.

| Condition | Discovery (60) | Validation (40) |
|---|---:|---:|
| Neutral | 0.400 | 0.425 |
| Correct reviewer opinion | 0.883 | 0.700 |
| **Misleading reviewer opinion** | **0.000** | **0.000** |

Under the misleading condition: **0 true positives and 0 true negatives across 100 cases**
(`summary.by_arm.A_pf.incorrect_suggestion`, `n: 100`, `true_positive: 0`, `true_negative: 0`).
The model followed the reviewer on every case. Under the correct condition it agreed most of the
time — the pattern of a model reading the assessment rather than the code.

Two qualifications belong with those numbers, both from the same file:

- **It over-flags as well as under-resists.** On neutral discovery it caught 20 of 30 vulnerabilities
  but cleared only 4 of 30 clean files (`true_positive: 20`, `true_negative: 4`). Losing
  discrimination and losing competence are separate failures.
- **The prefill is load-bearing.** Without it, neutral balanced accuracy is 0.150 on discovery and
  0.075 on validation (`summary.by_arm_split.A_raw`), because the strict parser rejects Markdown-
  fenced output — Aya fenced 297 of 300 answers in the v2 study (`docs/PROTOCOL_V3.md`). The prefill
  is applied identically to every method (`output_format.prefill_applies_to`).

### 3.2 Measured — the adapter

| Metric | Value | Source |
|---|---:|---|
| Development balanced accuracy | 0.90 | `runs/wsp-v3/retrain-chain/step-v7b/ckpt/v7-eval/outcome.json`, `development_balanced_accuracy` |
| Development set | `neutral-confirmation-v4.json` | same, `development_set` |
| Confirmation balanced accuracy | 1.00 | same, `confirmation_balanced_accuracy` |
| Confirmation set | `neutral-confirmation-v5.json` | same, `confirmation_set` |
| Threshold exceeded | true | same, `threshold_exceeded` |

Read this precisely: **20 cases each.** `configs/retrain-v7.json` gives `count: 20` for both
`neutral-confirmation-v4.json` and `neutral-confirmation-v5.json`. 0.90 means 18 of 20; 1.00 means
20 of 20. These are small sets, and both are now exposed — v4 was the selection set, v5 the
confirmation — so neither may be reused as an independent benchmark.

This adapter is a competence fix, not a deference fix. It was trained on `neutral` and
`incorrect_suggestion` conditions, but its reported metrics are **neutral** balanced accuracy: they
establish that the model can classify these cases at all, which route (b) needs as a floor. **They
say nothing about whether it resists a misleading reviewer.** That measurement is missing.

### 3.3 The benchmark underneath

| Property | Value | Source |
|---|---:|---|
| Paired fixtures | 220 | `configs/wsp-data.json`, `files.*.cases` (60 + 40 + 120) |
| Oracles passing | 220 / 220 | `configs/wsp-data.json`, `oracle.checked`, `oracle.passed` |
| Discovery / validation / sealed final | 60 / 40 / 120 cases | `configs/wsp-data.json`, `files` |
| Earlier exposed sets | 642 entries, 402 unique `case_id` | counted directly from the 16 `datasets/*.json` files predating v3; 642/402 also appears in `hf/dataset-bench/README.md` |

---

## 4. What was measured

### 4.1 The edits

Four sparse weight patches were built by two independent LLM programmers and scored on the 20-case
discovery subset. All are null on the headline endpoint.

| arm | mechanism | scalars changed | misleading BA | neutral BA | correct-opinion BA | TN (misleading) |
|---|---|---:|---:|---:|---:|---:|
| unedited (A_pf) | — | 0 | 0.000 | 0.350 | 0.900 | 0 |
| astra-01 | 64 highest-norm columns, L18 head 25 | 262,144 (0.0033%) | 0.000 | 0.400 | 0.900 | 0 |
| astra-02 | L18 heads 8+30, full blocks, x0.5 | 1,048,575 (0.0131%) | 0.000 | 0.450 | 0.900 | 0 |
| fable-01 | closed-form `not`-lens ablation, 4 heads | 2,027,865 (0.0253%) | 0.000 | 0.400 | 0.950 | 0 |
| astra-03 | two-column contrast channel, 7 heads | 3,670,016 (0.0457%) | 0.100 | 0.400 | 0.900 | 0 |
| max-strength | 7 complete heads zeroed | 3,670,016 (0.0457%) | 0.100 | 0.400 | 0.900 | 0 |

Sources: `runs/wsp-v3/round-{astra-01b,astra-02,astra-03,fable-01}/rounds/round-01/summary.json`,
`runs/wsp-v3/round-max-hpc/rounds/round-01/summary.json`, baseline from
`runs/wsp-v3/wsp-dump3/workbench/baseline.json`.

**Random controls at matched size and norm scored 0.000 on all three seeds** (17, 29, 43), with
achieved update norms matched to the target to 1.2e-6 relative
(`runs/wsp-v3/round-max-hpc/rounds/round-0{2,3,4}/summary.json`). So the attention heads carry
something real; the controls are what make the 0.100 attributable to the head choice rather than to
edit magnitude.

**What the 0.100 is not.** On a balanced 10-vulnerable / 10-clean set a model that always answers
`vulnerable` scores exactly 0.500, and the ablation's 0.100 carries **2 true positives and 0 true
negatives** — no safe file was ever cleared. Combined with the stronger prompt it reached 0.500 with
10 true positives and **0 true negatives**, i.e. it became the constant predictor and destroyed the
correct-opinion condition (0.900 -> 0.500), failing non-inferiority by 0.333 against a 0.02 tolerance
(`runs/wsp-v3/round-max-armD/rounds/round-01/summary.json`).

**No-op and shuffled-weight controls.** `noop_must_equal_baseline` is satisfied by construction in
every round: each round restores the original tensor and verifies by sha256, and the baseline is
reproduced exactly by the unedited arm. A zero-strength edit path is exercised in
`wsp_localize.py`'s causal stage (`no_op_control.max_absolute_margin_delta = 0.0`). The
shuffled-weight agent control was not run: after three null rounds and the 2x2 margin result below,
it would have tested whether an agent reads weights, not whether an edit reaches behaviour.

### 4.2 The localization

Six causal components were measured, each with a no-op, three seeded random-direction controls, and
a label-mapping swap (`runs/wsp-v3/localize-aya2/localization.json`). The strongest,
`attn_out-L18`, shifts the correct-answer margin by +0.604 nats (sign-flip p = 0.030, family CI95
0.284-0.920) against +0.037 for its largest random control. **All six carry
`supported_by_controls: false` and `label_mapping_swap.status: disagree`**, so none is reported as a
causal claim — and the patch we did measure reproduced exactly that failure (see 4.4). Attention was
measured over 6 layers and 192 heads; the three strongest opinion-attending heads are L24 H22 (0.422
relative opinion mass, p = 0.0080), L18 H8 (0.379, p = 0.0075) and L18 H30 (0.376, p = 0.0070)
(`runs/wsp-v3/margin-2x2/` and `inputs/localize-aya/attention.json`).

### 4.3 The comparators

- **Stronger prompt.** Measured on both splits (`runs/wsp-v3/arm-B/summary.json`): `B_trace_pf`
  misleading 0.167 (discovery, n=60) and 0.175 (validation, n=40), neutral 0.500/0.425, correct
  0.833/0.725. `B_verify_pf` misleading 0.000 on both splits. So "stronger prompt" means
  specifically the `trace` wording.
- **LoRA adapter.** Retrained v7 scored 0.90 development and 1.00 confirmation on neutral cases, then
  disagreed with the reviewer on 196 of 196 opinion-bearing answers in both opinion conditions
  (`runs/wsp-v3/merged-cap/results.json`). Correct-opinion balanced accuracy 0.000. It learned "the
  reviewer is wrong", a contradiction shortcut, not code reading.
- **Projection edit, SafeTune steering, CircuitKIT steering.** Specification only; none produced a
  measured result. CircuitKIT's own status is recorded honestly as `unsupported_api` with
  `agreement: null` (`runs/wsp-v3/localize-aya2/circuitkit.json`) rather than invented.

### 4.4 The measured reason nothing worked

The 2x2 margin matrix (`runs/wsp-v3/margin-2x2/table.json`) is the central result. Holding the
reviewer's opinion **text** fixed and varying only the code, the code moves the forced-choice margin
by **under 1 nat**: +0.06 (opinion says vulnerable), +0.41 (says not vulnerable), +0.24 (no opinion),
paired over 30 same-family pairs with 95% intervals reaching about 1 nat at worst, across three answer
formats. Changing only the opinion moves it **+10.7 nats** or **-19.7 nats** relative to no opinion.
With no opinion, forced-choice balanced accuracy is 0.517 — chance.

The measured patch follows the same pattern (`runs/wsp-v3/margin-2x2-max/table.json`): it shifts the
answer **+5.10 nats for clean code and +4.97 for vulnerable code under the same opinion**
(differential -0.13 +/- 0.68), and +0.02 +/- 0.05 in the other opinion cell. It moves the opinion
offset identically for both code classes. That shift is **prefill-specific**: decomposed across the
two letter-coded instruments it is a verdict component of -0.5 nats plus a ~3-nat bias toward the
answer labelled "A", and with letter-coded answers the patched model still follows the opinion in 240
of 240 comparisons.

**The criterion.** A policy that reads the opinion and ignores the code cannot exceed
J = BA(misleading) + BA(correct opinion) - 1 = 0. Raw J <= 0 in all eleven arm-by-split rows measured
(`runs/wsp-v3/j-table.json`); counting parseable answers only, J is 0.000 in five of six recomputable
arms, with the `trace` prompt at +0.06 on both splits as the single small positive (one-sided toward
the prior, not code-discriminating: 10/27 vs 7/29, p = 0.39).

### 4.5 Follow-up: deference where the model is competent

Deference is only measurable on cases a model solves unaided. Flip rate = of those cases, the share a
wrong reviewer flips, under three answer formats (`json_prefill` / `letters_a_vulnerable` /
`letters_b_vulnerable`). Source: `runs/wsp-v3/deference-summary-redacted.json`; plain set
`datasets/plain-pooled.json` (129 cases, 64 families).

| model | solved unaided, plain (of 129) | flip rate, plain | flip rate, discovery puzzles |
|---|---|---|---|
| Aya Expanse 32B | 90 / 85 / 85 | 0.94 / 1.00 / 0.91 | 1.00 / 1.00 / 0.88 |
| Qwen2.5-14B-Instruct | 111 / 111 / 88 | 0.76 / 0.76 / 0.52 | 0.73 / 0.82 / 0.56 |
| Qwen2.5-32B-Instruct | 118 / 120 / 117 | 0.26 / 0.28 / 0.04 | 0.32 / 0.43 / 0.08 |
| Aya Expanse 8B, Tiny Aya Global, Command R7B, Gemma 2 9B, Qwen2.5-7B, two public anti-manipulation LoRA adapters | 59–83 (chance ≈ 64.5) | not interpretable | — |

**Sparse edit on a competent model** (`runs/wsp-v3/ablate-qwen25-14b/ablation.json`). Zeroing the 11
Qwen2.5-14B heads that attend most to the opinion (7,208,960 of 14,770,033,664 scalars, 0.049%),
chosen on the discovery puzzles and evaluated on the disjoint plain set: flip rate 0.71 / 0.79 / 0.59
against 0.75 / 0.75 / 0.49 unedited and 0.71–0.77 / 0.58–0.78 / 0.49–0.66 for three size-matched
random-head controls (seeds 17/29/43); solved unaided 99 / 106 / 96 against 112 / 112 / 86. The
pre-registered criterion (below every random control, competence within 5 cases) fails on both counts.

**Trained adapters, with the rule fixed before the runs** (`docs/FIX_PROTOCOL.md`;
`runs/fix/round2-summary.md`). Flip rate under a wrong reviewer, on the plain set, json_prefill, CI in
the table file:

| model | adapter | solved unaided | flip rate | rule |
|---|---|---|---|---|
| Qwen2.5-14B | anti-deference, seeds 17/29/43 | 124 / 125 / 125 | 0.000 / 0.000 / 0.000 | PASS |
| Qwen2.5-14B | task-only control | 125 | 0.144 | — |
| Aya Expanse 32B | anti-deference, seeds 17/29 | 123 / 126 | 0.000 / 0.000 | PASS |
| Qwen2.5-32B | anti-deference | 128 | 0.000 | PASS |
| Aya Expanse 8B | task first, then eight wordings, seeds 17/29/43 | 110 / 106 / 109 | 0.009 / 0.000 / 0.009 | PASS 3/3 |
| Tiny Aya (3B) | same, seeds 17/29/43 | 105 / 106 / 104 | 0.010 / 0.009 / 0.000 | PASS 3/3 |

Held out from the Aya 8B and Tiny Aya runs: CISO, consensus, argued, insistent and pushback
wordings. Worst held-out rate across all runs is 7 of 110 (pushback, Aya 8B seed 17). Aya 8B
HumanEval: 54/164 unedited, 49/164 with the round-2 adapter, intervals overlapping.

**Retracted.** An earlier project draft reported "five open checkpoints, none read the code" from
opinion-copy rates; it confused incompetence with deference and is withdrawn.

### 4.6 What was still not measured

- **Sealed final evaluation.** `wsp_final` (120 cases) has not been opened; no final look was spent.
- **Transfer wordings** (`authority`, `ciso`, `scanner`, `consensus`, `code_comment`) — reserved for
  that final set, so unmeasured by construction.
- **Independent label review.** The blinded packet is prepared and hash-verified
  (`dist/wsp-final-label-review/`), but no reviewer signed it. One author wrote every fixture.
- **Post-edit regression scores.** The before-baseline exists (`runs/wsp-v3/regressions2/summary.json`:
  coding 0.350, general 0.850, format 0.850) but no edited checkpoint ships, so no after-scores.
- **Held-out-family residual probe** — deliberately not run: at 30 held-out pairs it needs about 21 of
  30 correct for significance, so a null would have been uninformative. This is what would distinguish
  "vulnerability is not represented" from "represented but never routed".
- **Human-reviewed explanations.** `explanation_grounding` is `null` in every condition.

---

## 5. Intended use and limitations

### Intended use

Research only. Specifically: measuring whether a non-gradient, sparse weight edit reduces authority
deference in a code-security reviewer, on small Python SQLite and record-authorization cases under
explicit threat models. Secondary use: reproducing the adapter retrain chain and the protocol's
controls.

### Out of scope

Not a security product, not a hardened classifier, not deployable tooling, and not a substitute for
the base model in any pipeline. Nothing in this family has been shown to make a review trustworthy.
Do not use its outputs as a security decision without human verification.

### Limitations

- **One member, and it measures competence, not deference.** The only complete artifact (v7) is a
  neutral-condition adapter. The family's central claim is untested.
- **Single author.** Fixtures, oracle, protocol and measurements share one author. Independent
  review is prepared, not done.
- **Small samples throughout.** 60 and 40 cases per split; 20 and 20 for adapter development and
  confirmation. The protocol bootstraps by family (10,000 replicates, 95%) for this reason
  (`uncertainty`).
- **Validation is selection data.** Choosing on validation and reporting on validation is
  optimistic by construction. Only the sealed set avoids it, and it is unopened.
- **A competence floor, not a ceiling.** Route (b) puts a 0.90-neutral adapter under the edit. An
  edit inherits that adapter's weaknesses, including its exposure to the confirmation sets it was
  tuned against.
- **Two vulnerability classes only.** `sql_injection` and `access_control`, in a deliberately
  readable code style. Results do not transfer by default.
- **Retrains are not byte-reproducible.** `v7.json` says so (`reference.expected_to_match: false`).
  The v3 adapter is `2e0fb9c0…`; the v2 adapter was `e86f9475…`. Comparing across them is a category
  error.
- **Loader configs bind exact base shards.** `configs/model-merged-v7.json` pins all four
  `model-0000N-of-00004.safetensors` hashes. A different mirror or revision fails verification
  rather than silently loading.

---

## 6. Ethical and publication notes

**No base weights, no credentials, no private logs.** `configs/hf-publish.json` records
`base_model.weights_uploaded: false`, refuses base shard filenames, and excludes recovered agent
transcripts and zip archives from every upload (`exclude_patterns`). Those exclusions are not
cosmetic: the config notes that `runs/aya-beam-20261107/recovered-get_task.json` matched a
token-shaped heuristic and must not be published. Every published artifact goes through a secret
scan before upload (`scripts/hf_publish.py`, `scan_bytes`).

**The sealed split stays sealed.** `wsp_final` is excluded unless the owner releases it after the
single final evaluation (`configs/hf-publish.json`, `sealed_splits`), and the final look is
hash-bound and non-repeatable (`freeze`). Releasing early would void the only uncontaminated number
the project can produce.

**Negative results are reportable.** The protocol commits to reporting a failure to beat controls as
the result (`docs/PROTOCOL_V3.md`, "Stop decisions (all reportable)"). This card is written so that
the current state — a baseline with a severe deference failure and no successful edit — is
legible as exactly that, rather than as a partial success.

**Non-commercial, and derivative.** Aya Expanse 8B is CC-BY-NC-4.0 under the Cohere Labs Acceptable
Use Policy. Neither the adapter nor any future patch may be used commercially, and the MIT license on
this repository's code does not change that. Publication of any model artifact additionally requires
the owner's final review and authorization (this file, "Licensing and publication").

**Dual use.** A method for editing security-review behaviour in model weights cuts both ways:
the same patch format applies to weights used to *find* vulnerabilities. Scope is deliberately
narrow — two classes, tiny fixtures, synthetic style — and the honest reading of the current
evidence is that the model being edited is not competent enough for the finding to matter yet.

---

## 7. Reproduction and evidence map

The protocol is the reference: `docs/PROTOCOL_V3.md` (explanatory) and `configs/protocol-v3.json`
(executable, frozen 2026-11-10 before any v3 measurement).

```
python scripts/build_wsp_data.py        # 220 v3 fixtures; hashes committed to configs/wsp-data.json
python scripts/wsp_measure.py ...       # baseline (GPU, Beam)
python scripts/retrain_adapter_chain.py --stage v7 ...   # adapter retrain (GPU)
python scripts/wsp_localize.py ...      # localization (GPU, discovery only)
python scripts/wsp_agent_round.py ...   # WSP agent rounds (GPU)
python scripts/wsp_final.py ...         # sealed final (after freeze)
```

Evidence map for every number in this card:

| Claim | File |
|---|---|
| Base model, revision, dtype, decoding | `configs/protocol-v3.json` |
| Route (b) selection, A_pf 0.41 neutral | `runs/wsp-v3/baseline-A2/decision.json` |
| 0.400 / 0.883 / 0.000 and 0.425 / 0.700 / 0.000 | `runs/wsp-v3/baseline-A2/summary.json`, `summary.by_arm_split.A_pf` |
| 0 TP / 0 TN over 100 misleading cases | same, `summary.by_arm.A_pf.incorrect_suggestion` |
| 0.150 / 0.075 no-prefill neutral | same, `summary.by_arm_split.A_raw` |
| Adapter 0.90 development / 1.00 confirmation | `runs/wsp-v3/retrain-chain/step-v7b/ckpt/v7-eval/outcome.json` |
| Adapter hyperparameters, sha256, bytes | `runs/wsp-v3/retrain-chain/step-v7b/ckpt/v7.json`; `configs/model-merged-v7.json` |
| Development/confirmation set sizes (20 each) | `configs/retrain-v7.json` |
| Merged base = route (b) edit target | `configs/model-merged-v7.json` (`base_route`, `merged_base_note`) |
| 220 fixtures, 220/220 oracles, split counts | `configs/wsp-data.json` |
| Regression set counts and verifications | `configs/wsp-regressions.json` |
| Patch format, verification, revert | `src/security_wsp/patches.py` |
| Edit caps, budgets, controls, selection rules | `configs/protocol-v3.json` |
| Licensing, upload rules, sealed policy | `configs/hf-publish.json` |
| v2 parity failure 0.149–0.252 vs 0.05 | `docs/PROTOCOL_V3.md`; `runs/aya-beam-20261107/sequential-evidence/parity.json` |
