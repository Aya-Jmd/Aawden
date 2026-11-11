# Phase 3: localization

`scripts/wsp_localize.py` finds where in Aya the deference-to-a-wrong-reviewer behaviour is measurable,
and tests the strongest candidates causally, before any weight patch is attempted. It is the phase-3
step of protocol v3 (`docs/PROTOCOL_V3.md`, `configs/protocol-v3.json`); the weight edit itself is
phase 4 and is not part of this script.

    python scripts/wsp_localize.py --dataset datasets/wsp-discovery.json \
        --output runs/wsp-v3/localize --layers all --top-k 6

    # offline plumbing on a tiny random model, CPU, about two seconds
    python scripts/wsp_localize.py --toy --limit 6 --layers 0,1 --top-k 2 \
        --attention-cases 3 --attention-call-cap 8 --causal-cases 4 --bootstrap-replicates 200 \
        --output /tmp/localize

## What it refuses

| refusal | why |
| --- | --- |
| any dataset other than `datasets/wsp-discovery.json` | validation and final are never visible to localization; the path must also match a commitment in `configs/wsp-data.json` and its sha256 must be that value |
| `datasets/wsp-final.json` | sealed: it opens only through final-evaluate after a freeze |
| a copied dataset with the right bytes | the path commitment is part of the check, so a copy is a different dataset |
| `--wording` other than `reviewer` | the five transfer wordings (`authority`, `ciso`, `scanner`, `consensus`, `code_comment`) are final-only in protocol v3 |
| `--condition` other than `incorrect_suggestion` | the frozen primary condition |
| a grid whose planned forward units exceed `--budget` | the grid is refused whole; layers are never silently sampled down |
| a no-op control that moves a margin by more than 1e-5 | the harness is invalid, so the run fails instead of reporting |
| a prompt over the context limit | truncation is prohibited |

## The three stages

### 1. Activation contrast: `components.json`

For each site and layer, `WspTools.activation_contrast` runs the neutral and misleading prompts over the
discovery cases and captures the answer-position activation. Sites are:

- `resid_post` — the decoder layer output, i.e. what the block writes into the residual stream;
- `mlp_out` — `mlp.down_proj` output;
- `attn_out` — `self_attn.o_proj` output.

Per component the artifact records the **separation statistic** (‖mean incorrect−neutral contrast‖ divided
by the RMS per-case contrast norm, in [0, 1]), its **permutation p-value** from seeded sign flips, the
**random-direction baseline** (the same statistic on 32 seeded unit directions, which is the floor a
meaningless direction reaches), the mean cosine of each case contrast to the mean direction, and the mean
pairwise cosine. Components are ranked by

    score = separation statistic − mean random-direction separation

descending, then by p ascending, then by site order and layer. A component whose contrast is degenerate
is ranked last and is never checked causally. The ranking is a filter for what to test, not a result.

### 2. Attention: `attention.json`

Which heads put more of the answer position's attention on the reviewer-opinion sentence when the
opinion is wrong. `WspTools.attention_to_span(layer)` returns every head of a layer from one call, so the
subsample is (case, layer) pairs, one call each, and `--attention-call-cap` is a cap on the **total number
of calls**, not on layers per case: `calls_used = layers_used × cases_used ≤ call_cap` always. Layers are
subsampled first (they are the quantity the stage studies); cases are cut only when the cap cannot afford
even one layer over every case. The artifact reports `call_cap`, `calls_used`, `call_budget` (the cap
clamped to the pairs the subsample can offer), `calls_per_layer`, `layers_used`, `layers_dropped_by_cap`,
`cases_dropped_by_cap` and `tool_units_used`. A case whose opinion or source span is not a unique token
match is excluded and counted in `match_failures`, with the reason (`no match in prefix`, `N matches in
prefix`) and the case id; nothing is dropped silently. If no case is usable the heads carry
`status: "no_usable_cases"`.

How to read a head row:

| field | meaning |
| --- | --- |
| `mean_opinion_mass` | absolute attention mass on the opinion span under the misleading prompt |
| `mean_relative_opinion_mass` | that head's mass minus the mean mass of all heads of the layer on the same cases; positive means "more than its siblings" |
| `relative_test` | sign-flip permutation p-value over the per-case relative differences |
| `relative_interval` | family bootstrap interval of the same mean |
| `mean_source_mass_shift` | source-span mass under misleading minus neutral; negative means the opinion sentence partly displaces the code |
| `source_shift_test` | sign-flip p-value of that shift |

The relative statistic is used because the tool reports per-head masses only, with no token
normalisation, so absolute masses are not comparable across heads with different attention entropy. A
head is interesting when the relative test is small and the interval clears zero; that is still
exploratory, because attention mass is not causal.

### 3. Causal check: `localization.json`

For each of the top-k components (default 6) the mean direction from stage 1 is projected out of the
residual-output weights that carry that component (`resid_post` projects both `o_proj` and `down_proj` of
the layer; `mlp_out` only `down_proj`; `attn_out` only `o_proj`) with
`interventions.project_output_tensor`, and the effect on the **correct-answer margin** is measured.
Every touched tensor is hashed before and after; a fill that does not restore the base model raises.

The margin is the summed log-probability of the correct verdict continuation minus the wrong one, read at
the answer position with the full continuation and never a single token. Three instruments are scored for
every case and every arm:

| instrument | suffix | continuations | role |
| --- | --- | --- | --- |
| `json_prefill` | protocol v3's `{"verdict":"` | `vulnerable`, `not_vulnerable_for_target` | primary; identical for every arm, and the same prefill every method receives |
| `letters_a_vulnerable` | forced-choice sentence where A is vulnerable | `" A"`, `" B"` | label-mapping-swap control |
| `letters_b_vulnerable` | same, A is clean | `" A"`, `" B"` | the swap's other half |

The letter mapping is written into the prompt text, so the letters are a diagnostic control, not a second
evaluation; the balanced accuracies in `scripts/wsp_measure.py` remain the evaluated quantities.

Arms per component, all on the same cases:

- `mean_direction` — the stage-1 direction at `--strength` (default 1.0, full projection);
- `noop` — the same tensors at strength 0.0; its margins must match the untouched model within `1e-5`
  or the run fails;
- `opposite_direction` — the negated direction at the same strength. A projection is idempotent and sign
  symmetric: `W - d(dᵀW)` removes everything along `d`, so `+d` and `-d` produce the same tensor. Both arms
  must therefore agree within `1e-5`, and disagreement would mean the projection is implemented wrongly.
  This is a verification, not an independent control, which is why it does not count as a fifth comparison;
- `random-17`, `random-29`, `random-43` — seeded random unit directions at the same strength on the same
  tensors, the protocol's random control seeds.

Each arm reports, per instrument, the mean per-case margin difference against the untouched baseline, a
seeded sign-flip p-value, a family-bootstrap interval over `family_id`, the fraction of cases whose margin
moved in the helpful direction, and `per_verdict` — the same differences split by oracle verdict.

The per-verdict split matters: a projection whose overall mean is near zero because it helps clean cases
while hurting vulnerable ones is not a deference edit. The causal label requires both classes positive. Forced-choice predictions built from the margin signs feed
`statistics.paired_cluster_interval` (`mean_direction` against `noop`), so the protocol's own family-level
uncertainty machinery applies where it is defined.

## The exploratory-claim policy

A component is labelled **"control comparison supports this component as causal for the margin"** only
when all of the following hold on the primary instrument:

1. the no-op control moved no margin by more than `1e-5`;
2. the family-bootstrap interval of the paired margin difference excludes zero;
3. the sign-flip p-value is below 0.05;
4. the effect exceeds the largest seeded random-direction effect;
5. the **label-mapping swap** passes — the same literal continuation shifts by the same amount under
   both A/B mappings (within `MAPPING_TOLERANCE = 1e-3` nats on the case mean). A letter margin is
   verdict-referenced, so it necessarily changes sign when the mapping swaps; the invariant that must hold
   is on the raw per-continuation shift. Disagreement means the effect tracked the mapping sentence rather
   than the verdict;
6. both verdict classes move in the helpful direction (`per_verdict`).

Everything else is labelled **"exploratory localization, not a causal claim"**. Both labels are the
literal `conclusion` values in `localization.json` and `components.json`, so a downstream reader never has
to interpret prose. A component can rank first on separation and still fail the causal check; when nothing
passes, the protocol reports deference as diffuse and the phase-4 search still runs.

## Cost

The unit is one physical forward: one batch through the model (a batch of up to `--batch-size` prompts
counts once), or one scored continuation in the causal stage.

| stage | units per call | calls | example |
| --- | --- | --- | --- |
| activation contrast | ⌈2 × cases / batch⌉ | 3 sites × layers | 60 cases, batch 16 → 8 units; 96 calls on 32 layers → 768 units |
| attention | ⌈2 / 4⌉ = 1 (attention batch is 4) | cases × layers used, total ≤ `--attention-call-cap` | 8 cases, 32-call cap → 4 layers, 32 units |
| causal | 1 per scored continuation | baseline table, then every arm's table | 12 cases, 3 instruments × 2 continuations: 72 forwards for the baseline, 72 per arm. 7 tables (baseline + no-op + mean + opposite + 3 random seeds) → 504 forwards per component |

The tool budget is planned before the first forward pass and the whole grid is refused if the planned
activation + attention units do not fit in `--budget` (default 2048). The causal stage scores continuations
directly rather than through `WspTools`, so it is **not** charged to the tool budget; it is metered by its
own counter (`UnitMeter`, called once per physical forward in `continuation_scores`) and reported
separately. `localization.json`'s `cost` block therefore carries `stages` (activation, attention and causal
with their units, `planned_units` where a plan exists, `metered_by` and `charged_to_tool_budget`),
`tool_units_used` (activation + attention, the quantity the budget check bounds) and `total_units_used`
(all three stages). `report.md` prints the same table under "5. Cost", so the headline forward-pass number
is the total rather than the activation count. A full 32-layer, 60-case run is roughly 800 tool units plus
504 forwards per causally checked component (top-k 6 → about three thousand), which is a GPU hour at Aya's
size, not a Beam job: this script never launches one.

## Outputs

| file | contents |
| --- | --- |
| `localization.json` | instruments, the causal results with controls, cost accounting, model metadata, dataset hashes |
| `components.json` | the ranking rule, every component's statistics, the top-k list |
| `attention.json` | layers used and dropped, the call cap, per-head rows, span-match failures |
| `circuitkit.json` | the optional cross-check record |
| `report.md` | human-readable tables generated from the same values, with the claim labels |
| `manifest.json` | `RunStore` identity, status, attempt log, and the sha256 of every artifact and prediction |
| `predictions/` | one hashed record per contrast, per baseline margin table, per causal component, plus attention and circuitkit |

Every artifact is written through `security_wsp.artifacts.RunStore`, whose identity binds the protocol
hash, the data commitments, the script hash, the package source fingerprint, the model config (or the toy
seed), the prompt version and a digest of every rendered prompt, the instrument definitions, the layer
list, the case limit, every sampling parameter, and **every flag whose value changes artifact content**:
the forward-pass budget, whether the CircuitKIT cross-check is enabled, and the toy seed (the budget
appears in `localization.json` and the report; `--no-circuitkit` changes `circuitkit.json` and
`localization.json`). An identical re-run resumes and reproduces byte-identical artifacts; any change is
rejected with "Resume rejected", including a changed `--budget` or a flipped `--no-circuitkit`. The run
ends with `verify_run`, which re-checks the identity digest and every artifact hash, and a tampered
artifact makes it raise. `report.md` is written through the same temp-file-then-`os.replace` sequence as
the JSON artifacts, so a crash cannot leave a truncated report for `verify_run` to hash.

## CircuitKIT

The cross-check is optional and never faked. `circuitkit.json` records one of:

- `absent` — the import failed, with the exact `ModuleNotFoundError` text (the local state);
- `disabled_by_flag` — `--no-circuitkit`;
- `unsupported_api` — the package imports but exposes no EAP-IG entry point;
- `available_but_not_run` — the package imports and exposes EAP-IG, but no run is made, with the
  discovered signatures and the reason: neither this repository nor the package pins the task adapter, the
  target logit pair or the score normalisation, so any agreement figure would be invented. Agreement stays
  `null`.

Protocol v3 makes CircuitKIT exploratory only (`parity_redefinition.circuitkit_role`), so its absence does
not block localization or the edit.

## The `--toy` mode

Protocol v3's `models.ToyCircuit` is a two-input linear circuit with no addressable layers, so `WspTools`
cannot be backed by it. `--toy` therefore builds the same tiny random Cohere model the WSP tests use: two
layers, hidden 64, four heads, a word-level tokenizer with punctuation splitting (so the JSON prefill
continuation is a token-level extension of its prefix), and re-drawn residual-output and LayerNorm weights
so the activations depend on the prompt. It runs on CPU in about two seconds and produces no evidence
about Cohere; the manifest records `evidence_kind: "constructed_toy"` and the model metadata says so.

## Tests

`tests/test_wsp_localize.py` runs the toy end to end in `tmp_path` (all five artifacts, a manifest that
`verify_run` accepts, and a tampered artifact that it rejects), checks refusal of the validation and final
datasets, of an uncommitted copy, of every final-only wording and of a non-primary condition, checks that a
budget-infeasible grid is refused rather than sampled, checks byte-identical artifacts across two runs,
checks resume with an identical identity and rejection with a changed one — including a changed `--budget`
and a flipped `--no-circuitkit` — and unit-tests the paired statistics, the exact negation of a true
label-mapping swap, the mapping-swap control on raw shifts, the per-verdict split, the exact equality of the
no-op arm to the baseline, the degenerate-component ranking and the CircuitKIT absence path on a tiny
`AutoModelForCausalLM.from_config` model. Specifically for the guarantees this document makes:

- the attention cap binds **total** calls: a plan with 8 cases and a cap of 4 must fit 4 calls in total and
  report the layers it dropped, over a grid of case counts, layer counts and caps, and a stage-level test
  with a fake tools factory asserts `calls_used = calls_issued ≤ call_budget ≤ call_cap` and that one metered
  unit is charged per issued call;
- the forced-choice rows read the prediction off the margin sign: a table that prefers the oracle
  continuation on every case gives `balanced_accuracy` 1.0 with `true_negative` equal to the clean count, an
  anti-oracle table gives 0.0, and a mixed table gives 0.5;
- the causal stage's own meter is pinned to an exact expected count (`cases × instruments × continuations`
  for the baseline plus the same again for each of the 7 arms), and `cost_block` is checked to sum all three
  stages while charging only activation and attention to the tool budget;
- `report.md` is written through the atomic helper: the helper replaces atomically and leaves no temp file,
  and the source contains no bare `write_text`.

All of it is offline, CPU-only and about 50 seconds in total.
