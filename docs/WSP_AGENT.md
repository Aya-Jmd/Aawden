# WSP agent: the weight-space-programming round protocol

Phase 4 of protocol v3. An external frontier LLM (the **programmer**) writes sparse weight edits for
Cohere Aya Expanse 8B by reading a compact workbench and nothing else. The programmer never runs
inference, never sees validation or final data, and its turns happen between GPU jobs. This document
describes the loop, the edit-spec interface, the budget arithmetic, and every way the driver refuses.

The driver is `scripts/wsp_agent_round.py`. It is the only thing that touches the model. It is
resumable from disk alone: `dump` is cheap and runs on any host that can load the model; `apply`
and `random-control` run one round each. Nothing is carried in memory between invocations.

## Why this shape

The frozen protocol (`configs/protocol-v3.json`, `docs/PROTOCOL_V3.md`) bounds the search:

| protocol key | value | meaning here |
| --- | --- | --- |
| `wsp_agent.visible_data` | `wsp_discovery` | the programmer sees discovery only |
| `wsp_agent.forward_pass_budget` | 50 | units available for all rounds plus workbench probes |
| `wsp_agent.max_submissions` | 10 | persisted patches, counted from disk |
| `wsp_agent.max_changed_fraction_of_weights` | 0.0005 | of all model parameters, per spec |
| `wsp_agent.editable_sites` | 4 tensor patterns | `mlp.down_proj`, `self_attn.o_proj`, `mlp.up_proj`, `mlp.gate_proj` |
| `wsp_agent.no_gradients` | true | values only, no loss, no backward pass |
| `selection.max_edit_candidates_per_method` | 10 | edited recipes, proposals and controls |
| `controls.random_control_seeds` | 17, 29, 43 | size- and norm-matched controls |

`scripts/wsp_measure.py` measures baselines and applies selection rules. `WspTools` in
`src/security_wsp/wsp_tools.py` is the only tool layer; this driver calls it rather than
reimplementing any of it. See `docs/WSP_TOOLS.md` for that API.

## The loop

```
GPU job 1:  dump                      -> workbench/          (cheap; 0-3 units typically)
                  programmer turn 1   -> edits/round-01.json  (offline; no model)
GPU job 2:  apply --spec edits/round-01.json  -> rounds/round-01/
            check --spec edits/round-02.json  (validates, costs nothing, loads no model)
                  programmer turn 2   -> edits/round-02.json
GPU job 3:  apply --spec edits/round-02.json  -> rounds/round-02/
            random-control --from 2   -> rounds/round-03..05/
...
GPU job n:  several apply rounds in one job, until the budget or submission limit stops it
```

One job may run several rounds: each is one `apply` invocation, and each is independent because
`state.json` is rewritten after every round. A job that dies between rounds loses nothing; the next
job continues from the same counters. A job that dies *during* a round leaves the model untouched
(the edit is restored in a `finally` block and verified by tensor hash before the result is written).

### Modes

| mode | what it does | forward units |
| --- | --- | --- |
| `dump` | builds `workbench/`: sites, contrasts, attention, baseline, schema, README | 0 if cached, else 1 per contrast and per baseline round |
| `check --spec F` | validates a spec against the saved workbench; loads no model | 0 |
| `apply --spec F` | validates, scores on discovery, writes a round, persists state | one round (see arithmetic) |
| `random-control --from N [--seeds ...]` | size- and norm-matched control of round N | one round per seed |

`--random-control` is also accepted as a flag with `--from`, so both
`random-control --from 1` and `--random-control --from 1` work.

## The edit-spec schema (the interface the programmer writes)

Format `aawden-wsp-edit-spec-v1`, version 1. Strict JSON: unknown keys are refused at every level.

```json
{
  "schema_version": 1,
  "rationale": "one or two sentences: why these slices, why these values",
  "edits": [
    {
      "tensor_name": "model.layers.31.mlp.down_proj.weight",
      "axis": 0,
      "indices": [0, 7],
      "values": [[0.0, 0.0, "...one entry per in-feature..."], [1.0, "...", "..."]]
    }
  ]
}
```

| field | rule |
| --- | --- |
| `tensor_name` | must equal a `tensor_name` in `workbench/sites.json`. Unknown tensors are refused. |
| `axis` | `0` overwrites complete **rows** (one per index), `1` overwrites complete **columns**. |
| `indices` | strictly ascending, unique, each in `[0, size)` where `size` is `shape[axis]`. |
| `values` | exactly one slice per index, in index order: `shape[1]` numbers for axis 0, `shape[0]` for axis 1. Every value finite. |
| one edit per tensor | a tensor may appear once per spec; a second appearance is refused. |
| changed fraction | `sum(len(indices) * slice_width) / total_model_parameters <= 0.0005`. |

The same schema is written to `workbench/schema.json` as a JSON Schema document and illustrated in
`workbench/README.md`, which is written for the programmer.

### Worked budget example

The exact parameter count of the pinned checkpoint is not recorded in this repository, so the
following uses a nominal 8 billion: 0.0005 x 8e9 = about 4 million scalars per spec. With
`hidden_size` 4096, one row of `mlp.down_proj` or `self_attn.o_proj` is 4096 scalars, so a spec can
overwrite on the order of a thousand full rows across the four tensor kinds. That is what "sparse"
means here: a very small fraction of the model, but not a handful of numbers. The driver computes
the real budget from the loaded model (`total_parameters`) and states the permitted scalar count in
every refusal, so never rely on this estimate.

The toy model in the tests is sized to behave the same way. On the driver's own path the bundle is
built from the whole discovery split, which puts a 273-entry word-level vocabulary into the
embedding, and the configuration used by `--toy` (hidden 96, intermediate 256, 4 layers, 8 heads,
8 kv heads) has **469,056** parameters. At the protocol fraction, `0.0005 x 469,056 = 234.5`, so the
limit admits **234 scalars**: two 96-wide rows (192) fit, three (288) do not, and one full column of
a 96-row tensor fits. Those are measured numbers, taken from the built model
(`sum(p.numel() for p in model.parameters())`), not estimates. A four-case `--limit-cases` prefix
shrinks the vocabulary and yields a smaller total (454,944 parameters, 227 scalars); the driver's
run path uses the full-split build. The real 8B checkpoint is a different scale entirely, and its
exact parameter count is not recorded in this repository.

## Budget arithmetic

The protocol's unit is a **forward pass**: "one try_edit or measurement call over at most 64
discovery prompts". `WspTools` charges per physical batch, which is stricter and is what this driver
reports. With `batch_size = 16`, one unit is at most 16 prompts.

The committed discovery split is **60 cases** (`configs/wsp-data.json`). Every scored recipe runs all
three conditions, so:

| quantity | value |
| --- | --- |
| prompts per scored round | 60 cases x 3 conditions = **180** |
| batches per scored round | ceil(180 / 16) = **12** |
| forward units per scored round | **12** |
| rounds affordable from a fresh 50-unit budget | 50 / 12 = **4** |

The figure of 100 discovery cases sometimes quoted for this phase is not what is committed; at 100
cases a round would be 300 prompts, 19 batches, 19 units, and only two rounds would fit. The driver
reads the real count from the dataset and reports the arithmetic it used in
`workbench/protocol.json -> budget_arithmetic`, so the number in a run is never assumed.

### What a job actually spends

| item | units | note |
| --- | --- | --- |
| `dump`, cold (no cache, one contrast, no attention) | 8 contrast + 12 baseline = **20** | activation contrast is 2 prompts per case = 120 prompts = ceil(120/16) = 8 batches; the baseline is one scored round |
| `dump`, warm (`baseline.json` and `contrasts.json` already on disk) | **0** | reused from disk; the workbench records the source |
| `dump`, cold with `--baseline-scores` from an existing `wsp_measure` summary | **8** | the baseline is read and hashed, not recomputed |
| one `apply` round | **12** | 180 prompts, 12 batches, one edit set |
| one `random-control` seed | **12** | same prompts, same scorer |
| `attention_to_span` (opt-in, `--attention-layers`) | 30 per layer at the default `--attention-batch-size 4` | 2 prompts per usable case = 120 prompts, ceil(120/4) = 30 batches. Costs most of the budget for one layer, so it is off by default. |
| attention and `causal.json` read from a handed-in `--localization` | **0** | `wsp_localize.py` writes `attention.json` beside `localization.json`; the dump reads the ranked heads and the causal components from those files. The measurement already happened, so nothing is charged again. |

The budget shapes the run, so choose deliberately:

* **Warm dump** (a `baseline.json` and `contrasts.json` already on disk, or `--baseline-scores`
  supplied): the dump costs 0 and four scored rounds fit (4 x 12 = 48 of 50).
* **Cold dump, one contrast:** 20 units, leaving 30, which is two scored rounds (2 x 12 = 24) and 6
  units spare. This is the default and the honest shape for a first run: one cold dump, then two
  rounds, and the second job's dump is warm.
* **Cold dump with attention on one layer:** 20 + 30 = 50, which spends the entire budget before any
  edit is scored. Attention is therefore off by default and should be enabled only when the contrast
  evidence is insufficient and one round is enough.

On the RTX 5090 the wall clock is dominated by generation, not by the batch count. For scale: the
recorded RTX 4090 baseline generated 150 Aya answers in 449.87 seconds (`runs/aya-beam-20261107/
evidence/baseline/report.md`), about 3.0 seconds per answer with `max_new_tokens=256` at batch 1. A
180-prompt round batched 16-wide is 12 generator calls; the exact wall clock for the 5090 is
**UNKNOWN - not measured** on this project. Measure it with one `apply` round before sizing a
multi-round job, and use the measured seconds rather than a projection.

### Smoke runs

`--limit-cases N` takes a balanced prefix (equal vulnerable and clean) so balanced accuracy stays
defined. With `--limit-cases 4`, a round is 12 prompts = 1 batch = 1 unit, which is what the offline
tests use. Smoke runs are for plumbing only; they are not evidence.

## Round artifacts

```
<output>/
  state.json                      identity, budget, counts, per-round records with artifact hashes
  workbench/                      what the programmer reads (byte-capped)
    README.md  protocol.json  sites.json  contrasts.json  causal.json
    attention.json  baseline.json  schema.json  manifest.json
  rounds/round-NN/
    spec.json                     the exact spec, byte-for-byte as accepted
    results.json                  the full WspTools.evaluate_edits output, including every row
    summary.json                  metrics, per-class counts, restoration hashes, diff
  submissions/submission-NN.{safetensors,json}   persisted patches from WspTools.submit
  transcripts/transcript-NN-*.json               every tool call in order, with unit costs
```

Every number a round reports comes from that call's own saved output:

* `summary.json -> metrics -> <condition> -> balanced_accuracy` and the per-class counts
  (`true_positive`, `true_negative`, `false_positive`, `false_negative`, `invalid_count`,
  `n_vulnerable`, `n_clean`).
* `summary.json -> restoration -> tensor_sha256_before` / `tensor_sha256_after_apply`: the same
  hashes `WspTools` verifies on restore. `restoration.verified` must be true.
* `summary.json -> diff`: deltas against the previous round (including how many slices are shared,
  added and removed) and against the baseline, all from saved numbers.
* `state.json -> rounds[].artifacts`: sha256 of `spec.json`, `results.json` and `summary.json` as
  written to disk.

`summary.json` itself cannot carry its own hash, so its hash lives in `state.json`.

## Refusals

Every refusal prints the reason and exits non-zero. Nothing is written to a weight before validation
passes and the budget is reserved.

**Data.** The **sealed final** split (`datasets/wsp-final.json`) is refused by name before any file
handle is opened, and no code path in this script reads it. The check is by resolved path and by
filename, both taken from `configs/wsp-data.json` rather than reconstructed from a name pattern, so a
copy of the sealed file under another name is caught by its committed sha256 as well. A **validation
split** is refused too: the programmer sees `wsp_discovery` only, and selection on validation happens
in a later stage with `scripts/wsp_measure.py`. Any dataset whose sha256 does not match
`configs/wsp-data.json` is refused, and so is a case whose split is `wsp_validation` or `wsp_final`.

The same guard covers the helper inputs. `--localization` and `--baseline-scores` are checked for
the sealed and validation splits under any name before either file is opened, so neither can be
smuggled into a round that way. Both are then **pinned by sha256** on first use, in `state.json ->
inputs` and in `workbench/protocol.json -> inputs`: a file that is **changed since this output
directory was created** is refused rather than read, and the refusal happens before any forward unit
is spent. The reason says the input **changed since this output directory was created**, so a reader
knows which file was pinned and which file replaced it.

**Wordings.** `--wording` is refused when it names a **transfer wording** listed in
`protocol["transfer_wordings_final_only"]` (`authority`, `ciso`, `scanner`, `consensus`,
`code_comment`), and refused when it is any wording other than the development wording `reviewer`.

**The spec.** A spec is refused when a `tensor_name` is an **unknown tensor**, when indices are not
ascending (**indices must be sorted**), when they repeat (**indices must be unique**), when an index
is **out of range** for that axis, when a slice has the wrong length, when a value is not finite,
when an unknown key appears, when a tensor appears twice, or when the **changed fraction** exceeds
0.0005 (the error states the scalar arithmetic and the exact permitted count). All problems in a
spec are reported together, not one at a time. `check --spec` runs this validation with no model and
no budget, so the programmer can test a spec cheaply.

**The budget.** A round is refused when it needs more forward units than remain
(**forward-pass budget**), and a submission is refused past the **submission limit** (10) or past the
**edited-recipe limit** (10). Units are booked per physical batch: `evaluate_edits` charges
`batches x number of edit sets`, `activation_contrast` charges the batches it runs, and
`list_sites`, `read_weights`, `propose_random_control`, `submit` and `read` of the state cost
nothing. The reservation happens before any weight is written, so an exhausted budget leaves the
model untouched.

**No truncation, no gradients.** No prompt is truncated. A prompt that does not fit the context
limit raises **truncation** rather than shortening the source; the round fails instead of scoring a
cut prompt. No gradients are ever taken: the driver writes values and never builds a loss or runs a
backward pass.

**State.** Reusing an `--output` directory with a changed dataset, protocol, model, scoring setting
or `--limit-cases` value is refused, so a round can never mix two identities.

`state.json` is also **hash-chained**. Every write seals the whole state body with a link that
records the sha256 of the previous link's head, and every link is appended to
`state-chain.jsonl`. Loading refuses a file that fails its **hash-chain** check, so editing the
budget or the counts across restarts cannot exceed the protocol caps, and it refuses a state whose
chain disagrees with the ledger, so a state file rewritten with a freshly computed chain does not
pass either. Every round artifact hash recorded in `state.json` is re-verified on load: a prior
`summary.json` or `results.json` that was **edited afterwards** is refused instead of being consumed
by the next round's diff. This is tamper-evidence, not a signature: someone who rewrites the state,
the chain and the ledger together, or who restores an older copy of the whole directory, is not
detected from the files alone.

**The workbench cap.** `dump` refuses rather than writing an oversized workbench; the error states
the minimum size that would fit. What was reduced is recorded in `workbench/manifest.json -> truncated`
and in the README's own Truncation section.

## Resume and identity

`state.json` records an identity: protocol version and sha256, dataset path and sha256,
`--limit-cases`, the model fingerprint (adapter, model id, revision, dtype, checkpoint hashes) and
the scoring settings (strong prompt flag, variant, wording, batch size, prompt limit). `apply` and
`random-control` refuse to resume unless the recomputed identity matches, so a stale directory can
never silently accept a different model or split. The model fingerprint deliberately excludes
environment noise (GPU name, library versions), which legitimately differ between GPU jobs.

## Reading a round

The result of `apply` is the JSON a programmer reads next. It carries, for each condition, balanced
accuracy and the per-class counts; the changed and touched scalars with their fractions; the update
norm of the edit and of each tensor; the restoration hashes; the diff against the previous round and
the baseline; the submission index and patch path; and the remaining budget, submissions and
recipes. The full artifact is `rounds/round-NN/summary.json`; the entire per-answer record is in
`results.json`.

For the endpoint itself, the protocol requires the misleading-condition gain of at least
`minimum_gain` against method B, with a paired family-bootstrap interval above zero, and
non-inferiority on neutral and correct-opinion within 0.02, all measured on validation and then
final by other scripts. This driver produces discovery evidence for selection; it does not claim the
endpoint.

## Controls

`random-control --from N` builds the protocol control: the same tensors, the same axis, the same
number of changed slices, random slice indices from a seeded permutation, Gaussian values rescaled
so the total update norm matches the proposal (`WspTools.propose_random_control`). The norm is
measured after casting to the tensor dtype and the relative mismatch is reported with a
`within_1e-3` flag. Controls are scored by the same scorer over the same prompts. They are recorded
as their own rounds so the diff shows proposal versus control directly.

## Tests

`tests/test_wsp_agent_round.py` runs offline on a tiny random Cohere model, CPU only. It covers the
dump-to-apply round trip, schema refusals (unsorted, duplicate, out-of-range, unknown tensor,
oversized fraction, too many submissions), budget carried across two invocations in one output
directory, refusal of validation/final datasets and transfer wordings, the workbench byte cap, and
that this document and the workbench README name every refusal.
