# WSP tools: API reference

`security_wsp.wsp_tools` is the tool layer for phase 4 of protocol v3: an LLM agent ("weight-space
programming") inspects a frozen Cohere/Aya Expanse checkpoint, writes sparse row/column weight edits,
and scores them against a small discovery database. It is pure library code. It does not train, take
gradients, load validation or final data, or launch GPU jobs.

```python
from security_wsp.wsp_tools import WspTools, Edit, BudgetExhausted

tools = WspTools(model, tokenizer, discovery_cases,
                 device="cuda:0",
                 site_patterns=["model.layers.*.mlp.down_proj.weight",
                                "model.layers.*.self_attn.o_proj.weight",
                                "model.layers.*.mlp.up_proj.weight",
                                "model.layers.*.mlp.gate_proj.weight"],
                 max_changed_fraction=0.0005,     # protocol-v3 wsp_agent.max_changed_fraction_of_weights
                 forward_pass_budget=50,          # protocol-v3 wsp_agent.forward_pass_budget
                 batch_size=16, seed=42,
                 output_dir=Path("runs/wsp-agent"))
```

## Constructor

| argument | meaning |
| --- | --- |
| `model` | Dense Cohere causal LM (`model_type` in `cohere`, `cohere2`), already on GPU or CPU. `attention_to_span` switches it to eager attention temporarily and restores the previous implementation. |
| `tokenizer` | Must have a chat template, and a pad or eos token (used as the left-pad id). |
| `cases` | Discovery-split `security_wsp.data.Case` objects only. |
| `device` | Device for inputs; the model is moved there and set to eval mode. |
| `site_patterns` | `fnmatch` patterns for editable tensor names; a name must also be a 2-D parameter. |
| `max_changed_fraction` | Upper bound on the fraction of **all** model parameters an edit set may touch. |
| `forward_pass_budget` | Units available to model forwards (see Budget accounting). |
| `batch_size` | Prompts per physical batch (protocol: at most 64 discovery prompts per unit). |
| `allowed_splits` | Defaults to `("wsp_discovery",)`. Must not contain `wsp_validation` or `wsp_final`. |
| `max_new_tokens` / `context_limit` | Generation length and the hard prompt limit; a prompt that would need truncation raises. The limit defaults to the smaller of `context_limit` and the model's `max_position_embeddings`. |
| `max_submissions` | Protocol's `max_submissions` (10). |
| `output_dir` | Required by `submit`. |
| `attention_batch_size` | Smaller physical batch for `attention_to_span`, because `output_attentions` returns every layer. |
| `permutations` | Sign-flip permutations for the contrast p-value (default 2000). |
| `top_k` | Columns reported by `list_sites` (default 5). |

The constructor raises `ValueError` for a non-Cohere model, a tokenizer without a chat template or pad/eos
token, an empty case list, a case from a forbidden or unlisted split, an empty `site_patterns`, or a
non-positive budget or batch size.

## Tools

Every tool method increments `calls` and appends exactly one transcript entry, including calls that
raise (status `error`).

### `list_sites(tensor_name_filter=None, top_k=None) -> list[dict]`
Editable tensors whose name matches `site_patterns` and contains `tensor_name_filter` (substring). Each
entry has `tensor_name`, `shape`, `dtype`, `total_scalars`, `stats` (mean, std, min, max in float32) and
`top_columns` — the `top_k` columns with the largest 2-norm, ranked descending. Costs 0 units.

### `read_weights(tensor_name, axis, indices=None, *, max_scalars=4096) -> dict`
Exact stored values for whole rows (`axis=0`) or whole columns (`axis=1`). `indices=None` means every
slice along the axis. Above `max_scalars` the reply carries `values_returned: false` plus shape and
statistics instead of values, so a large tensor can never flood the agent's context. Values are returned
as `slices` (slice-major: `len(indices)` slices of `slice_length`) in the tensor dtype cast to float.
Raises `PermissionError` for any name outside the editable sites and `ValueError` for out-of-range
indices. Costs 0 units.

### `activation_contrast(site, layer_index) -> dict`
For every discovery case, runs the neutral and `incorrect_suggestion` prompts (`prompts.render`,
`prefill=True`) and captures the activation at the answer position (last prefix token) at
`model.layers[i]` (`site="resid_post"`, decoder layer output), `model.layers[i].mlp.down_proj`
(`"mlp_out"`) or `model.layers[i].self_attn.o_proj` (`"attn_out"`). The `hook_` prefix is accepted.

Reports the mean incorrect-minus-neutral difference direction (unit length), its norm, per-case norms,
cross-case cosine similarity (to the mean direction and pairwise), and a separation statistic:
`||mean contrast|| / RMS per-case norm`, in [0, 1] — near 1 when cases agree, near 0 when signs scatter.
The p-value comes from seeded independent sign flips of the paired contrasts. A random-direction baseline
is reported for comparison: `|mean projection| / RMS projection` over 32 seeded unit directions. If the
mean contrast is zero the report says `degenerate: true` rather than inventing a direction.

### `attention_to_span(layer, head=None) -> list[dict]`
Attention weights from the final prefix position to tokens in the reviewer-opinion span, the source span
and everything else, one entry per head (or the single `head`), averaged over discovery cases, comparing
the neutral and misleading prompts. The opinion sentence is the one `Reviewer assessment` line of the
user prompt; the source span is the case code. Both are tokenized separately and matched as an exact
token subsequence; a case is excluded unless every match is **unique**, and exclusions are counted in
`match_failures` with reasons (`no match in prefix`, `N matches in prefix`). If no case is usable the
entries carry `status: "no_usable_cases"`, which is a reported failure, not an exception.

### `evaluate_edits(edits, *, limit_prompts=None, strong=False, strong_variant="trace", wording="reviewer") -> dict`
`edits` is one `Edit` set or a list of edit sets (each a list of `Edit`), applied one at a time.
Each set is validated, written in place under `torch.no_grad`, scored with batched greedy generation over
all discovery cases × the three conditions (or the first `limit_prompts` prompts in that order), then
restored. Results per set: `misleading_balanced_accuracy`, `neutral_balanced_accuracy`,
`correct_suggestion_balanced_accuracy`, `balanced_accuracy` (per condition), `per_class_counts`
(true negative/positive, false negative/positive, invalid count), touched and changed scalars with their
fractions, `restoration` hashes, and the raw `rows`.

Metrics come from `statistics.metrics`, so invalid JSON and abstentions count as wrong and balanced
accuracy is the mean of the vulnerable and clean per-class rates.

### `propose_random_control(patch, seed=42) -> dict`
Same tensors, same axis and same number of changed slices, but random slice indices drawn from a seeded
permutation and values drawn from a seeded Gaussian, rescaled so the total update norm matches the
target. The norm is measured after casting to the tensor dtype and the relative mismatch is reported with
a `within_1e-3` flag. Accepts a `patches.Patch` or an `Edit` set; the returned `edits` can be passed
straight to `evaluate_edits` or `submit`.

### `submit(edits) -> dict`
Validates one edit set against `max_changed_fraction` and `max_submissions`, then persists it with
`patches.build_patch` / `patches.save_patch` as `<output_dir>/submission-NN.{safetensors,json}`. The
model is never modified. The manifest records the submission index, the edit digest, forward units used
and the tool-call count. Submissions already on disk count toward the limit, so a restarted process
cannot exceed it.

### `transcript() -> list[dict]` and `save_transcript(path)`
Ordered log of every tool call: `seq`, `tool`, `arguments`, `status` (`ok`/`error`), `error`, `output`,
`forward_units`, `model_prompt_tokens`, and `estimated_tokens` (input/output, roughly four characters per
token). `save_transcript` writes the log with the budget state. Neither is itself logged.

## Budget accounting

- One unit per **physical batch** sent through the model: one greedy `generate` for up to
  `batch_size` prompts, or one forward pass. `activation_contrast` and `attention_to_span` charge the
  exact number of batches they run.
- `evaluate_edits` charges `batches × number of edit sets`. Editing itself is free; only scoring costs.
- `_prepare` (tokenization and the context-limit check) runs before the reservation, so a refused prompt
  costs nothing.
- The reservation happens **before any weight is written**. An exhausted budget raises `BudgetExhausted`
  with the model untouched, and the call is still logged as an error.
- `list_sites`, `read_weights`, `propose_random_control`, `submit`, `transcript` and `save_transcript` are
  free.
- The implementation charges per physical batch, which is stricter than protocol v3's "one
  try_edit/measurement call over at most 64 discovery prompts". At `batch_size=16` a 64-prompt call costs
  four units.

## Batching and determinism

Prompts are sorted by token length and grouped into batches of up to `batch_size`. Every batch is
left-padded with the pad token and carries an attention mask. Forward passes are given explicit
`position_ids = cumsum(mask) - 1`, so padded and unpadded rows see the same positions and differ only by
padding, and the answer position is the last column for every row. Greedy generation uses HF `generate`,
which derives positions from the mask. No prompt is ever truncated: a prompt that does not fit the
context limit raises before any model work. Batched arithmetic can differ from batch size 1 in the last
bits; the tests check that batched and sequential generation produce identical text on the tiny model.

## Safety invariants

1. **Discovery data only.** Cases whose split is `wsp_validation` or `wsp_final` are refused at
   construction, and `allowed_splits` may not contain them. The tools never read another dataset.
2. **Editable sites only.** Both reads and writes are gated on `site_patterns` and on the parameter being
   2-D. No gradients are taken, and no module or hook is installed permanently.
3. **No truncation.** Prompts over the context limit raise.
4. **Restoration is verified.** Each touched tensor is hashed before the edit and after the restore; a
   mismatch raises `RuntimeError`. The same hashes appear in the result, so a caller can check them.
5. **Bounded edits.** The touched-scalar fraction is checked against the whole model, one edit per tensor
   per set, unique in-range indices, finite values that survive the tensor dtype.
6. **Bounded submissions.** The count is checked against `max_submissions`, including files on disk.
7. **Everything is logged.** Every tool call and its outcome is in the transcript, with its unit cost.

## Tests

`tests/test_wsp_tools.py` builds a tiny random Cohere model and a word-level tokenizer from `tokenizers`
with a chat template, and runs on CPU with no network. It covers site listing and filtering, read caps and
permission errors, contrast finiteness, determinism, degenerate detection and p-value range, attention
span matching and reported failure, the ambiguous-match locator, exact apply/restore by tensor hash,
batched versus sequential generation, budget exhaustion without a write, fraction and context refusals,
random-control norm matching (including column patches), submission limits with a patch that applies and
reverts on a fresh copy, transcript ordering and costs, and refusal of validation and final splits.
