# Local validation record

## Current offline verification — 10 November 2026

- **165 tests passed, 2 skipped** in 2.02 seconds, measured on 2026-11-10 with `python -m pytest -q -p no:cacheprovider` in the torch venv. The skips are the AuditKIT source/install check (`tests/test_pipeline.py`) and a missing `huggingface_hub` (`tests/test_tensors.py`). Three torch-dependent tests in `tests/test_neutral_study.py` and `tests/test_offline_workflows.py` skip rather than fail when torch is absent (checked by shadowing `torch` with a stub that raises `ModuleNotFoundError`). The additional tests verify strict re-scoring of saved Aya answers, detection of changed prompts despite re-sealed metadata, checkpoint-byte checks, independent-review commitments, incomplete-form rejection, bundle inventories and refusal to package a private credential.
- All **190 executable development oracles** passed. Separate handoffs are prepared for 190 labels and 20 explanations; independent judgments remain pending.
- All **23 completed directory manifests** and **18 completed manifests inside evidence ZIPs** passed integrity verification. Canonical metadata exports have regenerated checksums and separate measurement receipts. **71 scored output sets** retained identical model-visible prompts, raw generations, labels, parsed decisions and correctness flags. These checks do not constitute new inference.
- The selected adapter remains byte-identical at SHA-256 `e86f9475e3c8871ba3c60a63b10704954f66edcb791bd9631866a07d82a9022f`. A fresh CPU process verified **448 FP32 tensors and 41,943,040 adapter parameters** against the pinned Aya architecture using meta-device base tensors. It loaded no base weights and verified no behavioral inference equivalence.
- The dashboard has a separately labelled recorded real-Aya panel. Source, resource documents, protocol names, launchers and current exports use Aawden. The current work used no Beam compute. The last verified balance remains the earlier **$24.00** snapshot.

The current audit is [AYA_VERIFICATION.json](AYA_VERIFICATION.json). The model architecture check is [AYA_ARCHITECTURE_CHECK.json](AYA_ARCHITECTURE_CHECK.json). See [BEST_RESULT.md](BEST_RESULT.md) for the 90% feasibility finding, uncertainty and remaining work.


## Latest competence checks — 7–8 November 2026

- **108 tests passed** in 48.70 seconds. Coverage includes adapter disabling/restoration after exceptions, saved adapter loading from JSON paths, full continuation scoring, option-letter counterbalancing, label-blind prompts, preservation of the fresh-data stop rule, rejection of changed data before model loading and the protected credit boundary.
- The actual pinned Aya tokenizer passed **40 additional continuation-boundary checks** for the new auxiliary A/B score prompts. These checks use only exposed validation cases and make no GPU calls.
- Three budget tests passed separately. They check the exact allowed cent boundary, rejection one cent below it, unknown GPU costs, excessive timeouts, attempts to weaken the floor and larger resource reservations.
- The next adapter refinement passed **300 local training-target token checks**, with a maximum of 442 tokens per example. Its ten validation families are absent from training.
- `python scripts/report_aya_progress.py` verified manifests, re-parsed answers and recomputed **17 completed scored arms**. V7 scored **85% validation and 90% on the next 20 fresh cases**, across ten families. SQL scored 10/10 and access control scored 8/10. The family-bootstrap interval is 75–100%. Training and validation scores remain separate in [the ledger](AYA_PROGRESS.md).
- The selected checkpoint passed its SHA-256 check and contains 448 finite FP32 tensors with 41,943,040 adapter parameters. Both scored stage manifests passed, covering 46 artifacts. The exact source ZIP matches the frozen Linux source fingerprint and training-script hash. All 20 confirmation oracles passed, and the saved prompts match the selected trace renderer. [Full evidence](BEST_RESULT.md)
- The execution-note v2 launch guard read **$24.65** available, reserved a conservative **$0.70** maximum bounded-job cost, and required a further **$1** above the protected **$15** balance. This is a prelaunch balance and cost reserve, not a final invoice.

The subsequent v7 launch read $24.46 and reserved $0.94 for its 20-minute cap. It completed in 17 minutes 23 seconds. The post-run balance was **$24.00**, with zero active Aawden tasks and all known temporary job secrets cleaned up. GPU iteration stopped after the fresh score met the target. The final reproduction utility passed syntax and command-help checks; no extra GPU run was spent on it.

The original screen and CircuitKIT failure records below remain unchanged. The LoRA and inference experiments are a separate competence branch.

Validated on 3–4 November 2026 in the supplied Windows workspace.

| Check | Outcome |
|---|---|
| Python suite | **82 passed**, 136.09 seconds; no skipped tests in this run |
| Vulnerability oracles | **50 / 50 passed**, balanced vulnerable/fixed examples |
| Grouped source families | **24**, after merging shared exact templates |
| Full demonstration | **Completed**; 1,200 constructed-toy generations across principal methods, no-op and three random controls |
| Sponsor scoring | **Actual AuditKIT execution**, 1,200 precomputed raw responses with the explicit maximize-direction strict scorer |
| Checkpoint reload | **Passed** in a fresh Python interpreter; 12 generated responses matched, without runtime hooks |
| Source preservation | Toy source weights preserved; small genuine PyTorch export/failure-rollback tests passed |
| Serialization | Small genuine safetensors tensor-byte checks passed; changed-entry accounting verified |
| Statistics | Paired cluster calculations tested; final demo uses 10,000 bootstrap replicates |
| Resumption/integrity | Configuration mismatch, artifact tampering and corrupt checkpoint rejection tested |
| Frontend | JavaScript syntax check passed; no captured browser warnings/errors |
| Browser flow | Overview, category/search filters, experiment ledger, integrity action, Markdown download and ZIP download checked; dashboard-triggered rerun completed all 1,200 predictions |
| Downloaded ZIP | 12 evidence files; ZIP integrity check found no corrupt member |
| Notebook | Valid nbformat 4 JSON, unexecuted research handoff cells |
| Research dependencies | Linux/Python 3.11 resolver succeeded for 169 pinned packages; full installation and compatibility unverified |

The local shell's network sandbox initially prevented browser access to its server. Browser verification used an authorized process listening only on 127.0.0.1 outside that network sandbox.

The installed PyTorch is 2.9.1+cpu, installed Transformers is 4.46.3, and TransformerLens is absent. The full pinned research stack has not been installed into that global environment. The gated Aya checkpoint is not cached.

The 3–4 November checks covered the local software and toy demonstration. The 7 November GPU checks below cover the original Aya baseline and CircuitKIT comparison.

Toy improvements are deliberately planted instrumentation outcomes. They cannot establish a real-model improvement or competition eligibility.

## Research handoff checks — 4 November 2026

- **89 tests passed**, including seven final-review tests that cover label blinding, exact commitments, evidence agreement, and rejection before loading a model.
- An isolated Windows CPU environment executed the pinned CircuitKIT and AuditKIT sources with Transformers 5.22.1, TransformerLens 3.8.0, and Hub 1.33.0.
- A tiny randomly initialized Cohere architecture passed 16 reference-to-CircuitKIT numerical checks. Residual and MLP hooks, counterbalanced continuation scoring, and causal controls executed.
- Both the unchanged checkpoint and a projected synthetic checkpoint passed fresh-process reload checks. The edited result matched predictions made with the temporary in-memory edit. AuditKIT scored six generated synthetic outputs. The full smoke record is in runs/architecture-smoke-2/smoke-summary.json.
- The pinned Aya preflight read the saved Hub login without displaying the credential, and the Hub returned **HTTP 403** for the model configuration. The local CPU-only PyTorch build also failed the CUDA readiness check. No Aya weights were loaded.

The smoke run checked the interfaces and serialization paths. The 7 November GPU evidence appears below.

## Offline protocol v2 checks — 7 November 2026

- **92 tests passed**. The new tests cover validation-only prompt choice, recovery of an interrupted candidate evaluation, binding the coding dataset before scoring, separate authority scores, and a blinded explanation review tied to the frozen results.
- The full toy run scored **1,200 answers** through AuditKIT. Its report includes paired comparisons for both prompts and the generation resource table. See `runs/offline-v2-demo/report.md`.
- The tiny Cohere CPU model passed **16 parity checks**, ran both attention and MLP interventions, and passed unchanged and edited checkpoint reload checks. AuditKIT scored six outputs from the smoke run. See `runs/architecture-offline-final-v2/smoke-summary.json`.
- The 50-case development review packet was generated without reference verdicts. The coding examples passed schema validation. Independent review and a new final coding dataset are pending.

## Real Aya baseline on Beam — 7 November 2026

- The Aya Hugging Face token authorized the pinned checkpoint, and a Beam serverless RTX 4090 loaded the four BF16 weight shards. The installed runtime used Python 3.11, PyTorch 2.9.1 with CUDA 12.6, Transformers 5.22.1, and the pinned AuditKIT and CircuitKIT revisions. [Run manifest](../runs/aya-beam-20261107/evidence/baseline/manifest.json)
- Aya completed **300 answers** in **896.85 seconds of generation time**. AuditKIT scored 300 responses. The integrity check verified 305 artifacts, including all 300 predictions. [Original report](../runs/aya-beam-20261107/evidence/baseline/report.md)
- The first GPU attempt reached the weights and exposed an incorrect cache requirement. The loader now hashes the pinned shard files and index directly. The regression check covers a cache with the checkpoint files and no repository README. The workspace test suite passed **93 tests**, and the sequential CircuitKIT path passed 16 checks and both checkpoint reloads on the local synthetic architecture. [Sequential smoke record](../runs/architecture-sequential-slim-smoke/smoke-summary.json)
- The strict scorer accepted 3 of 300 answers. The other 297 had an outer Markdown JSON fence. The screen failed and did not authorize an edit search.
- The post-run diagnostic removed only a whole-response JSON fence and reapplied the original parser. Of 300 answers, 268 were valid after unwrapping. With the standard prompt, the neutral-to-misleading drop was **48 percentage points** with a paired family-bootstrap interval of **43.8 to 50.0 points**. The stronger prompt's drop was **30 points** with an interval of **18.0 to 40.4 points**. Both prompts scored **48%** without advice after unwrapping. This exploratory diagnostic uses the same development answers, and the prespecified 75% neutral threshold remains unmet. [Diagnostic record](../runs/aya-beam-20261107/screen-format-diagnostic.json)

To reproduce the diagnostic from the sealed run directory, execute `python scripts/diagnose_screen_format.py --run runs/aya-beam-20261107/evidence/baseline`. It prints the result to standard output and leaves the original files intact.

## CircuitKIT comparison with real Aya — 7 November 2026

The first sequential check ran out of memory while computing unused per-head attention outputs. The retry disabled those extra hooks and preserved the combined attention, MLP, and residual output hooks. The retry used the exact shard hashes from the baseline and completed all 24 numerical checks on a single RTX 4090.

The top token matched in all 12 next-token checks. The largest absolute log-probability difference was **0.252** against a limit of **0.05**, and the highest KL divergence was **0.00670** against a limit of **0.001**. Eleven of 12 continuation checks passed. Overall, 11 of 24 checks passed, and the workflow stopped before exporting the original checkpoint for a fresh-process comparison. [Comparison record](../runs/aya-beam-20261107/sequential-evidence/parity.json)

Beam reported **38 minutes 53 seconds** of total task runtime across four completed RTX 4090 jobs. The posted GPU, CPU, and RAM rates put that at approximately **$1.46** in compute. Beam's credit check reported **$28.76 available** after the jobs ended. The A100 request was declined before a reservation started. [Beam task status](../runs/aya-beam-20261107/beam-task-status.json) · [Credit check](../runs/aya-beam-20261107/beam-credit-final.json) · [Beam rates](https://www.beam.cloud/pricing#serverless)
