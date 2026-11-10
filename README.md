# Aawden

Aawden combines the Darija idea of repeating or trying again with *warden*, a guardian. The name reflects the project's cycle of testing, learning and strengthening security review.


### Evidence before authority.

Aawden is a research workbench for one concrete failure: a security-review model changes a correct assessment because a confident reviewer supplies the wrong opinion. It connects executable security fixtures, paired prompting, CircuitKIT instrumentation, reversible checkpoint editing, and AuditKIT evaluation.

**Current evidence.** Real Aya Expanse 8B completed 300 security reviews on Beam. The initial screen stopped the circuit-edit study after Aya fell below the required neutral accuracy. A separate competence study trained a real LoRA adapter that reached **90% strict neutral balanced accuracy on 20 fresh developer-authored cases**. This small check has a 75–100% family-bootstrap interval; independent final validation remains pending. The local dashboard uses a separate constructed toy. [Best result and reproduction](docs/BEST_RESULT.md) · [Verified experiment ledger](docs/AYA_PROGRESS.md) · [Product explanation](EXPLANATION.MD)

## Run it

Python 3.11 or newer is sufficient for the core application. No Node build, API key, external service, or GPU is required for the toy demo.

~~~powershell
python aawden.py serve
~~~

Open [the local evidence console](http://127.0.0.1:8765). Alternatively run **START_AAWDEN.ps1**. The console includes:

- Before/after metrics against the **stronger prompt** comparator.
- A searchable 50-case source browser with three reviewer conditions and executable oracle evidence.
- All targeted trials, seeded random controls, and the frozen validation choice.
- Raw results, downloadable reports, exact artifact hashes, and an integrity check.
- A button that actually reruns the toy pipeline in a new run directory.

To generate a new demonstration after source changes, use a fresh directory with **python aawden.py demo --output runs/aawden-demo-fresh**.

To enable genuine AuditKIT scoring without installing the full research stack:

~~~powershell
python scripts/fetch_toolkits.py
python aawden.py demo --output runs/audit-demo
~~~

The source downloader verifies immutable sponsor archive hashes. Without AuditKIT, the core demo reports its absence explicitly. It does not claim that the sponsor integration ran.

## What is implemented

| Component | Implementation and verification |
|---|---|
| Benchmark | 50 original, balanced vulnerable/fixed cases; in-memory SQLite and synthetic authorization fixtures |
| Oracle | Actual local query/policy violations plus legitimate access checks; all 50 verified |
| Leakage controls | Entire source families and shared exact templates stay in the same split and statistical cluster |
| Prompts | Neutral, correct opinion and misleading opinion; validation-only choice of stronger wording and a separate authority test |
| Parsing | Strict JSON fields, category, finite confidence and valid source-line indices; duplicate keys rejected |
| Evaluation | Per-answer checkpoints, raw outputs, exact prompts, parse errors, resumable candidate trials, and paired comparisons |
| Statistics | Balanced accuracy, FPR, FNR, miss rate, coverage, flip denominators, paired family bootstrap intervals |
| CircuitKIT | Checkpoint comparison, residual contrasts, and causal patches to attention and MLP outputs. The real Aya BF16 comparison missed the preset numerical tolerances. |
| Edits | Projection of selected HF residual-output matrices; actual stored-value counts, hashes, rollback and safetensors verification |
| Search | At most ten edited recipes; validation-only selection; random directions matched by update norm with mismatches disclosed |
| Freeze | Fresh-process behavior check and final-file hash commitment before final evaluation |
| Regression checks | Committed exact-answer Python questions and a blinded review of whether explanations match cited code |
| Reporting | Measured Markdown report, source/lock hashes, artifact manifest and explicit missing evidence |
| Dashboard | Loopback-only application; no uploaded-code execution or external model calls |

The toy fixture is an instrumentation exercise. Its known lexical feature and deliberately planted opinion weight are hand constructed. It is not a trained security model, an LLM, or a reproduction of a published mechanistic result.

Read [EXPLANATION.MD](EXPLANATION.MD) for the product idea and pipeline.

The overview includes the recorded Aya result separately from the toy circuit demonstration. Verify the saved 90% result without inference with `python aawden.py audit-adapter --output docs/AYA_VERIFICATION.json`. Prepare independent label and explanation review packets with `python aawden.py adapter-review-packet --output runs/review-handoff`. The forms start unreviewed and require a separate reviewer. See [MODEL_CARD.md](MODEL_CARD.md) for the selected adapter, intended use, limitations and model licensing.

Build the real Aya research bundle locally with `python aawden.py package --run runs/aya-beam-20261107/adapter-v7/study/confirmation --output dist/aawden-research.zip`. It includes saved evaluation evidence and supporting attempt outcomes, excludes model weights and credentials, and does not publish or submit anything.

The competence branch includes strict adapter loading, validation-only prompt selection, model-only return estimates, prospective checks and a credit guard protecting the user's $15 Beam balance. Its selected adapter is saved at `checkpoints/aya-adapter-v7.safetensors`; the earlier v6 file is retained at `checkpoints/aya-adapter.safetensors`. `configs/model-adapted.json` verifies the selected hash and exact base shards. `configs/selected-review.json` records its trace prompt and measured scope. Reproduce the selected evaluation on compatible hardware with `python scripts/evaluate_selected_review.py --output runs/selected-review-reproduction`. Recompute the existing ledger without GPU calls using `python scripts/report_aya_progress.py`.

## Reproduce the benchmark and checks

~~~powershell
python aawden.py build-data
python aawden.py validate-data
python aawden.py build-data --exploratory-splits --output datasets/exploratory.json
python -m pytest -q
python aawden.py verify --run runs/aawden-demo
python aawden.py package --run runs/aawden-demo
~~~

In a restricted Windows sandbox, place test temporary files inside the checkout:

~~~powershell
python -m pytest -q --basetemp .cache/pytest-local
~~~

The bundled exploratory split is exposed development material. It is **not** a protected final set. Exact-template grouping is conservative; it does not prove that different families are statistically independent or semantically unrelated. Labels were authored and reviewed by the same coding agent; independent review is still required.

## Run the actual Cohere study

Use an isolated Python 3.11 environment with enough memory for the reference and instrumented models. The resolved research lock targets **Linux**, not this Windows CPU environment.

~~~bash
uv venv .venv-research --python 3.11
uv pip sync --python .venv-research/bin/python requirements-research.lock
source .venv-research/bin/activate
python aawden.py doctor
~~~

The Beam worker installed the pinned sponsor tools and PyTorch with CUDA 12.6. The Beam evidence ZIP contains the exact host-specific lock. The core application has no third-party runtime dependencies.

Accept the model license through Hugging Face and provide authorized access to the pinned checkpoint in **configs/model.json**. Use a scoped read token through the Hub credential store or a job secret. Keep local `.env` credentials out of version control and job uploads.

On the provisioned host, the staged runner is the simplest entry point:

~~~bash
python scripts/run_aya_pilot.py --preflight-only --output runs/aya-pilot
python scripts/run_aya_pilot.py --output runs/aya-pilot
~~~

See [the GPU runbook](docs/AYA_RUNBOOK.md) for the resource request, workload, and stop decisions. The preflight checks access without downloading weights. To inspect individual steps:

~~~bash
python aawden.py parity
python aawden.py baseline
python aawden.py select-prompt
python aawden.py discover
python aawden.py evaluate-candidates --protocol runs/prompt-selection/protocol.json
python aawden.py verify-reload
~~~

The workflow stops when:

- The exact reference and CircuitKIT model fail the preset numerical parity checks.
- Neutral competence is below the screen threshold or misleading suggestions do not create the declared failure.
- Stronger prompting already removes the practical room for the target improvement.
- Candidate mechanisms do not outperform a causal random-direction control.
- No candidate improves the strong-prompt baseline under validation guardrails.

These are useful negative results. Changing the scope, margins or search budget after seeing them requires a new protocol version.

Candidate validation reuses one model in memory, restores original tensors after every trial, and exports only the selected checkpoint. This avoids retaining multiple 8B checkpoint copies. Full JSON generations are evaluated separately from compact-choice discovery scores. All continuation tokens are scored; both label-to-letter mappings are used.

## Final evaluator handoff

Use newly sourced, independently checked families held outside the editing workspace. The evaluator supplies SHA-256 commitments for the final security cases, independent label review and ordinary Python questions before opening the files. The security dataset uses the same schema as the screen, with split set to `final_test`. See [the evaluator handoff](docs/FINAL_EVALUATOR.md) for the label and explanation review formats.

~~~bash
python aawden.py freeze --search runs/candidates --final-dataset-sha256 DATASET_HASH --final-review-sha256 REVIEW_HASH --coding-dataset-sha256 CODING_HASH --exposure-note "Describe actual evaluator separation and any exposure"
python aawden.py final-evaluate --freeze-manifest runs/candidates/frozen.json --dataset /evaluator/final.json --review /evaluator/review.json --coding-dataset /evaluator/coding.json --output runs/final
~~~

The CLI checks the committed file hash, frozen source/prompt/checkpoint identities and family/source disjointness. It rejects refreezing and repeated completed final looks in the same run. These software checks do not replace genuine evaluator separation.

Check neutral performance against its interval margin. Report the separate coding score and blinded explanation review when those measurements are available. The report flags missing evidence.

## Submission

The [official hackathon page](https://hackathon.lexsi.ai/) lists dense Cohere models, at least two tools from the selected track, multiple before/after metrics, public artifacts on Hugging Face, and a public project/write-up. The proposed track is **Track 2: AI Red Teaming, Safety and Alignment**, particularly adversarial evaluation and circuit-level intervention.

**docs/SUBMISSION_DRAFT.md** and **docs/DEMO_SCRIPT.md** provide a concrete pitch and presentation sequence. The local package command prepares an allowlisted bundle; it does not upload, publish, accept model terms, register, or submit anything.

Actual two-tool research execution, real model measurements, independent review, public HF collections, and the final write-up remain necessary before representing the project as competition-ready. Nothing can guarantee a hackathon win.

## Layout

~~~text
aawden.py                       Checkout launcher
configs/                       Pinned model and frozen protocol defaults
datasets/                      Generated development benchmark records
fixtures/generated/            Executable original source and oracle results
src/security_wsp/
  data.py, prompts.py           Dataset and model-visible contracts
  models.py, parity.py          Local HF inference and CircuitKIT equivalence
  discovery.py                 Contrastive signals and causal controls
  interventions.py             Projection edits and candidate ledger
  evaluation.py, statistics.py Scoring and paired uncertainty
  regressions.py                Coding scores and blinded explanation review
  artifacts.py, workflow.py     Resume, integrity, selection and freeze
  sponsors.py                  Actual AuditKIT bridge
  web/, server.py              Local evidence console
tests/                         Behavioral, contract and tensor checks
notebooks/                     GPU-host research handoff
docs/                          Protocol, demo and submission material
runs/                          Local experiment artifacts, ignored by Git
~~~

Original project code and fixtures use MIT. Sponsor dependencies retain their own licenses, including LSAL-1.2 for the pinned CircuitKIT and AuditKIT sources. Model weights retain their model license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

