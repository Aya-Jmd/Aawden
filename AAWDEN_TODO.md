# Aawden project checklist

Updated 9 November 2026 against the [developer guide](RESSOURCES/CYBER_DEFENSE_WEIGHT_SPACE_DEVELOPER_GUIDE.md). A checked box records work completed at the stated scope. Prepared review material, completed software checks and measured model results are tracked separately.

The local demo runs end to end. The real Aya Expanse 8B screen completed all **300 answers** on Beam and stopped the original edit study because neutral competence was too low. The v7 LoRA adapter scored **90% on 20 neutral cases**. The four broader harder confirmation scores were **18/32**, **21/32**, **25/32** and **25/32**. V10 got **29/32** on the supported Python subset, with three false alarms and zero missed bugs. All 208 new examples passed their behavior checks. The last post-run Beam balance was **$21.04**. [Earlier result](docs/BEST_RESULT.md) · [Harder review](docs/RIGOROUS_REVIEW.md)

## Next actions

- [ ] Have an independent reviewer check the 190 development labels using the [label packet](dist/aawden-label-review.zip). Give them the packet without the labelled datasets or model answers. The response form is still empty.
- [ ] After label decisions are recorded, have an independent reviewer assess the 20 saved explanations using the separate [explanation packet](dist/aawden-explanation-review.zip). Validate the returned forms using the [evaluator handoff](docs/FINAL_EVALUATOR.md).
- [ ] Obtain new, protected security and coding cases from a separate evaluator. Commit their hashes and the independent review before opening the final files. The cases used for the 90% result are now exposed and cannot serve as the protected final set.
- [ ] Arrange an authorized GPU run for the independent final evaluation. Run original Aya and the selected adapter on the same final cases under neutral, correct and misleading advice. Measure coding regressions and review the cited explanations.
- [ ] Review the final submission contents and licensing, confirm registration and obtain the owner's publication approval. The [local research bundle](dist/aawden-research.zip) is built; it is neither published nor submission-ready.

The original causal-edit experiment remains stopped at its neutral-accuracy and CircuitKIT parity gates. Passing the separate adapter milestone does not reopen that branch. Its conditional steps remain unchecked below.

## Current offline deliverables

- [x] Use Aawden consistently in source, resources, protocol identifiers, exports, commands and model documentation.
- [x] Preserve adapter tensor bytes and measured generation payloads, re-export metadata with consistent checksums, and re-score the 90% result without inference.
- [x] Verify all 23 completed directory manifests and 18 completed manifests inside the evidence archives.
- [x] Verify 448 real adapter tensors against the pinned Aya architecture in a fresh CPU-only process. This is an architecture/file check, not behavioral equivalence.
- [x] Prepare label-blind packets for all 190 exposed development cases and an independent explanation-review packet for the selected 20-case result. Both handoff ZIPs passed integrity checks on 9 November.
- [x] Reject incomplete, mismatched and unblinded explanation reviews. No independent judgment is fabricated.
- [x] Add the recorded real Aya result to the dashboard, with a separate label and limitations from the toy demonstration.
- [x] Write the selected model card, update attribution and provide a secret-screened real-evidence packaging command.
- [x] Build the local real-Aya research bundle, including both scored stages, training protocol, attempt outcomes, source/configuration and per-file integrity inventory. Model weights and credentials are excluded.

## Additional competence work

- [x] Run the original Aya weights using the project-specific Hugging Face account and Beam serverless compute.
- [x] Preserve failed prompt and training attempts, including their scored outputs and stop decisions.
- [x] Train, save, recover and hash-check a real rank-16 adapter while preserving the original base weights.
- [x] Add neutral confirmation sets with executable oracles, recorded hashes and whole-family separation. Each scored set becomes exposed before further tuning.
- [x] Compare four review instructions on exposed validation cases, then freeze the selected instruction before the next check.
- [x] Record the selected trace pipeline's fresh score of 75%, with SQL and access-control scores disclosed separately.
- [x] Implement and locally check model-only return estimates as an inference experiment. Preserve auxiliary scores and generated final JSON separately.
- [x] Enforce the $18 Beam floor before launch, with a $1 margin and a conservative reserve for the whole job timeout. Run one job at a time.
- [x] Exceed 75% on a new precommitted neutral confirmation set. V7 scored 18/20, with no invalid JSON, after 85% validation. The family-bootstrap interval is 75–100%.
- [x] Download and hash-check the selected adapter, retain the verified reproduction source export, reparse the saved answers, and rerun all 20 confirmation oracles. [Saved audit](docs/AYA_VERIFICATION.json)
- [x] Provide a reproduction command using the selected model and trace prompt. Preserve previous checkpoints and all failed attempts.
- [ ] Establish the trained pipeline's response to misleading and correct opinions, coding regressions and explanation quality on independent data.
- [ ] Verify a merged adapted checkpoint after fresh-process reload if pursuing an artifact that runs without adapter wrappers.

## Harder neutral security review

- [x] Author 48 additional Python code examples in 24 matched clean/vulnerable families for SQLite injection and owner-only access.
- [x] Execute both legitimate and adversarial fixture checks. Verify family, source and exact syntax separation from the previous benchmark.
- [x] Freeze a 32-case confirmation set before inference and keep it separate from the 16 new development examples and 20 exposed regression examples. [Protocol](configs/rigorous-review.json)
- [x] Record false alarms, missed bugs, abstentions, invalid outputs and generation errors separately. Count all of them against strict accuracy.
- [x] Complete the first harder run. Aya got 18 of 32 right, with eight false alarms, six missed bugs and valid JSON on all cases. Beam reported $23.72 after the job. [Harder review evidence](docs/RIGOROUS_REVIEW.md)
- [x] Commit the dataset hash for another 32 new examples and add syntax-guided review with conditional adapter refinement. [Second protocol](configs/rigorous-review-v2.json)
- [x] Refine the adapter and verify the downloaded checkpoint. The 30/36 validation result fell below the 90% gate, so the confirmation set stayed unopened.
- [x] Add static return-path notes and validate selection of a prompt per bug category using saved v8 answers and new development predictions. [Third protocol](configs/rigorous-review-v3.json)
- [x] Run the second confirmation set and verify the saved answers. Aya got 21 of 32 right, with nine false alarms, two missed bugs and valid JSON on all cases.
- [x] Build another 32-case confirmation set and add symbolic SQL and owner-guard notes to address the observed errors. [Fourth protocol](configs/rigorous-review-v4.json)
- [x] Verify the v9 checkpoint and the 31/36 development score, with four false alarms and one missed bug. The third confirmation set stayed unopened.
- [x] Implement a second-pass check for access-control explanations and a single-pass SQL route. [Fifth protocol](configs/rigorous-review-v5.json)
- [x] Test the second pass. It improved access review from 11/18 to 12/18 on the same GPU and still fell below the development gate.
- [x] Compare symbolic paths with the single-pass baseline on the same GPU. The selected route scored 32/36 on development and 25/32 on the third confirmation set, with six false alarms and one missed bug. [Sixth protocol](configs/rigorous-review-v6.json)
- [x] Freeze a fourth confirmation set with 32 new examples and prepare a separate decoding calibration for access control. [Seventh protocol](configs/rigorous-review-v7.json)
- [x] Verify the calibrated result. Aya got 25/32 on the fourth set, with one false alarm and six missed bugs.
- [x] Implement a shorter view of supported Python code and freeze a separate 32-case test of that scope. [Eighth protocol](configs/rigorous-review-v8.json)
- [x] Compare the compact view with the full-source baseline. The scores were 39/48 and 38/48 on the same GPU. The fifth set stayed unopened.
- [x] Freeze a compact-view refinement on 64 exposed examples with separate development and confirmation. [Ninth protocol](configs/rigorous-review-v9.json)
- [x] Train v10 and verify 29/32 on the fifth confirmation set. Aya detected all 16 bugs and flagged three clean cases.
- [x] Freeze a second-opinion rule that scored 36/36 on saved development answers and reserve 32 further cases in the same supported Python scope. [Tenth protocol](configs/rigorous-review-v10.json)
- [ ] Confirm 100% strict verdict accuracy on the harder set using frozen Aya settings and a verified Beam balance of at least $18.

This competence branch follows the guide's suggested scope change when neutral performance is poor. Its results do not satisfy the original causal-edit or final-evaluation gates below.

## 1. Project setup and dashboard

- [x] Create the Python project, command-line tools, reproducible dependency locks, and pinned CircuitKIT and AuditKIT sources.
- [x] Define the experiment rules before measurement. Record the model, data splits, prompts, metrics, candidate limit, and stop conditions. [Protocol](docs/PROTOCOL.md)
- [x] Build the local dashboard with a before-and-after view, searchable code cases, reviewer opinions, executable evidence, all demo trials, and a rerun button.
- [x] Show raw outputs, offer report and ZIP downloads, check run integrity, and label the toy results. The saved demo contains 1,200 scored generations.
- [x] Write the project explanation, Aya runbook, evaluator handoff, demo script, and submission draft. Build local handoff and demo ZIPs. [Project explanation](EXPLANATION.MD)
- [x] Test the data, prompts, parser, statistics, tensor edits, checkpoint reload, and artifact checks. Test both sponsor tools on a tiny randomly initialized Cohere model. [Validation record](docs/VALIDATION.md)

## 2. Benchmark and scoring

- [x] Build 50 original examples with explicit threat models and expected behavior. Cover SQL injection, object ownership, and role checks, with 25 vulnerable and 25 fixed examples.
- [x] Write and run local tests that demonstrate the violation in vulnerable code and check that the fixed code still permits legitimate use. Record source hashes, provenance, and licenses.
- [x] Keep related code and vulnerable/fixed pairs in the same split. Check source, family, exact-code, and exact-template overlap. The current exploratory split has 30 discovery, 10 validation, and 10 exposed test cases.
- [x] Render the same code with no opinion, with a correct opinion, and with a misleading opinion. The prompt exposes only the opinion selected for that test condition and the information available to a code reviewer.
- [x] Provide a standard instruction and a stronger instruction that tells Aya to check the code independently. Version and hash the prompts.
- [x] Require a verdict, category, cited code lines, explanation, and self-reported confidence. Retain raw answers and errors. Count invalid answers and abstentions as incorrect in the main score.
- [x] Calculate balanced accuracy, false alarms, missed vulnerabilities, answer coverage, both versions of the opinion-induced flip rate, token counts, and latency. Calculate paired 95% intervals by resampling code families.
- [x] Integrate AuditKIT to score raw generations against the executable benchmark labels. The bridge has run on toy and architecture test outputs.
- [x] Prepare the original 50-case blinded packet for a second reviewer and the expanded 190-case handoff. [Expanded review packet](dist/aawden-label-review.zip)
- [ ] Have a second reviewer check the development labels and the tests behind them. The current development labels have executable evidence and a single author.
- [x] Implement selection between two prespecified stronger prompts using validation examples. Save the selected prompt with the edit protocol.
- [x] Implement a separate final comparison under fresh wording and a claimed senior reviewer.
- [x] Add an exact-answer Python regression check and a blinded review for whether explanations match cited code. Test both workflows with local fixtures.
- [x] Implement reporting for the paired edited/standard versus original/standard comparison, alongside the trial ledger, attempt outcomes and generation resource use. The end-to-end causal-edit comparison has been demonstrated on the toy only.

## 3. Aya code that is built

- [x] Pin `CohereLabs/aya-expanse-8b` to an exact Hugging Face revision and configure BF16. Use its tokenizer and chat template, fixed decoding, and a shared context limit. Reject overlong prompts and fail explicitly if the model cannot load.
- [x] Implement the Aya runner and a lightweight preflight for model access, GPU memory and BF16 support, disk space, and the research libraries. [Aya runbook](docs/AYA_RUNBOOK.md)
- [x] Implement the CircuitKIT compatibility check against standard Transformers inference. Compare next-token scores and choices on identical input tokens in batches of one and two. Compare complete answer scores as well.
- [x] Implement the unchanged checkpoint export and a fresh-process check that its predictions survive saving and reloading.
- [x] Implement the original Aya baseline and stop rules. Compare standard and stronger prompts and check whether misleading opinions cause a large enough drop to investigate.
- [x] Implement discovery for residual signals, attention outputs and MLP outputs. Compare the same case with and without a misleading opinion, align the intervention to the answer position, and score every answer token with both A/B letter mappings. Rank changes by their effect on the correct answer. Include no-op and random interventions.
- [x] Implement the proposed weight change. If discovery identifies a promising signal, remove a tested fraction of that direction from a selected attention or MLP output weight matrix. The search tests strengths of 0.5 and 1.0 while preserving the original checkpoint and architecture.
- [x] Limit search to ten distinct edits, with up to four targeted edits and three seeded random edits in the current pilot. Record the recipes, results, and update sizes. Select the highest misleading-opinion score on validation subject to neutral and correct-opinion checks, breaking ties with the smaller weight change. Use similar-sized random edits and an unchanged run as controls.
- [x] Implement export to a separate safetensors checkpoint and a check that reloaded predictions match the in-memory edit without hooks. Record tensor hashes, precision, touched and actually changed values, and update norms. Restore the original weights after each trial.
- [x] Implement per-answer recovery for baseline and final scoring, run identity checks, atomic records, and artifact verification. Write the command that freezes the chosen checkpoint and checks the evaluator's committed files before scoring them.
- [x] Resume interrupted candidate evaluations from verified saved answers. Preserve attempt outcomes, limit stage retries, and publish a checkpoint directory atomically after its tensors pass verification.
- [x] Add exact base weight file hashes, rendered prompts, and loaded runtime identity to the evidence. Bind resumed runs to the same loaded checkpoint.

## 4. Run the experiment on real Aya

- [x] Install the pinned research environment on a Beam Linux RTX 4090 worker. The key has serverless credit. Beam declined the A100 reservation because the workspace had zero credits eligible for managed compute.
- [x] Verify the project Hugging Face token can read the pinned Aya config and access all four weight shards. [Local preflight](runs/aya-env-preflight-20261107/preflight.json)
- [x] Check the real Aya tokenizer and chat template. All 450 prompt variants fit the context budget and all 780 continuation checks passed. [Tokenizer check](runs/aya-beam-20261107/tokenizer-check.json)
- [x] Pass the Hugging Face credential to the Beam worker and verify the config and pinned weights load in BF16. The run manifest records hashes for the shard index and all four shards.
- [x] Run the real-weight comparison on the RTX 4090 with one model on the GPU at a time. The retry completed all 24 checks, with 11 passes. [Comparison record](runs/aya-beam-20261107/sequential-evidence/parity.json)
- [ ] Pass the numerical comparison and verify an unchanged checkpoint in a fresh process. The real run missed the prespecified numerical tolerances, so it stopped before the reload step.
- [x] Run the 50-case baseline with both prompts and all three opinion conditions. AuditKIT scored 300 answers. [Verified run report](runs/aya-beam-20261107/evidence/baseline/report.md)
- [x] Apply the stop rule. The strict screen failed. After removing the output fences in a separate diagnostic, the standard prompt scored 48% without advice and 0% with misleading advice. The required neutral score is 75%. [Diagnostic](runs/aya-beam-20261107/screen-format-diagnostic.json)
- [ ] If the screen passes, choose the stronger wording on actual Aya validation answers and save it for edit selection.
- [ ] If the screen passes, run causal tests on discovery cases. Check whether removing the suspected signal helps more than the random and no-op controls and whether it still helps with new wording. Record the outcome even if no mechanism is supported.
- [ ] If discovery supports an edit, run the bounded validation search on Aya. Check full generated answers, select using validation only, and keep the outcomes of every candidate. Stop with a documented result if no edit passes.
- [ ] If an edit qualifies, save the edited Aya checkpoint and verify on the GPU host that its predictions match the selected in-memory version after a fresh reload.

## 5. Final proof and hackathon handoff

- [ ] If the pilot supports proceeding, use measured effect, variation, family count, and throughput to size the expanded study. The guide's starting targets are 200 discovery, 100 validation, and 300 final cases.
- [ ] Have a separate evaluator prepare fresh code families, review near-duplicates, run the executable checks, and get an independent label review. Keep the cases, labels, and review outside edit selection. [Evaluator handoff](docs/FINAL_EVALUATOR.md)
- [ ] Get the final security dataset, coding dataset and label review hashes from the evaluator. Freeze the selected Aya checkpoint, prompts, decoding settings, and comparison plan before opening those files.
- [ ] Run the frozen original and edited Aya with both prompts on identical final cases. Include all three seeded random edits and the unchanged control. Audit the real outputs with AuditKIT and record any technical rerun.
- [ ] Check the main target on misleading advice. The edited model should beat original Aya with the same stronger prompt by at least 5 percentage points, with a paired 95% interval above zero. Check that the neutral and correct-opinion intervals meet the planned 2-point tolerance.
- [ ] Check false alarms, missed bugs, behavior under fresh wording, general coding scores, and the review of cited evidence. Report uncertainty and remaining limitations alongside gains.
- [x] Document the measured Aya feasibility result and why the edit study stopped. Preserve the baseline outputs and the post-run format diagnostic separately.
- [x] Finish the measured feasibility write-up and local reproducibility bundle, including saved raw Aya results, benchmark, oracle checks, attempt outcomes, protocol, source inventory, model card and resource record.
- [ ] Add the approved public code/model artifact revision and independently measured final comparison and regression results before calling the full submission complete.
- [ ] Review licensing, attribution, and contents for release. Confirm registration and obtain the owner's publication approval, then publish the permitted model artifacts, dataset and logs to Hugging Face, make the code/demo and write-up public, and submit.

Optional experiments from the guide include a SafeTune comparison, a separately budgeted LoRA comparison, and reproduction of the 2,046-input toy parity circuit.
