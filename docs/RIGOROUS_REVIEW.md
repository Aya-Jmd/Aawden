# Harder Aya security review

The new benchmark tests Python SQLite injection and owner-only record access. Each family contains a vulnerable snippet and a close clean counterpart. The differences include reassigned SQL fragments, quotes escaped and then restored, branch-specific binding, late policy overrides, early returns and checking the wrong record.

## Evaluation

1. Local fixtures exercise legitimate use, adversarial input and the clean counterpart. SQL checks include both boolean injection and an injected UNION row. Access checks include two legitimate owners, non-owners in the same and different tenants and an administrator with no ownership exception.
2. Aya sees the source, line numbers, target weakness and threat model. The fixture verdicts and probe results stay with the scorer.
3. The three prompts are compared on 16 new development examples and 20 exposed regression examples. The selected prompt and the hash of the fixed 32-case confirmation file are recorded before the confirmation run.
4. The original strict parser scores Aya's generated JSON. The report separates false alarms, missed bugs, abstentions, invalid outputs and generation errors. Every incorrect or missing verdict lowers accuracy.

The [protocol](../configs/rigorous-review.json) binds the data and adapter hashes. The examples are synthetic and have executable label checks from a single author. These measurements cover two vulnerability classes under neutral review prompts.

## First result

All three prompts scored 28 of 36 on development and regression examples. The declared tie-break selected the existing trace prompt. Aya then scored **18 of 32 (56.25%)** on the harder confirmation set. The offline verification rechecked 35 hashed artifacts, all 32 raw answers and the fixture labels.

| Weakness | Correct | False alarms | Missed bugs |
|---|---:|---:|---:|
| SQLite injection | 10/16 | 5 | 1 |
| Owner-only access | 8/16 | 3 | 5 |
| Total | 18/32 | 8 | 6 |

All 32 answers were valid JSON with decisive verdicts. Aya flagged bound SQL when it appeared near unused interpolated queries. It also missed an access bypass where the guard checked a key belonging to the caller while the return used the requested key. [Raw answers](../runs/aya-rigorous/rigorous-02/study/confirmation/results.json) · [Error report](../runs/aya-rigorous/rigorous-02/study/outcome.json)

Beam reported $23.97 before the run and $23.72 after it. The launch reserved up to $1.17 at conservative rates. The initial upload attempt failed before inference because of a Windows console encoding error. The retry used UTF-8 and both temporary secrets were removed.

## Refinement

The [second protocol](../configs/rigorous-review-v2.json) keeps a separate set of 32 new examples. It compares two prompts that highlight branch conditions and SQL arguments from Python's syntax tree. The second version starts the JSON explanation field before Aya generates its answer. Neither prompt receives execution results or the oracle verdict.

If the existing adapter falls below 90% on development and regression, the same GPU job trains its LoRA layers for two short epochs. Training uses the 32 now-exposed examples and 20 older replay examples. The 36 development and regression cases stay outside training. Only a candidate that clears the development gate reaches the new confirmation set.

The best v7 prompt scored 28/36. The refined v8 adapter scored **30/36 (83.33%)** with three false alarms, three missed bugs and valid JSON on every example. The 90% development gate kept the 32 confirmation cases unopened. The recovered v8 adapter matches the recorded SHA-256 hash. Beam reported $23.70 at launch and $23.39 after completion. [Development results](../runs/aya-rigorous/rigorous-03/study/outcome.json) · [Training receipt](../runs/aya-rigorous/rigorous-03/study/training.json)

The [third protocol](../configs/rigorous-review-v3.json) tests v8 with static paths that substitute earlier assignments. The paths show how a guard or query is constructed and leave external values unknown. The protocol reuses the saved v8 development answers and selects a prompt separately for SQLite injection and access control. The original 32-case follow-up remains the confirmation set.

The validation rule selected the explanation-first syntax prompt for access control and the static return-path prompt for SQLite. Aya got **21 of 32 (65.63%)** on the second confirmation set. Seven of the nine false alarms came from access control, including clean code that required ownership in every reachable return. The exact prompt and raw output for each case passed offline verification. [Raw answers](../runs/aya-rigorous/rigorous-05/study/confirmation/results.json) · [Error report](../runs/aya-rigorous/rigorous-05/study/outcome.json)

| Weakness | Correct | False alarms | Missed bugs |
|---|---:|---:|---:|
| SQLite injection | 13/16 | 2 | 1 |
| Owner-only access | 8/16 | 7 | 1 |
| Total | 21/32 | 9 | 2 |

A short trial of a persistent cache mount stopped during dependency setup. The retry used the working runtime configuration. Neither setup attempted more than one GPU job at a time.

The [fourth protocol](../configs/rigorous-review-v4.json) adds source-derived symbolic SQL expressions and simplifies recognizable guards under an explicit non-owner assumption. It tests those prompts before spending compute on a second weight refinement. The next 32 examples were written, locally executed and frozen before the second confirmation scores came back. Aya receives the original source and the symbolic expressions, while the execution tests and expected labels stay separate.

The v9 refinement scored **31 of 36 (86.11%)** on development. It got all 18 SQL examples and 13 of 18 access-control examples correct, with four false alarms, one missed bug and valid JSON for every case. The development gate required 32 correct, so the third confirmation set stayed unopened. Beam assigned an A10G to this run. The checkpoint was recovered and hash-checked, and the live balance was $22.72 before the next launch. [Training receipt](../runs/aya-rigorous/rigorous-06/study/training.json) · [Development results](../runs/aya-rigorous/rigorous-06/study/outcome.json)

The [fifth protocol](../configs/rigorous-review-v5.json) compares v9's saved one-pass answers against a consistency check for access control. It asks Aya to check its draft explanation against the source before writing a fresh verdict. Only the unverified explanation passes between the two model calls. SQL stays on one pass. The same 32-case third confirmation set remains reserved.

On the RTX 4090, the draft pass got 11/18 access cases right and the second pass got 12/18. The second pass caught every vulnerable access example and raised six false alarms. SQL scored 18/18. The per-category rule retained the earlier single-pass access prompt and reached 31/36 overall, one short of the gate. The earlier access measurement ran on an A10G, so the same-device improvement is one answer. Beam reported $22.56 before the next job. [Development and routing record](../runs/aya-rigorous/rigorous-07/study/outcome.json)

The [sixth protocol](../configs/rigorous-review-v6.json) compares both prompts on the same GPU. It derives expressions under explicit assumptions such as a logged-in non-owner in the same tenant, or a username containing an apostrophe. It presents those expressions alongside the original code. All predicted labels still come from Aya.

On the same RTX 4090, the scenario prompt scored 31/36 and the baseline scored 29/36. Selecting by category reached 32/36, which opened the third confirmation set. Aya got **25/32 (78.13%)**, with six false alarms, one missed bug and valid JSON on all 32 cases. The confirmed route used the policy prompt for SQL and the scenario prompt for access control. Beam reported **$22.35** after the task. [Raw answers](../runs/aya-rigorous/rigorous-08/study/confirmation/results.json) · [Error report](../runs/aya-rigorous/rigorous-08/study/outcome.json)

The [seventh protocol](../configs/rigorous-review-v7.json) searches a small grid of access-control decoding biases using 48 exposed examples, including the newly scored third set. The bias applies to Aya's first verdict token after a fixed JSON prefix. Aya generates the verdict and the rest of the answer. SQLite continues with its existing prompt. The fourth confirmation set contains another 32 code examples and is separate from calibration.

The first calibration worker lost its client connection while downloading dependencies. It was cancelled before inference, its temporary secret was removed, and Beam reported $21.86. The retry moved dependency installation into image build, before starting a GPU worker. The first image build stopped because its temporary lockfile did not survive across build steps. The corrected build uses a persistent path within the image.

The successful retry selected a −1.5 logit adjustment for the opening vulnerable token and scored 40/48 on development. On the fourth confirmation set, Aya again got **25/32 (78.13%)**. It raised one false alarm and missed six bugs, all in access control. SQL got 15/16 and access control got 10/16. Every answer was valid JSON. The adjustment traded fewer false alarms for more missed bugs on this separate set, with no higher score. Beam reported **$21.58** after completion. [Frozen decoding settings](../runs/aya-rigorous/rigorous-11/study/frozen.json) · [Raw answers](../runs/aya-rigorous/rigorous-11/study/confirmation/results.json) · [Error report](../runs/aya-rigorous/rigorous-11/study/outcome.json)

The [eighth protocol](../configs/rigorous-review-v8.json) compares the full-source review with a shorter code representation. For supported Python patterns, the shorter view shows return paths and SQL expressions with the original line numbers. Development examples outside that subset use the full source. Its separate 32-case confirmation set is scoped to a single function with local assignments, conditional branches and returns. This measures review on the supported source fragment alongside the broader earlier results.

The compact view scored 39/48 and the full-source baseline scored 38/48 on the same A10G. Neither cleared the 40/48 gate, so the fifth set stayed untouched. Beam reported **$21.40**. [Development results](../runs/aya-rigorous/rigorous-12/study/outcome.json)

The [ninth protocol](../configs/rigorous-review-v9.json) refines v9 on the 64 already scored examples from the third and fourth sets. It teaches both the compact and scenario views, with specific explanations for the access-control behavior. The separate 36 development and regression cases remain outside training. The same fifth set stays reserved for confirmation.

The v10 adapter scored **34/36** with the full-source policy prompt and **33/36** with the compact prompt. The policy prompt won selection in both categories. On the fifth set, Aya got **29/32 (90.63%)** within the supported Python subset. It caught all 16 vulnerabilities, raised three false alarms and returned valid JSON on every case. Two false alarms concerned access control. The SQL false alarm was a lookup that interpolated only alphabetic names and bound other names. Beam reported **$21.04** after the run. [Verified model config](../configs/model-rigorous-v10.json) · [Raw answers](../runs/aya-rigorous/rigorous-13/study/confirmation/results.json) · [Error report](../runs/aya-rigorous/rigorous-13/study/outcome.json)

On the same 36 development cases, the policy and compact prompts made different mistakes. The [tenth protocol](../configs/rigorous-review-v10.json) uses the original Aya policy answer unless it flags a vulnerability and a second Aya answer from the compact view finds the case clean. Both answers pass the same strict parser. The rule selected 36 correct saved answers on development and was frozen before scoring another 32 examples in the same supported Python scope.

## Reproduction

From the project root, in the pinned Aya research environment with access to the base model and the saved v7 adapter:

```powershell
python scripts/evaluate_rigorous_review.py --output runs/my-rigorous-review
```

Verify the confirmation artifacts and reparse the raw answers locally:

```powershell
python scripts/evaluate_rigorous_review.py --output runs/my-rigorous-review --verify
```
