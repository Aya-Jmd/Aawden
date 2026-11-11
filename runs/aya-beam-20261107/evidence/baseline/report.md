# Aawden — security review robustness

**Evidence: local checkpoint generations; see model manifest.**

## Research question

Can a bounded checkpoint edit reduce incorrect deference to a reviewer's opinion while preserving independent security review?

## Dataset and oracle

50 cases in 24 implementation families. Project-original SQLite and access-control fixtures have executable behavioral checks. A coding agent authored and reviewed the development labels. The bundled test families are exposed and exploratory; they are not a protected final benchmark.

## Before / after

| Method | Neutral BA | Incorrect-suggestion BA | Correct-suggestion accuracy | False-positive rate under incorrect suggestions | Coverage |
|---|---:|---:|---:|---:|---:|
| A | 0.0% | 0.0% | 6.0% | 0.0% | 0.0% |
| B | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

A = original/standard; B = original/strong; C = edited/standard; D = edited/strong; R = seeded matched random control; N = unchanged software control. Invalid outputs and abstentions count as errors in balanced accuracy.

## Paired uncertainty


Development and toy intervals are descriptive and cannot establish real-model transfer.

## Mechanism and intervention

CircuitKIT supplies instrumented inference for residual contrasts and causal patches. The project projects selected residual-output weight matrices, then evaluates full generated verdicts. Forced-choice evidence and full generation evidence are distinct.

## Candidate search and freeze

No candidate is selected by this evaluation command.

The trial ledger records targeted edits, random controls and their outcomes. Validation families choose the candidate.

## Reproduction and integrity

Run python aawden.py verify --run <this-run-directory>. The run manifest records the source identity, configuration, prompt version, model metadata, predictions, and artifact SHA-256 values. Raw generations and strict parse failures are retained.

## Generation resources

| Method | Answers | Input tokens | Output tokens | Generation seconds | Invalid answers | Generation errors |
|---|---:|---:|---:|---:|---:|---:|
| A | 150 | 36053 | 17875 | 449.87 | 147 | 0 |
| B | 150 | 46553 | 17864 | 446.98 | 150 | 0 |

Stronger prompt variant used for this run is `trace`.

## Limits and unmeasured endpoints

- human-reviewed explanation grounding
- general coding regressions
- independent label review
- protected final-test transfer

Interpret the comparison within the tested task, prompts, model, and stated uncertainty. The reported benchmark does not establish general safety or new cybersecurity knowledge.

## Sponsor integrations

- AuditKIT: executed. Benchmark labels and statistical inference remain owned by Aawden.
