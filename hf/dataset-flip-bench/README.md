---
license: cc-by-nc-4.0
pretty_name: Aawden reviewer-flip benchmark
language:
  - en
task_categories:
  - text-classification
tags:
  - security-review
  - benchmark
  - code
  - sycophancy
  - deference
  - executable-oracle
  - sql-injection
  - access-control
size_categories:
  - n<1K
---

# A benchmark for deference to a wrong reviewer

A model reads a security-review question and a reviewer states a verdict. The code and the reviewer
opinion are **the same length, in the same place, and differ only in the verdict**. The question is
whether the model answers from the code or from the reviewer.

## Why the obvious metric does not work

The usual score is "accuracy when the reviewer lies", and it is gameable: a model that always answers
`vulnerable` scores exactly 50% on a balanced set, and a benchmark with no competence precondition
cannot tell that apart from a real repair.

So the headline measure is **flip rate on solved cases**: take only the files the model gets right
with **no reviewer present at all**, then attach a reviewer who states the opposite. The flip rate is
the share of those correct answers that go wrong. A model that was guessing has no correct answers to
lose — which is why no flip rate is quoted for a model that is at chance.

## What is in here

| file | what it is |
|---|---|
| `datasets/plain-pooled.json` | 129 plain cases: 64 families, no puzzles in them |
| `datasets/wsp-discovery.json` | 60-case puzzle set, used for selection |
| `datasets/wsp-validation.json` | 40-case puzzle set, held for validation |
| `datasets/wsp-coding.json` | 40 coding-regression questions |
| `datasets/realcode-sinkchecked.json` | 64 cases from externally written Python pairs |
| `configs/wsp-data.json` | committed hashes for every fixture file |
| `runs/fix/round2-summary.json` | every final number, machine-readable |
| `runs/fix/round2-summary.md` | the same numbers as tables, generated from the JSON |

A sealed 120-case final set exists with committed hashes and is deliberately **not** published, and
was never opened.

## What it measures

| model | right on its own, of 129 | flip rate unedited | right on its own with the adapter | flip rate with the adapter |
|---|---:|---:|---:|---:|
| Qwen2.5-14B-Instruct | 111 | 0.757 | 374 (3 runs) | **0.000** |
| Aya Expanse 32B | 90 | 0.944 | 249 (2 runs) | **0.000** |
| Qwen2.5-32B-Instruct | 118 | 0.263 | 128 | **0.000** |

Three numbers per cell in the source report are three answer formats: one JSON answer and two A/B
letter mappings with the labels swapped. Qwen2.5-32B's own rate moves tenfold across formats, so no
single format can be trusted.

The small-model recipe, on models that cannot do the task unaided:

| model | recipe | right on its own, 3 runs | worst held-out wording | rule |
|---|---|---|---|---|
| Aya Expanse 8B | task first, then eight wordings | 110, 106, 109 | Pushback on the model's own answer: 15 of 325 | PASS 3/3 |
| Aya Expanse 8B | all wordings from scratch | 90, 91, 108 | Reviewer gives a reason: 14 of 289 | PASS 3/3 |
| Tiny Aya Global (3B) | task first, then eight wordings | 105, 106, 104 | Pushback on the model's own answer: 12 of 315 | PASS 3/3 |

## Labels are executable, not judged

Each fixture is a runnable pair: a vulnerable implementation and a fixed one. The verdict comes from
what the code does — SQL against an in-memory SQLite database, access control against a fixed record
table. All 220 oracle checks pass. `datasets/realcode-sinkchecked.json` is the exception and says so
in its own `label_status`: its labels are asserted by the upstream dataset and checked with a sink
scan, not by an executable oracle.

## Limitations

- **Templated.** The plain set has 64 families; the puzzle set 30. Python only.
- **Single author, not independently reviewed.**
- **Public data.** These fixtures are public, so pretraining exposure is possible for any model
  measured on them.
- **The fixture code is synthetic by design.** It is small enough to read in one screen so that a
  human can check the oracle by hand.

## Licence

Fixture code and text are project-original. The result tables quote generations from Aya Expanse 8B
and other open checkpoints, so those files carry the model licences. The collection is published
under the stricter licence, CC-BY-NC-4.0.

## Files

{{FILES_TABLE}}
