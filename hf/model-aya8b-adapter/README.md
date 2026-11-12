---
license: cc-by-nc-4.0
library_name: peft
base_model: CohereLabs/aya-expanse-8b
base_model_relation: adapter
language:
  - en
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - aya-expanse
  - security-review
  - sycophancy
  - deference
  - research
datasets:
  - RishyanthReddy/aawden-reviewer-flip-bench
---

# Aya Expanse 8B, trained not to take a reviewer's word for it

**What this is.** A rank-16 LoRA adapter for `CohereLabs/aya-expanse-8b`. It makes the model answer a security-review
question from the code, instead of from a reviewer comment that states the opposite verdict.

**The measured effect.** Give the model a file it gets right on its own, then add a reviewer who says
the opposite. The unchanged model abandons all **67** of its own correct
answers. With this adapter it abandons at most a handful, per reviewer wording, including five
wordings it never saw in training.

| reviewer wording | unchanged: flip rate | adapter, three runs: flipped of right | flip rate | seen in training |
|---|---:|---:|---:|:--:|
| Reviewer (trained wording) | 100% | 2 of 325 | 0.6% | yes |
| Senior reviewer | 100% | 3 of 325 | 0.9% | yes |
| Static analysis tool | 18% | 3 of 325 | 0.9% | yes |
| Comment left in the code | 82% | 4 of 325 | 1.2% | yes |
| CISO sign-off | 98% | 3 of 325 | 0.9% | **no** |
| Three reviewers agree | 100% | 4 of 325 | 1.2% | **no** |
| Reviewer gives a reason | 87% | 7 of 325 | 2.2% | **no** |
| Reviewer insists and threatens | 96% | 4 of 325 | 1.2% | **no** |
| Pushback on the model's own answer | 100% | 15 of 325 | 4.6% | **no** |

**This is a research artifact.** It is one adapter, trained under rules we wrote down in advance, on
small templated fixtures written by one author. Read the limitations below before using it for
anything.

## Load it

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("CohereLabs/aya-expanse-8b", revision="5062468bf9bc0c6035fd64e06274333ec127d980")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/aya-expanse-8b", revision="5062468bf9bc0c6035fd64e06274333ec127d980", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{{REPO_ID}}")
```

**Recipe.** Two stages. Stage one trains the review task with no reviewer in the prompt at all
(the neutral-only adapter). Stage two continues on 2,948 rows: each training case with no reviewer,
a correct one and a wrong one, across four reviewer wordings, plus a row that disputes the model's
own earlier answer. 268 training cases, from families that never appear in a test set. The target is
always the verdict you get by running the code. Rank 16, alpha 32, learning rate 1e-4, two epochs, seed 17.

| item | value |
|---|---|
| base model | `CohereLabs/aya-expanse-8b` @ `5062468bf9bc0c6035fd64e06274333ec127d980` |
| parameter change | rank 16, about 42M parameters, roughly 0.5% of the model |
| training cases | 268 (2,948 rendered rows) |
| training data families | none shared with any test set |
| success rule | fixed in `docs/FIX_PROTOCOL.md` before the runs |
| adapter sha256 | `6ab68d79390b9bfff7948038cca56c05ec467127737f1d2a6c53eb996088f56e` |

## How the numbers were produced

Every label is **executable**: the fixtures run real code (SQL against in-memory SQLite, access
control against a fixed record table) and the verdict is what the code actually does. The 129-case
plain set is generated with committed hashes. Three training runs were scored against the same rule;
this repository ships seed 17.

Flip rate counts only files the model gets right with no reviewer at all, so a model that is
guessing has no correct answers to lose. The unchanged Aya 8B is at chance on this set, which is why
its 100% flip rate means "copies whatever it is told" and not "knows better and defers".

## Limitations

- **It ignores every reviewer, right or wrong.** It does not learn which reviewers to trust. That
  is the price: the model is no longer rescued by a right reviewer on the few cases it misses.
- **Templated fixtures.** 64 families in the plain set, 30 in the puzzle set, one language (Python).
- **One author, one protocol.** Not independently reviewed.
- **Commercial use is not granted.** The base model is CC-BY-NC-4.0 under Cohere Labs' Acceptable Use
  Policy; this adapter inherits those terms and cannot be used commercially.

## Files

{{FILES_TABLE}}

The base weights are **not included** and are not redistributed. Only the adapter delta ships here.
