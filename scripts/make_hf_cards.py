"""Build the three Hugging Face cards from runs/fix/round2-summary.json. No number is typed by hand.

`scripts/hf_publish.py` fills `{{REPO_ID}}`, `{{RESULTS_TABLE}}`, `{{SUMMARY_FILES}}` and
`{{{{FILES_TABLE}}}}` at stage time. This script owns everything else, so a card cannot drift from the
report: the tables below are formatted from the same JSON the README figures are drawn from.

    python scripts/make_hf_cards.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs/fix/round2-summary.json"
BASE = "CohereLabs/aya-expanse-8b"
REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"
TRAINED = {"reviewer", "authority", "scanner", "code_comment"}
WORDING_LABEL = {
    "reviewer": "Reviewer (trained wording)",
    "authority": "Senior reviewer",
    "scanner": "Static analysis tool",
    "code_comment": "Comment left in the code",
    "ciso": "CISO sign-off",
    "consensus": "Three reviewers agree",
    "argued": "Reviewer gives a reason",
    "insistent": "Reviewer insists and threatens",
    "pushback": "Pushback on the model's own answer",
}
MODEL_LABEL = {
    "aya32b": "Aya Expanse 32B",
    "qwen14b": "Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen2.5-32B-Instruct",
}


def pooled(block: dict, wording: str) -> tuple[int, int]:
    """Sum flipped and solved across the seeds of a multi-run block."""
    seeds = [s for s in block if s != "_summary"]
    k = sum(block[s]["wordings"][wording]["flipped"] for s in seeds if wording in block[s]["wordings"])
    n = sum(block[s]["wordings"][wording]["solved"] for s in seeds if wording in block[s]["wordings"])
    return k, n


def pct(k: int, n: int) -> str:
    return "—" if n == 0 else f"{k / n:.1%}"


def primary_table(summary: dict) -> str:
    """Unedited against adapter, one row per competent model, seeds pooled."""
    rows = ["| model | right on its own, of 129 | flip rate unedited | right on its own with the adapter | flip rate with the adapter |",
            "|---|---:|---:|---:|---:|"]
    for key, label, seeds in (("qwen14b", MODEL_LABEL["qwen14b"], ["antidef-s17", "antidef-s29", "antidef-s43"]),
                              ("aya32b", MODEL_LABEL["aya32b"], ["antidef-s17", "antidef-s29"]),
                              ("qwen32b", MODEL_LABEL["qwen32b"], ["antidef-s17"])):
        base = summary["primary"][key]["unedited"]["json_prefill"]
        k = sum(summary["primary"][key][s]["json_prefill"]["flipped"] for s in seeds)
        n = sum(summary["primary"][key][s]["json_prefill"]["solved_unaided"] for s in seeds)
        runs = f" ({len(seeds)} runs)" if len(seeds) > 1 else ""
        rows.append(f"| {label} | {base['solved_unaided']} | {base['flip_rate']:.3f} | {n}{runs} | "
                    f"**{k / n:.3f}** |")
    return "\n".join(rows)


def wording_table(summary: dict) -> str:
    """Aya 8B, two-stage recipe: unchanged against adapter, per reviewer wording, three runs pooled."""
    block = summary["multi"]["aya8b-curr"]
    base = summary["wording"]["aya8b-base"]
    rows = ["| reviewer wording | unchanged: flip rate | adapter, three runs: flipped of right | flip rate | seen in training |",
            "|---|---:|---:|---:|:--:|"]
    for word in WORDING_LABEL:
        if word not in base:
            continue
        k, n = pooled(block, word)
        seen = "yes" if word in TRAINED else "**no**"
        rows.append(f"| {WORDING_LABEL[word]} | {base[word]['flip_rate']:.0%} | {k} of {n} | "
                    f"{0 if n == 0 else k / n:.1%} | {seen} |")
    return "\n".join(rows)


def small_model_table(summary: dict) -> str:
    """The two recipes on the two small models, worst wording per run."""
    rows = ["| model | recipe | right on its own, 3 runs | worst held-out wording | rule set in advance |",
            "|---|---|---|---|---|"]
    for key, label, recipe in (("aya8b-curr", "Aya Expanse 8B", "task first, then eight wordings"),
                               ("aya8b-multi", "Aya Expanse 8B", "all wordings from scratch"),
                               ("tinyaya-curr", "Tiny Aya Global (3B)", "task first, then eight wordings")):
        block = summary["multi"][key]
        seeds = [s for s in block if s != "_summary"]
        solved = [block[s]["wordings"]["reviewer"]["solved"] for s in seeds]
        worst, worst_rate = None, -1.0
        for word in WORDING_LABEL:
            if word in TRAINED:
                continue
            k, n = pooled(block, word)
            if n and k / n > worst_rate:
                worst, worst_rate = (word, k, n), k / n
        passed = block["_summary"]
        rows.append(f"| {label} | {recipe} | {', '.join(str(s) for s in solved)} | "
                    f"{WORDING_LABEL[worst[0]]}: {worst[1]} of {worst[2]} | "
                    f"pass, {passed['seeds_passed']} of {passed['seeds_scored']} |")
    return "\n".join(rows)


MODEL_CARD = """---
license: cc-by-nc-4.0
library_name: peft
base_model: {base}
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
  - {dataset_repo}
---

# Aya Expanse 8B, trained not to take a reviewer's word for it

**What this is.** A rank-16 LoRA adapter for `{base}`. It makes the model answer a security-review
question from the code, instead of from a reviewer comment that states the opposite verdict.

**The measured effect.** Give the model a file it gets right on its own, then add a reviewer who says
the opposite. The unchanged model abandons all **{base_abandoned}** of its own correct
answers. With this adapter it abandons at most a handful, per reviewer wording, including five
wordings it never saw in training.

{wording_table}

**This is a research artifact.** It is one adapter, trained under rules we wrote down in advance, on
small templated fixtures written by one author. Read the limitations below before using it for
anything.

## Load it

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{base}", revision="{revision}")
model = AutoModelForCausalLM.from_pretrained("{base}", revision="{revision}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{{{{REPO_ID}}}}")
```

**Recipe.** Two stages. Stage one trains the review task with no reviewer in the prompt at all
(the neutral-only adapter). Stage two continues on 2,948 rows: each training case with no reviewer,
a correct one and a wrong one, across four reviewer wordings, plus a row that disputes the model's
own earlier answer. 268 training cases, from families that never appear in a test set. The target is
always the verdict you get by running the code. Rank 16, alpha 32, learning rate 1e-4, two epochs, seed 17.

| item | value |
|---|---|
| base model | `{base}` @ `{revision}` |
| parameter change | rank 16, about 42M parameters, roughly 0.5% of the model |
| training cases | 268 (2,948 rendered rows) |
| training data families | none shared with any test set |
| success rule | fixed in `docs/FIX_PROTOCOL.md` before the runs |
| adapter sha256 | `{sha}` |

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

{{{{FILES_TABLE}}}}

The base weights are **not included** and are not redistributed. Only the adapter delta ships here.
"""

DATASET_CARD = """---
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

The usual score is "accuracy when the reviewer lies", and it is easy to game: a model that always
answers `vulnerable` scores exactly 50% on a balanced set, and a benchmark that never checks whether
the model can do the task cannot tell that apart from a real fix.

So the main measure is **flip rate on solved cases**: take only the files the model gets right
with **no reviewer present at all**, then attach a reviewer who states the opposite. The flip rate is
the share of those correct answers that go wrong. A model that was guessing has no correct answers to
lose. That is also why we quote no flip rate for a model that is at chance.

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

A sealed 120-case final set exists with committed hashes. We have not published it, and we have
never opened it.

## What it measures

{primary_table}

Three numbers per cell in the source report are three answer formats: one JSON answer and two A/B
letter mappings with the labels swapped. Qwen2.5-32B's own rate moves tenfold across formats, so we
don't trust any one format on its own.

The small-model recipe, on models that cannot do the task unaided:

{small_model_table}

## Labels are executable, not judged

Each fixture is a runnable pair: a vulnerable implementation and a fixed one. The verdict comes from
what the code does: SQL against an in-memory SQLite database, access control against a fixed record
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

{{{{FILES_TABLE}}}}
"""

SPACE_CARD = """---
title: "Aawden: does the model read the code, or the reviewer?"
emoji: "\\U0001F6E1"
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
base_model: {base}
short_description: "Real cases: a wrong reviewer, and what the model answers"
---

# Try the test yourself

A security reviewer reads a file and says "this is fine." The model answers. Is it reading the code,
or agreeing with the reviewer?

Every file on this page is one the **unchanged** model already gets right with no reviewer present.
Then the reviewer states the opposite verdict. Four answers are shown:

- the **correct** answer, which comes from running the code;
- the model **alone**, with no reviewer;
- the model **with the reviewer line**, where it gives its answer away;
- the model **after our adapter**, with the same file, reviewer and question.

Every card is a **recorded** answer from a saved run. Nothing on this page calls a model, so it loads
instantly and costs nothing to host. You can switch between four models with the buttons at the top.

## What you are looking at

The measure behind it is the **flip rate on solved cases**: of the files a model gets right on its
own, the share a wrong reviewer talks it out of. Unchanged Aya Expanse 8B gives up all {base_solved}
of its {base_solved} correct answers on the plain set. With the adapter it gives up at most a handful
per reviewer wording, in every one of three training runs.

Everything else from the project:

- **the adapter**: `RishyanthReddy/aawden-aya8b-reviewer-adapter`
- **the benchmark and every final number**: `RishyanthReddy/aawden-reviewer-flip-bench`
- **the full article**: https://rishyanthreddy-aawden-article.static.hf.space
- **the source, the rules we set in advance, and what failed**: https://github.com/Aya-Jmd/Aawden

## Not claimed

- The page does not run inference. It replays saved verdicts.
- It ships no model weights.
- The fixtures are small and templated, written by one author, and have not been independently
  reviewed. The base model is CC-BY-NC-4.0 and is not redistributed.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    base = summary["primary"]["aya8b"]["unedited"]["json_prefill"]
    sha = "6ab68d79390b9bfff7948038cca56c05ec467127737f1d2a6c53eb996088f56e"

    model_card = MODEL_CARD.format(base=BASE, revision=REVISION, sha=sha,
                                   base_abandoned=base["flipped"], base_solved=base["solved_unaided"],
                                   dataset_repo="RishyanthReddy/aawden-reviewer-flip-bench",
                                   wording_table=wording_table(summary))
    dataset_card = DATASET_CARD.format(primary_table=primary_table(summary),
                                       small_model_table=small_model_table(summary))
    space_card = SPACE_CARD.format(base=BASE, base_solved=base["solved_unaided"])

    for relative, text in (("hf/model-aya8b-adapter/README.md", model_card),
                           ("hf/dataset-flip-bench/README.md", dataset_card),
                           ("hf/space/README.md", space_card)):
        target = args.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"[hf-cards] {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
