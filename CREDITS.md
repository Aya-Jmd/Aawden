# Credits

Two people built Aawden. This file says who did what, and how each claim can be checked against the
repository rather than taken on trust.

## Aya-Jmd — the library, the fixtures, and the platform the study runs on

Handover commit `3a1f7ac` (10 Nov 2026, "preparing the repo for handoff"), preceded by `a564d52`
(8 Nov) and `de4c69a` (10 Nov).

At handover the tree held **2,783 files, 114 Python files, 8,446 lines of Python**, including:

- **`src/security_wsp/`** — 42 Python modules plus a small web UI: the prompt renderer
  (`prompts.py`), the `Case` model and dataset loader (`data.py`), parsers, statistics, the
  intervention scaffolding (`interventions.py`), the CLI, and the early confirmation-review variants.
  This is the machinery every experiment in the repository imports.
- **`fixtures/`** — 51 executable fixtures, unchanged since.
- **`datasets/`** — 16 datasets, including the base fixture pools.
- **`configs/`, `tests/`, `docs/`** — the configuration surface, 12 test modules, and the first
  documentation.
- **`runs/aya-beam-20261107/`, `runs/aya-rigorous/`** — the first recorded runs.

**It was not replaced.** Of those 2,783 files, **2,753 are byte-identical in the current tree (98.9%)**.
Of the 114 Python files, **97 are untouched, 17 modified, 0 deleted**. Seven files were removed in
total, and none of them was working code:

| removed | why |
|---|---|
| `AAWDEN_TODO.md`, `EXPLANATION.MD` | handover notes, superseded by `README.md` and `docs/SUBMISSION_FINAL.md` |
| `MODEL_CARD.md` | superseded by `docs/MODEL_CARD_V3.md` |
| `START_AAWDEN.ps1` | a Windows launcher; the CLI is `python aawden.py` |
| `notebooks/01_cohere_pilot.ipynb` | an early pilot notebook |
| `runs/aya-beam-20261107/recovered-*.json` | 316 KB of recovered Beam logs, re-recorded under `runs/` |

`fixtures/` has not changed at all.

## M. Rishyanth Reddy — the measurement, the fix, and the published artifacts

Commits `466d039` onward.

- **The measurement.** The competence gate (a flip rate is only quoted where the unaided score clears
  chance), the flip rate on solved cases, the three answer formats, and the `J` criterion that rules
  out a code-blind policy.
- **The diagnosis.** The 2×2 decomposition showing the reviewer's opinion moves the verdict by 10.7–
  19.7 nats while the code moves it by +0.06–0.41, and the confound in the earlier grouping that
  reported +6.3 nats instead.
- **The fix.** The rank-16 anti-deference LoRA, the neutral-only task control that separates
  "learned the task" from "learned to resist the reviewer", the two-stage recipe for Aya Expanse 8B,
  and the strength sweep showing what reviewer-invariance costs.
- **The nulls.** Head ablation, direction editing, distillation, and the from-scratch recipe — each
  reported with the numbers it produced, including the two claims withdrawn after checking.
- **The artifacts.** The live GPU endpoint, the adapter repository, the benchmark repository with
  executable oracle labels, the demo page, and every figure.

Code added since handover: **43 scripts (1,007 → 16,308 lines), 18 test modules (1,027 → 7,311),
31 documents (2,121 → 7,173)**, and 4 new run directories. Three modules were added to `src/`
(`patches.py`, `wsp_cases.py`, `wsp_tools.py`); the library itself grew 6,487 → 8,648 lines and
remains mostly Aya-Jmd's.

## How this is checkable

The handover commits are **not in the published history**. `master` on GitHub begins at a single
commit that imports the whole tree, so a fresh clone shows every file authored by one account. The
provenance above is therefore stated as fact, not offered as something a reader can reconstruct from
`git log`. If the handover branch is published, these commands recover it:

```bash
# the handover commits, their dates and messages
git log --author='Aya-Jmd' --format='%h %ad %s' --date=short archive/pre-release

# the handover tree and its size
git ls-tree -r --name-only 3a1f7ac | wc -l          # 2783

# how much of that tree is untouched today
git diff --shortstat 3a1f7ac HEAD                   # 1902 changed, 2753 identical

# every run table quoted in the README postdates the handover
git log --oneline --diff-filter=A -- runs/fix runs/wsp-v3 | tail -3
```

What a reader *can* check from the published tree today:

- `fixtures/` (51 files) and `runs/aya-beam-20261107/`, `runs/aya-rigorous/` are carried over
  unchanged, and were written before the measured sets existed.
- `src/security_wsp/` is the library every script imports; only three modules in it postdate the
  handover.
- Every figure and table the README quotes resolves to a file under `runs/wsp-v3/`, `runs/fix/`,
  `runs/bench/` or `runs/distill/`, all of which are newer than that library.

The short version: **Aya-Jmd built the platform; the study, the fix and the published artifacts were
built on it.** Neither half stands alone — the measurement needed the library, and the library on its
own had no result to report.

## External material

Third-party dependencies, pinned sponsor sources, and the base model are credited separately in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Two adapters trained by other teams are referred
to as `third-party-adapter-A` and `third-party-adapter-B` and are not named here.
