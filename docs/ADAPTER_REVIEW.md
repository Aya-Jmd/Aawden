# Frozen adapter and review instructions

The focused-verdict adapter reached 92.3% on 130 exposed development cases and 70% on the next 20-case check. Its weights remain frozen for this cheaper follow-up. The 20 scored cases now serve as prompt-validation data.

Compare four prespecified instructions on those cases. The first uses the existing standard prompt. The other three add the category's policy checklist, a checklist plus a concrete counterexample, or a checklist plus general vulnerable/safe contrasts. The model generates all five JSON fields. There is no response prefill or relaxed parser.

Reuse the prior 20 standard-prompt answers only when the adapter hash, GPU model, PyTorch runtime, cases, and exact messages match. Otherwise regenerate that arm. Choose the highest validation balanced accuracy, then the fewest invalid answers, then the declared variant order. If no variant exceeds 75%, stop before confirmation.

Save the prompt choice and the existing adapter hash before evaluating 20 new cases across ten families. The new set is balanced across labels and categories. Its executable checks passed before the run, and its file hash is fixed in `configs/adapter-review.json`. It covers delegated SQL execution, repeated bindings, trusted query structure, command objects, lambda policies, structural pattern matching, key-owner pairs, and ignored guard results. It has one author and supports a feasibility finding. A separate final evaluation remains necessary for broader claims.

From the project root, run `python aawden.py adapter-review --previous runs/aya-beam-20261107/adapter-v6/study/confirmation --output runs/adapter-review-reproduction`. The local adapter lives in `checkpoints/aya-adapter.safetensors`. The loader verifies the adapter bytes and all pinned base shard hashes. The same adapter is available to existing inference commands through `configs/model-adapted.json`.

This follow-up performs no training. It evaluates at most 100 answers and has a 15-minute cloud timeout. The launcher checks the live credit balance against the user's $18 floor (`configs/compute-budget.json`) and the additional $1 safety margin before launching.

The first launch stopped during adapter loading because a JSON path string reached a file-hashing helper that expects a `Path`. No selection answers were generated. The loader now normalizes the path, and a local regression test verifies loading from a JSON-style string before the technical rerun.

The completed comparison scored 70%, 80%, 70%, and 70% on the exposed validation set. It selected the trace instruction, froze that choice, and scored **75% on the fourth fresh 20-case set**. All ten SQL answers were correct and five of ten access-control answers were correct. The strict goal of exceeding 75% remained unmet. These results measure neutral competence on small developer-authored fixtures. They do not establish robustness to misleading advice or a causal circuit repair.
