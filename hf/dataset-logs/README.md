---
license: cc-by-nc-4.0
pretty_name: Aawden run logs
language:
  - en
task_categories:
  - text-classification
tags:
  - security-review
  - logs
  - traces
  - evaluation
  - aya-expanse
  - research
size_categories:
  - n<1K
---

# Aawden run logs

**Status: research logs. Single author. Small runs. Independent review pending.**

Run manifests, per-answer predictions, summaries, decisions and agent transcripts from the Aawden experiments on Aya Expanse 8B. Each prediction row keeps the raw model output, the parsed verdict, the oracle verdict and the prompt hash, so every number in the summaries can be recomputed.

## License

These logs contain Aya Expanse 8B generations. They are derived from CC-BY-NC-4.0 model output and are subject to the [Cohere Labs Acceptable Use Policy](https://huggingface.co/CohereLabs/aya-expanse-8b). The collection is published under CC-BY-NC-4.0. The Aawden code is MIT, but it does not relicense the model output.

## Contents

- `runs/wsp-v3/`: protocol-v3 run directories, including Beam launch records and smoke-run check files.
- `runs/aya-beam-20261107/`: Beam GPU runs for adapter study v1 to v7, adapter review, mental review and evidence bundles.
- `runs/aya-rigorous/`: rigorous development and confirmation runs.
- Transcripts in `runs/**/transcripts/**/*.md`, if present.

Only the files listed below are included. Zip archives, checkpoints and base weights are not.

{{FILES_TABLE}}

## Results

Balanced accuracy from the saved summaries listed below. Each row is one run and split.

{{RESULTS_TABLE}}

Summary files used for this card:

{{SUMMARY_FILES}}

## How to read the files

- `results.json` holds one row per answer. Fields include `case_id`, `split`, `category`, `method`, `condition`, `raw_output`, `predicted_verdict`, `oracle_verdict`, `correct`, `prompt_sha256` and token counts.
- `summary.json` recomputes the metrics from `results.json`. Its `model` block records the revision and environment.
- `manifest.json` and `check.json` files record the source fingerprint and hashes the runner verified.

## Limits

- Logs cover only the runs listed here. They are not a complete record of every experiment attempted.
- Failed or superseded attempts are kept where the project documents them, so the attempt history is visible.
- Use for research measurement only. Outputs need human verification before any security decision.
