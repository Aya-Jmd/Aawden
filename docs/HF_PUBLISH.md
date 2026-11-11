# Phase 6 publishing to Hugging Face

Everything in this document is offline except the last step, `publish`. That step is one approved
command away and refuses to run without `--confirm-owner-approved`.

Publishing account: Hugging Face user `RishyanthReddy`. The write token is `HF_PUBLISH_TOKEN` in
`.env` (scopes `repo.write` and `collection.write`). `AAWDEN_AYA_HF_TOKEN` is a read token for the
gated base model and is **never** used for writes; `publish` refuses if the two are the same or
nested, and the secret scan refuses to upload a file containing the Aya token.

## What is published

Two public, ungated collections under `RishyanthReddy`.

**Collection "Aawden — models"**

| Repo | Type | Contents | Source |
|---|---|---|---|
| `aawden-wsp-patch` | model | Sparse WSP patch: changed and original slices plus manifest | `runs/wsp-v3/patches/wsp-patch.{safetensors,json}` (not built yet) |
| `aawden-projection-patch` | model | Sparse projection patch plus manifest | `runs/wsp-v3/patches/projection-patch.{safetensors,json}` (not built yet) |
| `aawden-aya-lora-v7` | model | Retrained rank-16 LoRA adapter and loader config | `checkpoints/aya-adapter-v7.safetensors` (not on this machine) |

**Collection "Aawden — datasets"**

| Repo | Type | Contents | Source |
|---|---|---|---|
| `aawden-authority-bench` | dataset | Discovery, validation, regression and earlier case sets, hash commitments, oracle results | `datasets/*.json`, `configs/wsp-*.json`, `runs/wsp-v3/oracle/**` |
| `aawden-run-logs` | dataset | Manifests, predictions, summaries, decisions, transcripts | `runs/wsp-v3`, `runs/aya-beam-20261107`, `runs/aya-rigorous` (JSON only) |

**Space** `aawden-demo` (Gradio, CPU basic, static saved outputs). It is not in a collection, since a
collection holds models or datasets. Link the Space from the article or README.

The sealed `wsp_final` split is **not** uploaded unless `--include-final` is passed, and that is only
correct after `final-evaluate` has run once and the owner has decided to release the labels.

## Steps

```bash
PY=/Users/rishyanthreddy/.t3/scratch/2026-11-10-ok-clone-this-github-repo-496b33f9/.venv-aawden/bin/python
cd /Users/rishyanthreddy/.t3/scratch/2026-11-10-ok-clone-this-github-repo-496b33f9/Aawden

# 1. Plan: sizes, availability, weight rules, secret scan. Offline. Exit 1 if refused.
$PY scripts/hf_publish.py plan

# 2. Stage: copy files and render the cards into dist/hf-staging. Offline.
$PY scripts/hf_publish.py stage --out dist/hf-staging

# 3. Space examples (optional, replaces "results pending"). Offline.
$PY scripts/hf_publish.py build-space-examples \
    --run runs/wsp-v3/<baseline run> --run runs/wsp-v3/<patched run> \
    --patch-manifest runs/wsp-v3/patches/wsp-patch.json \
    --out hf/space/examples.json
# then re-run stage so the Space folder picks it up

# 4. Publish. Network. Requires the owner's explicit approval for this exact git state.
$PY scripts/hf_publish.py publish --confirm-owner-approved
```

`publish` re-stages from the current tree, re-runs the secret and weight checks on the staged bytes,
verifies that the token belongs to `RishyanthReddy`, then creates each repo public and ungated,
uploads the staged folder, and creates or updates the two collections.

Review `dist/hf-staging/<repo>/README.md` after staging. The rendered cards are the actual model cards
and dataset cards; they are not regenerated on the Hub side.

## Safety gates

All gates live in `scripts/hf_publish.py` and are covered by `tests/test_hf_publish.py`.

1. **Owner approval.** `publish` refuses without `--confirm-owner-approved`. Nothing else in the tool
   writes to the network.
2. **Sealed split.** Records with split `wsp_final` are dropped by `build-space-examples` unless
   `--include-final` is given, and the dataset entry for `datasets/wsp-final.json` is marked sealed.
3. **No base weights.** A file over `max_weight_bytes` (1 GiB) with a weight suffix is refused.
   Files named like base shards (`model-0000N-of-0000M.safetensors`) are refused.
4. **No implicit checkpoints.** Anything under `checkpoints/` must be listed by explicit `path` in
   `configs/hf-publish.json`; a glob into `checkpoints/` is refused, and a weight file found by a glob
   anywhere is refused.
5. **No base slices inside a patch.** A `::original` tensor in a patch is a copy of base weights, so
   `patch_violations` reads the safetensors header and refuses the file. The sparse patch format
   stores original slices for verified revert, so a public patch must be built without them.
6. **Secrets.** The scan mirrors `src/security_wsp/submission.py`: any `.env` value under a key
   containing TOKEN, KEY or SECRET (12+ characters) is matched byte-for-byte, and it also adds an
   `hf_` token shape, a `token = "..."`-style assignment rule, and a Beam-style long-token rule.
   Matched values are never printed, only the file and rule name. The scanner logic is reproduced
   rather than imported because `submission.py` keeps it inline inside `package_submission`.
7. **Aya token.** The Aya read token is added to the scan list, so a log that captured it blocks the
   upload.

`runs/aya-beam-20261107/recovered-get_task.json` matched the Beam-style rule on 76- and 85-character
mixed-class runs and is excluded by `exclude_patterns`, together with `recovered-logs.json`, zip
archives and per-answer temp files. **If that file contains a live credential, rotate it.** The tool
only guarantees that it is not uploaded.

## Licensing

- Base model: Aya Expanse 8B, `CohereLabs/aya-expanse-8b`, revision
  `5062468bf9bc0c6035fd64e06274333ec127d980`, CC-BY-NC-4.0 plus the Cohere Labs Acceptable Use Policy.
- No base weights are uploaded. Every model repo is a derivative (adapter or patch) and its card says
  so and says how to obtain the base model.
- Aawden code and fixtures are MIT. That does not relicense the model, the adapter, the patches, or
  generations sampled from the model.
- Datasets and the Space contain Aya generations, so they are published under CC-BY-NC-4.0, not MIT.
  `aawden-authority-bench` mixes MIT fixtures with model-derived oracle results and therefore carries
  CC-BY-NC-4.0. The card says so.
- Non-commercial only. Do not describe any of this as a Cohere product or as approved by Cohere Labs.

## After publishing

1. Check each repo in the browser: the card renders, the front matter parses, the files listed in the
   card are the files present.
2. Check both collections contain the expected items and are public.
3. Check the Space starts and shows either saved examples or the "results pending" message.
4. Write the HF article or point the README at the collections, and state the evidence honestly:
   single-author fixtures, small scale, independent review pending.
