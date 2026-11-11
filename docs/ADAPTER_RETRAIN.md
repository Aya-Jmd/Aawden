# Retraining the v7 adapter lineage

The adapter files `checkpoints/aya-adapter-v7.safetensors` and the v8-v10 files are absent (`checkpoints/` does not exist, checked 2026-11-10). This note reconstructs the recipe from the recorded evidence and describes how to retrain v7. Retrained weights will not be byte-identical to the recorded hashes. Every fact below names its evidence file. Anything not found is marked UNKNOWN.

Files added for this: `configs/retrain-v7.json` (recipe), `scripts/retrain_adapter_chain.py` (runner), `tests/test_retrain_chain.py` (tests).

## Lineage: v5 -> v6 -> v7 (all on the same base)

Base for all three: `CohereLabs/aya-expanse-8b`, revision `5062468bf9bc0c6035fd64e06274333ec127d980`, rank 16, alpha 32, seven projections, seed 42, AdamW without weight decay, accumulation 2, gradient clip 1. Evidence: `runs/aya-beam-20261107/adapter-v7/study/protocol.json` (rank, alpha, modules, seed, conditions, accumulation), `scripts/train_aya_adapter.py` (AdamW weight decay 0, clip 1), and the per-version protocols in the table.

| Version | Resumes from | Training cases (file, sha256) | Epochs / LR / verdict weight / trace prompt | Steps | Recorded result | Evidence |
|---|---|---|---|---|---|---|
| v5 | nothing (fresh LoRA) | `screen.json` 3fedabb7 (50), `neutral-confirmation.json` 98ab77d1 (40), `neutral-confirmation-v2.json` 61e6ba84 (40) = 130 | 4 / 2e-4 / 20 / no | 1040 | Dev 50.0% on the 130 training cases; gate (75%) failed; no confirmation | `runs/aya-beam-20261107/adapter-v5/study/{protocol,training,outcome}.json`, `.../study/development/summary.json`. Adapter sha256 `7ff5fdda...` (from summary.json) |
| v6 | v5 (`7ff5fdda...`, protocol `resume_sha256` matches) | same 130 | 6 / 2e-4 / 50 / no | 1560 | Dev 92.3% on the 130 training cases; passed gate; confirmation 70.0% on `neutral-confirmation-v3.json` 5b13a9ae (20) | `runs/aya-beam-20261107/adapter-v6/study/{protocol,frozen,outcome,training}.json`. Adapter sha256 `354c3e4c...` |
| v7 | v6 (`354c3e4c...`, protocol `resume_sha256` matches) | 130 + `neutral-confirmation-v3.json` 5b13a9ae (20) = 150 | 3 / 1e-4 / 10 / yes (`CATEGORY trace` instruction) | 900 | Validation 85.0% on `neutral-confirmation-v4.json` 8c1f1da0 (20, not trained on); passed gate; confirmation 90.0% on `neutral-confirmation-v5.json` dcab8306 (20) | `runs/aya-beam-20261107/adapter-v7/study/protocol.json`, `.../frozen.json`, `.../outcome.json`, `.../training.json`, `.../development/summary.json`, `.../confirmation/summary.json`. Adapter sha256 `e86f9475...` (448 FP32 tensors) |

Lineage answer: yes, v7 resumed from v6, and v6 resumed from v5. Each link is confirmed by a `resume_sha256` in the later protocol that equals the earlier adapter's recorded sha256. v5 started fresh (no `resume_sha256` in its protocol).

The Steps column below is examples x epochs, which is how the recorded `training.json` files count them (one forward pass per example). The chain's optimizer applies 1 update per 2 such steps.

Gates: the v5 and v6 development gate used the training cases (no held-out set). v7 is the only version with a held-out validation gate (`nc-v4`). Its `frozen.json` records the adapter and the confirmation hash before scoring.

Device: v5 and v6 ran on an RTX 4090 (`gpu.txt`). v7 ran on an RTX 4090 (`gpu.txt`).

### What the recipe does and does not pin down

- Training code: v7's `protocol.json` records `training_script_sha256 ae89ac42...`, which equals the HEAD script. The v7 snapshot (`runs/aya-beam-20261107/adapter-v7-reproduction-source.zip`, zip sha256 `2d4e83fb...`, fingerprint `52ec422a...`) reproduces that fingerprint exactly.
- v5 and v6 manifests record source fingerprint `c10302a6...` (`adapter-v5/study/development/manifest.json`, `adapter-v6/study/development/manifest.json`). **UNKNOWN - not checked for the exact code**: no zip or git commit in this repo reproduces it. The repository has three commits (`git rev-list --all`, 2026-11-10). Their `src` fingerprints, computed with the same algorithm as `artifacts.source_fingerprint`, are `3a1f7ac` = `2c1f31a6...`, `de4c69a` = `2c1f31a6...`, `a564d52` = `ca32da26...`. The v4 receipt (`runs/aya-beam-20261107/adapter-v4-source-receipt.json`) records a different `scripts/train_aya_adapter.py` (`2f20f5db...`), which also matches neither HEAD (`ae89ac42...`) nor the working tree (`f403c543...`).
- v5 and v6 protocols record no dataset or training-script hashes. The dataset hashes above are inferred from v7's `training_files` list, which names the same files. UNKNOWN - not recorded for v5 and v6.
- LoRA scale: the historical code hard-coded 2.0. The v7 snapshot computes `alpha / rank = 32 / 16 = 2.0` (`src/security_wsp/adaptation.py`, `lora_scale`). The numbers match, so this is probably not a source of difference. Verified by reading the code only.

## v8-v10 (brief)

Each step is a `security_wsp.rigorous_training.refine` run (`src/security_wsp/rigorous_training.py`). The refine uses the training block of its protocol config.

| Version | Parent | Protocol (training block) | Training cases | Recorded result | Evidence |
|---|---|---|---|---|---|
| v8 = `checkpoints/aya-adapter-v8` (sha256 `60154fe8...`) | v7 `e86f9475...` | `configs/rigorous-review-v2.json`: exposed `rigorous-confirmation.json` ccb235aa (32) + replay `neutral-confirmation-v4.json` (20); variants source_map, source_map_first; LR 1e-4; 2 epochs; seed 94; no verdict weight | 52 | Refined development: source_map 75.0%, source_map_first 83.3%; below the 0.90 gate; no confirmation | `runs/aya-rigorous/rigorous-03/study/{training,outcome}.json` (RTX 4090, `gpu.txt`) |
| v9 = `checkpoints/aya-adapter-v9` (sha256 `e458a1b0...`) | v8 `60154fe8...` | `configs/rigorous-review-v4.json`: exposed `rigorous-confirmation-v2.json` 2ba8eb02 (32) + replay `rigorous-confirmation.json` (32); variants policy, policy_first; LR 8e-5; 2 epochs; verdict weight 4; seed 109 | 64 | Refined development: policy 86.1%; below the 0.88 gate; no confirmation | `runs/aya-rigorous/rigorous-06/study/{training,outcome}.json` (**A10G**, `gpu.txt`) |
| v10 = `checkpoints/aya-adapter-v10` (sha256 `0ea3441c...`) | v9 `e458a1b0...` | `configs/rigorous-review-v9.json`: exposed `rigorous-confirmation-v3.json` 50566f16 (32) + replay `rigorous-confirmation-v4.json` c2b2e09e (32); variants compact, scenarios; LR 6e-5; 2 epochs; verdict weight 4; seed 137; trained before validation | 64 | Development policy 94.4% (gate 0.88 met); routed confirmation 90.6% (29/32) on `rigorous-confirmation-v5.json` f7a7ba5d | `runs/aya-rigorous/rigorous-13/study/{training,outcome,confirmation/summary}.json`, `configs/rigorous-review-v10.json` (adapter `0ea3441c`) |

The v8-v10 chain is documented but not implemented in `scripts/retrain_adapter_chain.py`. Its target choice is `v7` only. The v10 path would call `rigorous_training.refine` three times with the training blocks above. Per-job wall clock is UNKNOWN - not recorded (the rigorous runs record only per-arm evaluation latency: 84-110 s for 36 cases).

## Recipe for v7

`configs/retrain-v7.json` holds the full recipe, dataset hashes, the recorded hashes and the timing basis. The runner `scripts/retrain_adapter_chain.py` is a standard-library planner plus a training loop. It reuses `train_aya_adapter.py`'s prompt, target and request helpers from the snapshot. Its loop mirrors `main()` step for step. It cannot call that script directly: the non-refine path hard-codes v6 values, and the refine path refuses any resume whose hash is not the original v6 hash.

Commands (GPU host; the snapshot must be extracted first, see below):

```
# plan only, no torch import, deterministic
python scripts/retrain_adapter_chain.py --dry-run --source-root SNAPSHOT

# one job per stage (each fits the 2100 s cap); each stage reads the previous output
python scripts/retrain_adapter_chain.py --stage v5 --output checkpoints/retrained --source-root SNAPSHOT
python scripts/retrain_adapter_chain.py --stage v6 --output checkpoints/retrained --source-root SNAPSHOT
python scripts/retrain_adapter_chain.py --stage v7 --output checkpoints/retrained --source-root SNAPSHOT

# all three in one process: estimated 56 min, refused without --allow-long-run
python scripts/retrain_adapter_chain.py --output checkpoints/retrained --source-root SNAPSHOT --allow-long-run

# DEVIATION: one stage, v7 recipe trained from the base model (skips v5 and v6; 900 steps)
python scripts/retrain_adapter_chain.py --shortcut --output checkpoints/retrained --source-root SNAPSHOT
```

Outputs: `<output>/<stage>.safetensors` and `<output>/<stage>.json` (adapter sha256, settings, history, source fingerprint, chain-script and training-script hashes, evaluation results). The runner refuses to overwrite any existing file (`os.link` plus pre-checks), refuses a resume file that disagrees with its own record, refuses a source fingerprint other than `52ec422a...`, and refuses dataset hash or count changes.

### GPU estimate

Basis: RTX 4090 step time 0.5627 s (Beam task `54b135a6`, steps 525 to 900, `recovered-logs.json`). Setup 311 s and post 120 s are INFERRED from the recorded 17 min 23 s task time. Evaluation is 2.5 s per case (v7: 50.8 s and 49.8 s for 20 cases each).

| Job | Steps | Estimated job time | Within 2100 s cap |
|---|---|---|---|
| v5 | 1040 | ~17 min (1016 s) | yes |
| v6 | 1560 | ~22 min (1309 s) | yes |
| v7 (plus two 20-case evaluations) | 900 | ~17 min (1037 s); recorded 1043 s | yes |
| Total, three jobs | | ~56 min | 3 Beam jobs |
| Shortcut, one job | 900 | ~17 min | 1 Beam job |

Beam cost at the configured all-in rate ($1.70/h, `configs/beam.json`): lineage about $1.59; shortcut about $0.49. Current Beam balance: UNKNOWN - not checked. The $18 floor in `configs/compute-budget.json` applies. Each launch needs `--max-cost-usd 1.00` (the guard at timeout 2100 s is $0.99).

## Blockers and deviations

1. **Snapshot delivery to Beam (not solved).** `.beamignore` excludes `runs/` and `checkpoints/`, and the snapshot zip is under `runs/`. The launcher uploads the working tree, not the snapshot. UNKNOWN - not checked whether `beam_launch.py` can upload the snapshot or a volume file. The working tree also does not match the v7 fingerprint (`a338449c...` now; it was `36914f81...` earlier, so the tree changed during this work). Options: an extracted, uploaded snapshot directory, or a volume upload step.
2. **Chained adapters across Beam jobs.** `beam_entry.py` copies `runs/<output_subdir>` to the volume. Use `--output runs/retrain-v7/v5`, then pass `--resume-from /volumes/aawden-v3/retrain-v7/v5/v5.safetensors` to the v6 job. This is the intended path but UNKNOWN - not run.
3. **The 3-job split is the only way to meet the cap.** The plan refuses the single process by default because its estimate (3362 s) exceeds 2100 s.
4. **Shortcut quality is unmeasured.** Earlier from-scratch runs scored 58% (v1, 4 epochs at 2e-4) and 50% (v5, 4 epochs at 2e-4) on development. The shortcut at 3 epochs and 1e-4 has no evidence.
5. Retraining will not reproduce v7's weights, the v5/v6 source and script are not recovered, and the v7 result was tuned on exposed sets.

## Verification

- `tests/test_retrain_chain.py`: 11 tests. They check that stage hyperparameters and resume hashes match the recorded protocols, that dataset hashes match the files, that the dry-run plan is deterministic and matches the step counts in the recorded `training.json` files (1040, 1560, 900), that `--dry-run` does not import torch (checked in a subprocess with `-I`), that the shortcut is labelled, that existing checkpoints are refused, that a missing or mismatched resume file is refused, and that the all-in-one run is refused without a flag.
- Full suite at the time this note was written: 285 passed, 1 skipped, 1 failed. The failure is `tests/test_wsp_tools.py::test_activation_contrast_reports_a_degenerate_pairing`, a `transformers` cohere-model `TypeError` in a test file that appeared during this work and is outside this task's scope. The 11 tests in `tests/test_retrain_chain.py` all pass.
- No GPU job was launched.
