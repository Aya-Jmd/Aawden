# Best observed Aya result

The frozen v7 adapter scored **90% strict neutral balanced accuracy on 20 fresh cases across ten families**. It correctly reviewed 18 cases, including nine of ten vulnerable cases and nine of ten safe cases. SQL scored 10/10 and access control scored 8/10. All answers passed the original five-field JSON parser. No output repair or relaxed parser was used.

Validation scored 85% on a separate 20-case set excluded from training. The three-epoch candidate was declared in advance, with the adapter and confirmation dataset fixed before scoring. The confirmation set became exposed after this single evaluation. Artifact metadata uses canonical project identifiers; its current file checksums describe the exported evidence. Measurement receipts separately bind the unchanged model-visible inputs, raw answers, labels and scored decisions.

The 95% family-bootstrap interval is **75% to 100%**, with 10,000 replicates and seed 42. The point estimate exceeds the requested threshold. This small, developer-authored feasibility check does not establish population accuracy above 75%, independent generalization, advice robustness, explanation quality or preservation of general coding ability. Previous checks informed later experiments, so the best observed result also has selection bias.

## What changed

Starting from the saved v6 adapter, v7 trained on 150 exposed cases under neutral and misleading-opinion prompts. It used the category trace instruction, three epochs, learning rate 0.0001, verdict-token weight 10, rank 16 and all seven attention/MLP projection types. The original Aya parameters remained frozen. This is supervised LoRA adaptation. CircuitKIT's original numerical parity failure remains unresolved, and no causal circuit repair is claimed.

The model uses the original Aya revision `5062468bf9bc0c6035fd64e06274333ec127d980`. The adapter contains 448 finite FP32 tensors and 41,943,040 learned parameters, occupying 167,819,216 bytes. Its SHA-256 is `e86f9475e3c8871ba3c60a63b10704954f66edcb791bd9631866a07d82a9022f`.

## Inspect the evidence

- [Verified experiment ledger](AYA_PROGRESS.md) separates training, validation and prospective checks.
- [Raw confirmation answers](../runs/aya-beam-20261107/adapter-v7/study/confirmation/results.json) include exact prompts, source citations and parse results.
- [Frozen manifest](../runs/aya-beam-20261107/adapter-v7/study/frozen.json), [training protocol](../runs/aya-beam-20261107/adapter-v7/study/protocol.json) and [verification proof](../runs/aya-beam-20261107/adapter-v7/verification.json) bind the reported identities.
- [Reproduction source ZIP](../runs/aya-beam-20261107/adapter-v7-reproduction-source.zip) matches the exported source fingerprint and training-script hash.
- [Evidence ZIP](../runs/aya-beam-20261107/adapter-v7-evidence.zip) preserves the run metadata and both scored stages.
- [Selected pipeline](../configs/selected-review.json) and [model configuration](../configs/model-adapted.json) identify the retained local checkpoint, `checkpoints/aya-adapter-v7.safetensors`. The v6 checkpoint is retained separately for the earlier studies.

Both completed stage manifests passed integrity verification, covering 46 artifacts in total. Local verification re-parsed every answer, checked the exact prompts, reran all 20 executable oracles, matched the frozen source and protocol hashes, and verified the downloaded adapter bytes and tensor shapes. The real adapter was reloaded into the training process before scoring. A separate-process real-model reload and a dense merged checkpoint have not been verified.

The CPU-only `audit-adapter` command independently re-scores saved generation text and validates the selected tensor bytes. A fresh Python process also verified all 448 adapter tensor names, shapes, FP32 values and parameter counts against the pinned Aya architecture on the meta device. It loaded no base weights and performed no GPU inference; it does not establish behavioral reload equivalence. Independent label and explanation handoffs are ready, with deliberately incomplete review forms.

## Reproduce without retraining

From the project root on a compatible GPU host with the pinned research environment and model access, run the following. It makes no Beam provisioning or billing call. It generates a new saved neutral evaluation with the selected model and trace prompt.

```powershell
python scripts/evaluate_selected_review.py --output runs/selected-review-reproduction
python aawden.py verify --run runs/selected-review-reproduction
```

Recompute the existing ledger locally without loading Aya or using a GPU with `python scripts/report_aya_progress.py`.

## Resources

The v7 training task completed in 17 minutes 23 seconds. Beam reported **$24.00 remaining** after completion. The observed balance reduction across the last two jobs of that study was approximately $0.65, including billing lag. No Aawden tasks remained active at the end of that study, and all 20 known temporary job-secret names were cleaned up. The separate harder benchmark tracks the subsequent runs and the current $18 credit floor. [Harder review](RIGOROUS_REVIEW.md)
