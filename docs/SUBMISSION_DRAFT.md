# Aawden — submission draft

**Current status.** Aawden has a real Aya feasibility result, a saved trained adapter and a separate toy integration demonstration. The original Aya screen stopped the causal weight-edit study. A subsequent adaptation study reached 90% on a small fresh neutral check. Protected final evaluation and a demonstrated causal repair remain pending.

## Abstract

Security-review models should follow source evidence when a confident reviewer supplies an incorrect assessment. Aawden provides an executable benchmark and an auditable research pipeline for that failure.

Its 50 project-original SQL-injection and authorization fixtures include vulnerable/fixed pairs, explicit threat models and local behavioral oracles. Each case appears under neutral, correct-opinion and incorrect-opinion prompts. A stronger-prompt comparator prevents an edit from receiving credit for a failure a cheap instruction can solve.

The real-checkpoint implementation uses CircuitKIT for numerical parity, contrastive residual measurements and causal interventions on attention and MLP outputs. A bounded search projects selected output matrices, validates full generated verdicts, compares seeded matched random directions and saves the selected checkpoint. AuditKIT scores raw outputs through a strict semantic scorer. Aawden records paired family intervals, regression comparisons, checkpoint hashes and fresh-process reload checks.

Original Aya Expanse 8B completed 300 answers across 50 cases and 24 families on a Beam RTX 4090. AuditKIT scored the raw generations. The prespecified parser accepted 3 of 300 answers because Aya wrapped 297 in Markdown. In a separate diagnostic, removing only those fences gave the standard prompt 48% balanced accuracy without advice, 82% with correct advice, and 0% with misleading advice. The stronger prompt reached 18% with misleading advice and 48% without advice. Both fell below the 75% neutral threshold for investigating a weight edit. [Measured results](../EXPLANATION.MD#what-aya-did-in-the-first-gpu-run)

CircuitKIT loaded the same weights and matched all 12 top-token choices. The BF16 score differences exceeded the recorded numerical tolerances. The two gates stopped the intervention study. [Comparison record](../runs/aya-beam-20261107/sequential-evidence/parity.json)

The separately labeled supervised LoRA branch trained a rank-16 adapter while freezing Aya's original weights. Its final candidate used 150 exposed training cases, scored 85% on 20 separate validation cases and then froze before the next prospective check. It scored **90% strict balanced accuracy on 20 fresh neutral cases**, with no invalid JSON. SQL scored 10/10 and access control 8/10. The family-bootstrap interval is 75–100%, and the cases share a single author. This meets a competence feasibility target and does not establish the original opinion-robustness endpoint or a causal mechanism. The retained adapter, manifests, frozen source and all attempts are documented in [the verified result](BEST_RESULT.md).

## Before submitting

- Retain both sponsor-tool execution records and resolve the CircuitKIT numerical mismatch before claiming a causal repair.
- Independently review fixture truth conditions and add genuinely new final families.
- Report D versus B and C versus A, neutral/correct controls, all random seeds, invalids, uncertainty and denominators.
- Run the frozen coding comparison and complete the blinded review of cited evidence.
- Publish permitted models/deltas, datasets, logs, traces and benchmark artifacts to the required public Hugging Face collections.
- Add actual public code/demo and write-up links.
- State negative findings and resource use.

The [official submission page](https://hackathon.lexsi.ai/) currently specifies November 14, 2026 end of day AoE. This corresponds to November 15 at 12:59 Paris time; check the live portal before submitting. Registration is separate and has not been verified by this implementation.

