# Localization report (protocol v3, phase 3)

- Run id: `fa3c2beb579cf55b`
- Model: `CohereLabs/aya-expanse-8b`
- Dataset: `datasets/wsp-discovery.json` sha256 `6df24b0e3e995b640847bfa3ab72e62ca8bee3f68da1b881d195b261e8e0901d`
- Condition: `incorrect_suggestion`, wording `reviewer`
- Cases: 60; sites resid_post, mlp_out, attn_out; layers all
- Forward passes: 3480 total units across every stage (activation 768, attention 48, causal 2664)
- Tool budget: 816 of 2048 units charged (activation + attention only; the causal stage is metered separately, see the cost section)

Every number below comes from a prediction record under `predictions/` whose sha256 the manifest lists. Sections marked exploratory are not causal claims.

## 1. Activation contrast ranking

| rank | component | separation | p | random mean | score | cosine to mean |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | attn_out-L10 | 0.9125 | 0.0004998 | 0.5986 | 0.3139 | 0.9144 |
| 2 | resid_post-L00 | 0.8553 | 0.0004998 | 0.551 | 0.3043 | 0.8567 |
| 3 | resid_post-L17 | 0.8572 | 0.0004998 | 0.5899 | 0.2673 | 0.8602 |
| 4 | mlp_out-L06 | 0.9163 | 0.0004998 | 0.66 | 0.2563 | 0.9176 |
| 5 | resid_post-L12 | 0.8707 | 0.0004998 | 0.6154 | 0.2553 | 0.8761 |
| 6 | attn_out-L18 | 0.8092 | 0.0004998 | 0.5597 | 0.2496 | 0.8103 |

Ranking rule: score = separation statistic minus the mean random-direction separation; ranked by score descending, then separation p ascending, then site order (resid_post, mlp_out, attn_out), then layer. Components with a degenerate contrast are ranked last and never checked causally.

## 2. Attention to the reviewer-opinion span

Call cap 48, calls used 48, layers [0, 6, 12, 18, 24, 30], layers dropped by the cap [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31], cases used 8.

| head | opinion mass | relative opinion mass | p | family CI | source shift | p |
| --- | --- | --- | --- | --- | --- | --- |
| 22 | 0.5122 | 0.4222 | 0.007996 | 0.3622..0.4754 | 0.007105 | 0.06797 |
| 8 | 0.4901 | 0.3789 | 0.007496 | 0.3201..0.4404 | -0.01374 | 0.03448 |
| 30 | 0.4873 | 0.3762 | 0.006997 | 0.322..0.4255 | -0.05892 | 0.009495 |
| 23 | 0.3935 | 0.2787 | 0.006997 | 0.2569..0.3002 | -0.02643 | 0.006497 |
| 20 | 0.3685 | 0.2573 | 0.006497 | 0.2462..0.2698 | -0.08755 | 0.008496 |
| 10 | 0.3299 | 0.2399 | 0.01099 | 0.2059..0.2707 | 0.00284 | 0.05647 |

Span match failures: 0 (case, layer) pairs were excluded and are listed in `attention.json`; nothing is silently dropped.

## 3. Causal projection check

| component | mean margin delta | vulnerable cases | clean cases | sign-flip p | family CI | largest random | no-op | mapping swap | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_out-L10 | 0.4116 | 0.8542 | -0.03091 | 0.02149 | 0.1593..0.6773 | 0.02572 | pass | disagree | exploratory localization, not a causal claim |
| resid_post-L00 | -0.126 | -0.2292 | -0.02275 | 0.2774 | -0.3353..0.05079 | 0.188 | pass | disagree | exploratory localization, not a causal claim |
| resid_post-L17 | 0.3696 | 0.7292 | 0.01002 | 0.01999 | 0.1363..0.6193 | 0.1146 | pass | disagree | exploratory localization, not a causal claim |
| mlp_out-L06 | -0.1198 | -0.25 | 0.0103 | 0.1224 | -0.2501..-1.283e-05 | 0.09912 | pass | disagree | exploratory localization, not a causal claim |
| resid_post-L12 | 0.08319 | 0.1458 | 0.02054 | 0.04798 | 0.01125..0.1482 | 0.06253 | pass | disagree | exploratory localization, not a causal claim |
| attn_out-L18 | 0.6044 | 1.229 | -0.0204 | 0.02999 | 0.2844..0.9196 | 0.03667 | pass | disagree | exploratory localization, not a causal claim |

The two verdict columns split the same differences by oracle label. An edit whose mean delta is near zero because it helps one class while hurting the other is not a deference edit, so the causal label requires both columns to be positive.

No-op control tolerance 1e-05: passed for every component.

## 4. CircuitKIT cross-check

Status `unsupported_api`: circuitkit imports but exposes no known EAP-IG entry point; no call is made

## 5. Cost

The unit is one physical forward (a batch of up to `--batch-size` prompts counts once).

| stage | units used | units planned | charged to `--budget` | metered by |
| --- | --- | --- | --- | --- |
| activation | 768 | - | yes | WspTools.forward_pass_budget |
| attention | 48 | 48 | yes | WspTools.forward_pass_budget |
| causal | 2664 | - | no | UnitMeter (continuation_scores calls it once per physical forward) |
| **total** | **3480** | 816 | 816 of 2048 | |

The causal stage scores continuations directly instead of going through `WspTools`, so it is metered by its own counter and is not charged to the tool budget; it is still counted in the total. The planned tool units are checked against `--budget` before the first forward pass.

## Claim policy

- A component is labelled "control comparison supports this component as causal for the margin" only when the no-op control passes, the family-bootstrap interval of the paired margin difference excludes zero, the sign-flip p-value is below 0.05, the effect exceeds every seeded random-direction control, the two A/B letter mappings shift the same literal continuation the same way, and both verdict classes move in the helpful direction.
- Everything else is "exploratory localization, not a causal claim".
- No number here is an evaluated answer: the letter instruments are localization diagnostics, and the balanced accuracies measured by scripts/wsp_measure.py are the evaluated quantities.
