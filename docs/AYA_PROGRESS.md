# Aya competence progress

Neutral competence on successive small single-author prospective checks. Earlier checks become exposed after scoring.

Scores below are recomputed from saved answers after manifest verification. Invalid responses remain errors. This ledger covers only `runs/aya-beam-20261107`; the harder benchmark runs live in `runs/aya-rigorous` and are described in `docs/RIGOROUS_REVIEW.md`.

| Experiment | Stage | Method | Cases | Strict balanced accuracy | Invalid | SQL | Access control |
|---|---|---|---:|---:|---:|---:|---:|
| adapter-v1 | development | adapted | 50 | 58.0% | 0 | 50.0% | 65.4% |
| adapter-v2 | development | adapted | 50 | 100.0% | 0 | 100.0% | 100.0% |
| adapter-v2 | confirmation | adapted | 40 | 72.5% | 0 | 80.0% | 65.0% |
| adapter-v3 | development | adapted | 90 | 100.0% | 0 | 100.0% | 100.0% |
| adapter-v3 | confirmation | adapted | 40 | 72.5% | 0 | 80.0% | 65.0% |
| adapter-v5 | development | adapted | 130 | 50.0% | 0 | 50.0% | 50.0% |
| adapter-v6 | development | adapted | 130 | 92.3% | 0 | 100.0% | 84.8% |
| adapter-v6 | confirmation | adapted | 20 | 70.0% | 0 | 90.0% | 50.0% |
| adapter-review-v1 | development | plain | 20 | 70.0% | 0 | 90.0% | 50.0% |
| adapter-review-v1 | development | trace | 20 | 80.0% | 0 | 90.0% | 70.0% |
| adapter-review-v1 | development | counterexample | 20 | 70.0% | 0 | 90.0% | 50.0% |
| adapter-review-v1 | development | contrasts | 20 | 70.0% | 0 | 90.0% | 50.0% |
| adapter-review-v1 | confirmation | trace | 20 | 75.0% | 0 | 100.0% | 50.0% |
| mental-review-v1 | development | mental | 20 | 75.0% | 0 | 100.0% | 50.0% |
| mental-review-v2 | development | mental | 20 | 75.0% | 0 | 100.0% | 50.0% |
| adapter-v7 | development | adapted | 20 | 85.0% | 0 | 100.0% | 70.0% |
| adapter-v7 | confirmation | adapted | 20 | 90.0% | 0 | 100.0% | 80.0% |

Best observed prospective score is **90.0%** in `adapter-v7` on 20 cases.

The strict above-75% target was met on this feasibility check.

The 95% family-bootstrap interval is 75.0% to 100.0% across 10 families. The point estimate meets the goal; the interval does not establish that population accuracy exceeds 75%.

- Adaptive sequence of experiments; best observed result has selection bias.
- Developer-authored synthetic families, not an independent final benchmark.
- Neutral accuracy does not establish opinion robustness, explanation quality, or general coding preservation.
- CircuitKIT numerical parity failed; no causal circuit repair is claimed.

The JSON companion records exact manifest hashes, model adapter hashes, evidence paths and recomputed counts.
