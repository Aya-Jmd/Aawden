
### Coding ability (public benchmarks, greedy)

| arm | benchmark | pass@1 | 95% CI |
|---|---|---|---|
| baseline_humaneval | humaneval | 54/164 = 0.3293 | [0.262, 0.404] |
| sft_humaneval | humaneval | 51/164 = 0.311 | [0.245, 0.385] |

### Review skill and deference (plain-pooled, 129 cases)

| arm | instrument | solved unaided | competent? | flip rate | 95% CI |
|---|---|---|---|---|---|
| baseline | json_prefill | 67/129 [0.434, 0.604] | no (at chance) | 1.0 | [0.946, 1.0] |
| baseline | letters_a_vulnerable | 64/129 [0.411, 0.581] | no (at chance) | 1.0 | [0.943, 1.0] |
| baseline | letters_b_vulnerable | 64/129 [0.411, 0.581] | no (at chance) | 1.0 | [0.943, 1.0] |
| sft | json_prefill | 64/129 [0.411, 0.581] | no (at chance) | 1.0 | [0.943, 1.0] |
| sft | letters_a_vulnerable | 64/129 [0.411, 0.581] | no (at chance) | 1.0 | [0.943, 1.0] |
| sft | letters_b_vulnerable | 64/129 [0.411, 0.581] | no (at chance) | 1.0 | [0.943, 1.0] |
