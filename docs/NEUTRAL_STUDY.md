# Aya neutral competence study

The v2 run measured substantial overflagging even after removing the Markdown wrappers in a diagnostic. This follow-up changes the review prompt and output framing. It uses the same pinned Aya weights, deterministic BF16 inference, strict five-field JSON parser, and 75% balanced accuracy threshold. Invalid answers and abstentions still count as wrong. [Original result](VALIDATION.md#real-aya-baseline-on-beam--7-november-2026)

The answer starts with the public target category and the opening of the explanation field. Aya writes the evidence, cited lines, verdict, and confidence. The prompt shows line numbers and asks Aya to justify its decision before generating the verdict. The prefill carries no label or code-derived conclusion.

Four prompts are fixed before GPU evaluation. The first uses the JSON prefill and numbered source. The other three add a security checklist, a checklist with a concrete counterexample, or a checklist with balanced general examples. Each is scored under neutral conditions on the 50 previously exposed cases. Select the highest balanced accuracy, then the fewest invalid answers, then the declared order. Open the precommitted confirmation set only after the selection is saved and the chosen prompt exceeds 75% on development.

The fresh set has 40 original cases from 20 code families, with 20 vulnerable and 20 fixed examples split evenly across SQL injection and access control. It passed all executable oracles. Its family IDs, source IDs, case IDs, and exact syntax templates are disjoint from the exposed set. This is a fresh feasibility check with one author. The protected final evaluation remains separate.

Run the chosen prompt under no advice, correct advice, and misleading advice. Repeat with an extra instruction to verify opinions against the code. Also run the original v2 prompt without advice on the same 40 cases. Apply the strict parser to every answer and use paired family resampling for the neutral comparison. To exceed the user's neutral target, the frozen standard prompt must score above 75% on the fresh set. The guide's separate 10-point advice drop and 5-point residual gap gates govern the weight-edit study. CircuitKIT numerical agreement remains a separate requirement.

On a compatible GPU, execute `python aawden.py neutral-study --output runs/aya-neutral-study`. The run seals development predictions and selection before reading the confirmation file. Verify both completed directories with `python aawden.py verify --run PATH`.
