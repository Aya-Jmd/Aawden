# Three-minute Aawden demonstration

**Say first:** This local view is a constructed toy instrumentation demo. It does not show a measured Cohere improvement.

1. **The user problem — 25 seconds.** Open Overview. Explain that code-review models can defer to a confident wrong reviewer. The goal is to follow code evidence while retaining neutral competence.
2. **The evidence — 45 seconds.** Open Benchmark. Select a vulnerable SQL fixture. Switch between neutral, correct and misleading claims. Show that the code is unchanged. Expand the threat model and oracle; the local SQLite test reproduces the violation. Then choose its fixed pair.
3. **The mechanism — 35 seconds.** Explain the toy's known opinion feature and stored weight. Show original/strong versus edited/strong metrics. State that real CircuitKIT residual contrasts and causal tests are implemented but have not executed on Aya here.
4. **The controls — 35 seconds.** Open Experiment ledger. Show every strength, validation selection, three seeded random controls, no-op comparison and update counts. Explain why D versus B is the practical comparison.
5. **Reproducibility — 20 seconds.** Open Artifacts. Click Verify artifact integrity. Download the report or evidence bundle. Show the fresh-process reload record.
6. **What real Aya did — 20 seconds.** Open the measured results. Aya answered 300 times on the same code fixtures. The raw JSON check rejected 297 Markdown-wrapped answers. A separate diagnostic recovered the verdicts and found 48% balanced accuracy without advice and 0% with misleading advice under the standard prompt. The planned neutral threshold stopped the edit study. [Measured results](../EXPLANATION.MD#what-aya-did-in-the-first-gpu-run)

Keep the evidence labels on each screen. The local dashboard shows the constructed toy. The measured Aya results cover the exposed development fixtures and have their own report, raw answers and manifest.

