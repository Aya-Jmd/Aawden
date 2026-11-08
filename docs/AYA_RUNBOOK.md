# Run the real Aya pilot

The project `.env` token authorized the pinned Aya checkpoint. The first full baseline ran on a Beam serverless RTX 4090 in BF16 with all four weight shards. When launching on Beam, pass only the model read token to the pilot as `HF_TOKEN` through a workspace secret. The laptop's saved login is separate from that secret. [Baseline manifest](../runs/aya-beam-20261107/evidence/baseline/manifest.json) · [Beam secret setup](https://docs.beam.cloud/v2/environment/secrets)

## Resource request

| Resource | Pilot request |
|---|---|
| GPU | One 80 GB A100 or H100 is the conservative choice. A BF16-capable 48 GB card is a possible starting point if the measured conversion peak fits. |
| Host RAM | 64 GB recommended for weight loading and conversion. |
| Disk | 80 GiB free for the base cache, original and edited checkpoint exports, environment, and logs. Confirm scratch quotas if the cache and run directory are on different filesystems. |
| Pilot allocation | Budget 2–4 GPU-hours. The run stops early if the baseline, mechanism, or candidate gate fails. |
| Final allocation | Estimate another 10–20 GPU-hours for 300 cases with all eight arms. Use the observed baseline throughput before booking this. |

Two BF16 copies of 8,028,033,024 parameters require 29.91 GiB for weights alone. The parity stage also needs conversion buffers and activations. The default runner checks for native BF16 and at least 42 GiB of free VRAM. With `--sequential-gpu`, it checks for 21 GiB and keeps one model on the GPU at a time, holding the other model's weights in host memory during conversion. Both values are planning thresholds. Use 64 GiB of host RAM for the sequential path.

The pilot has 50 baseline cases, three suggestion conditions and two baseline prompts, for 300 generations. If the screen passes, prompt selection adds 60 generations on 10 validation cases. Discovery uses 30 development cases and tests counterbalanced continuation scores with attention and MLP interventions. Edit selection uses 10 validation cases, up to four targeted edits, three matched controls, and one unedited comparison. A full 300-case final run has 7,200 generations for the primary and control comparisons, plus 600 for the additional authority wording and two generations per independent coding case. Use measured throughput from the baseline and discovery stages to refine the estimates.

## Launch on the authorized host

From this checkout on a compatible Linux GPU allocation:

~~~bash
uv venv .venv-research --python 3.11
uv pip sync --python .venv-research/bin/python requirements-research.lock
. .venv-research/bin/activate
python scripts/run_aya_pilot.py --preflight-only --output runs/aya-pilot
python scripts/run_aya_pilot.py --output runs/aya-pilot
~~~

The preflight fetches a small configuration file to check access. The second command checks parity and the unchanged checkpoint reload, measures the baseline, and runs prompt selection, discovery, and edit validation if warranted. It checks a selected checkpoint again in a fresh process. It writes stage logs and a throughput estimate in `baseline-budget.json`. Validation resumes from verified saved answers. A failed stage can be retried once with the same inputs and retains its attempt history.

If the baseline already ran on a separate GPU job, add `--reuse-baseline PATH` with its completed run directory. The runner checks the saved artifacts and confirms that the protocol, cases, prompt version, model configuration, and actual checkpoint hashes match before using those scores.

A scheduler job can execute the same two commands on its allocated device. Supply a matching CUDA driver/runtime for the locked PyTorch build. Use a new output directory after changing code, protocol, model configuration or fixture files. If a forced kill leaves `.pilot-running` or `candidates/.search-running`, confirm its recorded PID has stopped before clearing that lock and resuming.

The runner never provisions hardware, changes accounts, or submits artifacts. When a candidate is ready, use the [external evaluator instructions](FINAL_EVALUATOR.md).

## Interpret the outcome

- **Negative screen or prompt suffices:** report the strongest original prompt as the practical alternative; do not continue editing to manufacture a gain.
- **No supported causal direction:** record the null patch result and the random controls.
- **No eligible validation candidate:** preserve the trial ledger; the edit is unproven.
- **Candidate ready:** request the external commitments, freeze once, and measure the final comparison without further tuning.

The local synthetic architecture check is for toolkit compatibility. Its randomly initialized weights and generated output provide no security-performance evidence about Aya Expanse.

