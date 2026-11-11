# Beam GPU launcher

Runs pinned Aawden commands on a Beam serverless GPU in Beam workspace 2 only.

- `scripts/beam_launch.py`: local CLI (`launch`, `status`, `fetch`, `smoke`).
- `scripts/beam_entry.py`: Beam image, volume, secret, and the `run_command` entry function. Its
  `check` subcommand is the in-container readiness probe.
- `configs/beam.json`: volume, secret, default resources, price table, and the smoke preset.
- `.beamignore` (repository root): upload exclusions.
- `tests/test_beam_launch.py`: offline tests (no Beam calls).

## Usage

Run from the repository root with an interpreter that has `beam-client` installed. This checkout
used a venv outside the repo (`uv pip install --python <venv>/bin/python beam-client`, SDK 0.2.239).

```bash
set -a && . ./.env && set +a      # BEAM_TOKEN_2 is read from here; values are never printed

# One command on a GPU, waits for it. --max-cost-usd is required.
python scripts/beam_launch.py launch --gpu RTX4090 --timeout 1800 --max-cost-usd 0.80 \
    --output-subdir wsp-v3/example-20261110 -- python aawden.py doctor --config configs/model.json

python scripts/beam_launch.py status <task_id>          # Beam task state and start/end times
python scripts/beam_launch.py fetch wsp-v3/example-20261110 runs/wsp-v3/example-20261110

# Verification job: doctor, then the torch/toolkit probe. Writes a new volume folder each run
# and fetches it to runs/wsp-v3/beam-smoke by default.
python scripts/beam_launch.py smoke
```

Options: `--cpu` (default 2), `--memory` (default `32Gi`), `--timeout` (default 1800; at most 2100
per `configs/compute-budget.json`), `--balance-cents` (runs `security_wsp.budget.reserve_job` with
the $18 floor and $1 margin, and refuses if the balance cannot cover it), `--python`.

Inside a command, `python` and `{python}` both mean the container interpreter.

Use a new `--output-subdir` for every launch. See "Known issues".

## Credentials and isolation

- Only `BEAM_TOKEN_2` is used. It is read from the environment or the repo `.env`.
- The launcher refuses to run if `BEAM_TOKEN_2` is missing or equals `BEAM_TOKEN` (the old
  workspace). It never falls back to `BEAM_TOKEN`.
- The Beam CLI and SDK run with a temporary `HOME` holding only a workspace-2 `config.ini`
  (`[default]`, `token`, `gateway_host = gateway.beam.cloud`, `gateway_port = 443`). The directory
  is deleted when the call returns. `~/.beam/config.ini` is not read or written. `BEAM_*`,
  `BETA9_*`, and `CONFIG_PATH` are removed from the child environment.
- The job receives the workspace secret `AAWDEN_AYA_HF_TOKEN` and exports it as `HF_TOKEN`. The
  name is removed from the job environment after mapping. Logs never contain the value.

## Uploads

Beam syncs the working directory. The root `.beamignore` lists `.env`, `.env.*`, `.git`, virtual
environments, `runs/`, `checkpoints/`, `dist/`, `build/`, `.vendor/`, caches, and weight files. The
SDK writes its own default `.beamignore` only when none exists, so the file keeps the SDK's usual
cache exclusions and adds the secret and weight exclusions. Sync size for the verification job was
2.1 MB. The probe also reports any `.env*` file it finds under the job's working directory, and
sets `ok` to false if there is one.

## Image

Built on Beam's `beta9-runner` base with Python 3.11 (`python_version` in `configs/beam.json`).

- **Python packages:** all 167 exact pins in `requirements-research.lock`, installed with uv in one
  step. This includes `torch==2.15.0` (CUDA 13.2 wheels from PyPI), `transformers==5.22.1`,
  `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.33.0`, `numpy==2.4.6`,
  `tokenizers==0.23.3`, `transformer-lens==3.8.0`, and `sentence-transformers==6.2.0`. The two git
  lines in the lock are not used.
- **Sponsor toolkits:** CircuitKIT `e71d5e17fe6e2abfde47ec9a7fbaf31649601b78` (installed as 0.2.0) and
  AuditKIT `71316da345784ff220d5ed59337441a59ab31526` (installed as 1.1.3). Each archive is downloaded
  from codeload.github.com, its SHA-256 is checked against `dependency-sources.lock.json`, and it is
  installed with `pip -c` on the lock, so the install cannot move a pin.
- **Gate:** the same build step checks that every lock pin is installed at its exact version and
  fails the build otherwise. Build output: `lock pin check: 167 pins; mismatches: []`. It then runs
  `pip check`, which is informational. Its current output reports two conflicts that the lock does
  not resolve: `pygobject` needs `pycairo` (not installed), and `betterproto-beta9` wants
  `python-dateutil<3` while the lock pins 4.0.0.
- **Environment:** `HF_HOME=/volumes/aawden-v3/hf-cache`, `PYTHONUNBUFFERED=1`,
  `TOKENIZERS_PARALLELISM=false`.
- **Build time:** 48.7 s build plus 37.3 s publish (3.47 GiB, 15 layers) for the first full build,
  so about 1 min 33 s to "Image ready". Later launches reuse the cached image (under 1 s).

The constraints file and the sponsor installs are one shell step, because the builder does not keep
`/tmp` between RUN steps. Two earlier image attempts failed, and both are recorded here:

1. Top-level pins only: the sponsor `pip install` pulled `sentence-transformers` 6.2.0 and upgraded
   `transformers` to 6.2.0, which the pin forbids. The full-lock install fixed it.
2. Constraints file in its own RUN step: `pip -c` could not open it in the next step.

## Volume and outputs

- Volume `aawden-v3`, mounted at `/volumes/aawden-v3`.
- Each job writes `aawden-v3/<output_subdir>/job.log` (streamed as the command runs), `result.json`
  (exit code, duration, command, task id, log tail), and `check.json` if the probe ran. Files the
  command writes to `$AAWDEN_BEAM_OUTPUT` (`runs/<output_subdir>` inside the job) are copied to the
  volume when the command ends.
- `fetch` lists the folder with `beam ls`, copies each file with `beam cp` from the folder that
  contains the target, and checks each downloaded size against the listing.

## Cost model and guard

| Item | Rate | Source |
|---|---|---|
| RTX4090 serverless GPU | $0.69/hr | Beam public pricing page, read 2026-11-10 by web search excerpt (the operator reported $0.66; the higher public figure is used) |
| RTX5090 serverless GPU | $1.09/hr | same page |
| A10G | $1.10/hr, **UNVERIFIED** | placeholder. The public page lists RTX4090 and RTX5090 as the serverless GPUs; the operator said A10G is available. |
| vCPU | $0.18972/hr | derived from the page's worked example (4 vCPU = $0.76) |
| RAM | $0.0198/GiB-hr | derived from the same example (16 GiB = $0.32) |

Estimate for the guard: `(GPU + cpu x vCPU rate + GiB x RAM rate) x timeout / 3600`. For the
default RTX4090 with 2 vCPU and 32 GiB, that is $1.703/hr. The launch refuses if the estimate is
above `--max-cost-usd`. The estimate is a worst case, assuming the job runs to its timeout. A
GPU-only rate would understate it, because serverless bills CPU and RAM too.

Measured spend, from Beam task timestamps (`started_at` to `ended_at`), at the all-in rate:

| Task | Result | Billed seconds |
|---|---|---|
| `4a298a99-8242-42b6-a038-2d9ade8f205b` | cancelled before it started (first image) | 0 |
| `e053851d-c8c8-4c77-a61a-e3912c097a9c` | exit 1, probe import-order bug | 16.7 |
| `3ee1d042-f7bf-4af7-b703-e0821c22db51` | exit 0, probe ok, fetched stale copies | 13.5 |
| `a9b8a978-5a1f-4484-acf0-2e08378e0177` | exit 0, probe ok, verified (canonical) | 21.4 |

Total 51.6 billed seconds, about $0.024 at the all-in rate ($0.010 at GPU-only). Beam's invoice is
not checked here: UNKNOWN. Whether image builds are billed is UNKNOWN; the task records do not show it.
Queueing: the first GPU task waited about 7 minutes in PENDING (created 09:28:34, started 09:35:31
UTC). The later tasks started within 3 seconds.

The `$18` floor and `$1` margin apply only when `--balance-cents` is given. The workspace balance was
not supplied, so that check was not exercised against a live balance.

## Verification results

Canonical run: `a9b8a978-5a1f-4484-acf0-2e08378e0177`, RTX4090, 2 vCPU, 32 GiB, timeout 600 s,
volume folder `aawden-v3/wsp-v3/beam-smoke-v2`, fetched to `runs/wsp-v3/beam-smoke/`
(`check.json`, `job.log`, `result.json`). The command was
`sh -c "python aawden.py doctor --config configs/model.json && python scripts/beam_entry.py check"`.

- `doctor`: `core_ready` true, `research_runtime_ready` true, `local_checkpoint` false (no weights
  downloaded), `model_access_verified` false (not checked by design).
- `check` (`ok: true`):
  - torch 2.15.0+cu132, CUDA available, device NVIDIA GeForce RTX 4090, bf16 supported, 23.52 GiB
  - transformers 5.22.1, transformer-lens 3.8.0, accelerate 1.15.0, safetensors 0.8.0,
    huggingface-hub 1.33.0, numpy 2.4.6 (all import)
  - circuitkit 0.2.0 and auditkit 1.1.3 import
  - `dotenv_files_in_workdir`: none (no `.env`, `.env.*`, or `.env.example` in the job's checkout)
  - `AAWDEN_AYA_HF_TOKEN` absent from the job environment after mapping; `HF_TOKEN` present
  - volume `/volumes/aawden-v3` exists; it contains `wsp-v3`
- Job exit 0 in 18.8 s (the job's own timer); billed 21.4 s.

## Known issues

- **Stale downloads.** During testing, `beam cp` returned an older copy of `check.json` and
  `result.json` from a previous run in the same folder. The volume listing showed the new sizes,
  and `job.log` (appended from an empty file) came through correctly. The cause was not isolated.
  Mitigations: `smoke` now uses a UTC-stamped folder by default; `fetch` fails if a downloaded size
  differs from the listing; use a fresh `--output-subdir` for every launch.
- **Folder copy.** `beam cp` cannot copy a folder or a glob, and it only writes inside its working
  directory. `fetch` handles both. A listing entry without a `size` is treated as a sub-folder and
  walked recursively. That rule was not exercised, because the outputs so far have no sub-folders.
- **A10G price** is a placeholder and must be checked before any A10G launch.
- **Balance:** `--balance-cents` was not exercised against a live balance.
- **Cleanup:** the cancelled and failed attempts left records under `runs/wsp-v3/beam-launches/`.
  They are kept as evidence. `runs/` is git-ignored.

## Tests

`tests/test_beam_launch.py`: 38 offline tests (`python -m pytest -q -p no:cacheprovider
tests/test_beam_launch.py`). They cover the cost guard (including CPU and RAM), the timeout and GPU
policy, argument parsing, the refusal of `BEAM_TOKEN`, the temporary-HOME config, the `.beamignore`
exclusions, the match between image pins and the research lock, the sponsor SHA-256 commands and
constraints, the probe import order, the local `execute_job` path (secret mapping, output copy, exit
codes), the `.env` detection, the budget refusal, and fetch size verification and recursion.
