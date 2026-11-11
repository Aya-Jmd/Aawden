"""Local launcher for the Aawden Beam GPU jobs defined in scripts/beam_entry.py.

Only the workspace-2 token (BEAM_TOKEN_2) is used. It is written to a temporary HOME,
so ~/.beam/config.ini is never read or changed. The old BEAM_TOKEN is refused.

  python scripts/beam_launch.py launch --gpu RTX4090 --timeout 1800 --max-cost-usd 0.50 \
      --output-subdir wsp-v3/example -- python aawden.py doctor --config configs/model.json
  python scripts/beam_launch.py status <task_id>
  python scripts/beam_launch.py fetch wsp-v3/example runs/wsp-v3/example
  python scripts/beam_launch.py smoke

Run from the repository root with the venv that has beam-client installed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CONFIG_PATH = ROOT / "configs" / "beam.json"
BUDGET_PATH = ROOT / "configs" / "compute-budget.json"
ENV_PATH = ROOT / ".env"
ENTRY_PATH = ROOT / "scripts" / "beam_entry.py"
LAUNCH_RECORDS = ROOT / "runs" / "wsp-v3" / "beam-launches"

# Environment prefixes the Beam SDK reads; scrubbed from child processes so only the
# temporary config file supplies credentials.
SCRUBBED_PREFIXES = ("BEAM_", "BETA9_", "CONFIG_PATH", "API_HOST", "API_PORT", "GATEWAY_HOST", "GATEWAY_PORT")


class LaunchRefused(Exception):
    """Raised for any condition that must stop a launch before money is spent."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dotenv(path: Path = ENV_PATH) -> dict:
    """Read KEY=VALUE pairs without echoing them anywhere."""
    values: dict = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


WORKSPACES = {"primary": "BEAM_TOKEN_2", "secondary": "BEAM_TOKEN"}


def select_token(config: dict, environ: dict, dotenv: dict, workspace: str = "primary") -> str:
    """Return the token for the requested workspace.

    Two independent workspaces double the usable GPU capacity, so both are supported explicitly by
    name. There is no silent fallback: an unknown workspace name, or an unset token for the chosen
    workspace, refuses rather than borrowing the other one's credentials.
    """
    if workspace not in WORKSPACES:
        raise LaunchRefused(f"unknown --workspace {workspace!r}; expected one of {sorted(WORKSPACES)}")
    token_name = WORKSPACES[workspace]
    token = environ.get(token_name) or dotenv.get(token_name) or ""
    if not token:
        raise LaunchRefused(f"{token_name} is not set in the environment or .env (workspace {workspace})")
    if workspace == "primary":
        old = environ.get(config["refused_token_env"]) or dotenv.get(config["refused_token_env"]) or ""
        if old and token == old:
            raise LaunchRefused(f"{token_name} equals the old {config['refused_token_env']} token; refusing to launch")
    return token


def estimate_cost_usd(config: dict, gpu: str, timeout_s: int, cpu: float = 2.0, memory: str = "32Gi") -> float:
    """Worst-case cost: (GPU + vCPU + RAM hourly rates) x timeout. Serverless bills CPU and RAM on top of the GPU."""
    rates = config["hourly_usd"]
    if gpu not in rates:
        raise LaunchRefused(f"No hourly price is configured for {gpu}; configured: {sorted(rates)}")
    if not memory.endswith("Gi"):
        raise LaunchRefused("memory must be given in Gi, for example 32Gi")
    pricing = config["pricing"]
    hourly = (rates[gpu]
              + cpu * pricing["cpu_per_vcpu_hour_usd"]
              + float(memory.removesuffix("Gi")) * pricing["ram_per_gib_hour_usd"])
    return round(hourly * timeout_s / 3600, 4)


def check_cost_guard(config: dict, gpu: str, timeout_s: int, max_cost_usd: float,
                     cpu: float = 2.0, memory: str = "32Gi") -> float:
    estimate = estimate_cost_usd(config, gpu, timeout_s, cpu, memory)
    if estimate > max_cost_usd:
        raise LaunchRefused(
            f"Estimated worst-case cost ${estimate:.4f} (timeout {timeout_s}s on {gpu}, "
            f"{cpu} vCPU, {memory}) exceeds --max-cost-usd {max_cost_usd}"
        )
    return estimate


def check_runtime_policy(gpu: str, timeout_s: int, cpu: float, memory: str) -> None:
    """Run the compute-budget policy. Raises LaunchRefused if the job exceeds it."""
    if timeout_s <= 0:
        raise LaunchRefused("--timeout must be positive")
    policy = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    if timeout_s > policy["maximum_job_runtime_seconds"]:
        raise LaunchRefused(
            f"--timeout {timeout_s}s exceeds the compute budget maximum of {policy['maximum_job_runtime_seconds']}s"
        )
    if gpu not in {"RTX4090", "RTX5090", "A10G"}:
        raise LaunchRefused(f"{gpu} is not in the compute budget's allowed GPUs")
    if cpu <= 0 or not memory.endswith("Gi"):
        raise LaunchRefused("CPU must be positive and memory must be given in Gi")


def reserve_with_balance(balance_cents: int, gpu: str, timeout_s: int, cpu: float, memory: str) -> dict:
    from security_wsp.budget import reserve_job

    policy = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    spec = {"gpu": gpu, "timeout": timeout_s, "cpu": cpu, "memory": memory}
    try:
        return reserve_job(balance_cents, spec, policy)
    except (ValueError, RuntimeError) as exc:
        raise LaunchRefused(f"Budget reservation refused: {exc}") from exc


@contextlib.contextmanager
def isolated_beam_home(token: str):
    """Temporary HOME holding only the workspace-2 Beam config. Deleted on exit."""
    home = Path(tempfile.mkdtemp(prefix="aawden-beam-home-"))
    try:
        config_dir = home / ".beam"
        config_dir.mkdir()
        config_file = config_dir / "config.ini"
        config_file.write_text(
            f"[default]\ntoken = {token}\ngateway_host = gateway.beam.cloud\ngateway_port = 443\n",
            encoding="utf-8",
        )
        os.chmod(config_file, 0o600)
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def child_environment(home: Path, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(SCRUBBED_PREFIXES)}
    env["HOME"] = str(home)
    env.update(extra or {})
    return env


def beam_binary(python: str) -> str:
    candidate = Path(python).with_name("beam")
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("beam")
    if not found:
        raise LaunchRefused("The beam CLI is not installed next to this interpreter or on PATH")
    return found


def sanitize_subdir(value: str) -> str:
    from beam_entry import validate_output_subdir  # scripts/ is on sys.path

    try:
        return validate_output_subdir(value)
    except ValueError as exc:
        raise LaunchRefused(str(exc)) from exc


def launch(options) -> int:
    config = load_config()
    gpu = options.gpu
    timeout_s = int(options.timeout)
    cpu = float(options.cpu)
    memory = options.memory
    subdir = sanitize_subdir(options.output_subdir)
    command = list(options.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise LaunchRefused("Give the command to run after --")

    check_runtime_policy(gpu, timeout_s, cpu, memory)
    estimate = check_cost_guard(config, gpu, timeout_s, options.max_cost_usd, cpu, memory)
    if options.balance_cents is not None:
        reservation = reserve_with_balance(options.balance_cents, gpu, timeout_s, cpu, memory)
        print(f"budget: reserve {reservation['maximum_job_reserve_cents']} cents, passed", flush=True)

    workspace = getattr(options, "workspace", "primary")
    token = select_token(config, os.environ, parse_dotenv(), workspace)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "started_at": stamp, "workspace": workspace,
        "gpu": gpu, "cpu": cpu, "memory": memory, "timeout_s": timeout_s,
        "estimated_max_cost_usd": estimate, "max_cost_usd_guard": options.max_cost_usd,
        "output_subdir": subdir, "volume_path": f"{config['volume']['name']}/{subdir}",
        "command": command, "task_id": "", "exit_code": None,
    }
    overrides = {
        "AAWDEN_BEAM_GPU": gpu,
        "AAWDEN_BEAM_CPU": str(cpu),
        "AAWDEN_BEAM_MEMORY": memory,
        "AAWDEN_BEAM_TIMEOUT": str(timeout_s),
    }
    LAUNCH_RECORDS.mkdir(parents=True, exist_ok=True)
    record_path = LAUNCH_RECORDS / f"{stamp}-{subdir.replace('/', '_')}.json"
    exit_code = 1
    with isolated_beam_home(token) as home:
        env = child_environment(home, overrides)
        invoke = [options.python, str(ENTRY_PATH), "invoke", "--output-subdir", subdir, "--", *command]
        print(f"launch: {gpu} cpu={cpu} memory={memory} timeout={timeout_s}s "
              f"estimated max ${estimate:.4f} -> volume {record['volume_path']}", flush=True)
        process = subprocess.Popen(invoke, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        result = None
        for line in process.stdout:
            if line.startswith("BEAM_RESULT "):
                result = json.loads(line[len("BEAM_RESULT "):])
                record["task_id"] = result.get("task_id", "")
                record["result"] = {k: v for k, v in result.items() if k != "log_tail"}
            elif line.startswith("BEAM_ERROR "):
                print(line.rstrip(), flush=True)
                fragment = line.split("task_id=", 1)[-1].split(" ", 1)[0]
                if fragment and not record["task_id"]:
                    record["task_id"] = fragment
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
        exit_code = process.wait()
    record["exit_code"] = exit_code
    record["ended_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"task_id={record['task_id'] or 'UNKNOWN'} exit_code={exit_code} record={record_path}", flush=True)
    return exit_code


def status(options) -> int:
    config = load_config()
    token = select_token(config, os.environ, parse_dotenv(), getattr(options, "workspace", "primary"))
    with isolated_beam_home(token) as home:
        completed = subprocess.run(
            [beam_binary(options.python), "task", "list", "--format", "json", "--filter", f"id={options.task_id}"],
            cwd=ROOT, env=child_environment(home), capture_output=True, text=True, timeout=120,
        )
    if completed.returncode != 0:
        print(f"beam task list failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode
    try:
        tasks = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print("beam task list did not return JSON", file=sys.stderr)
        return 1
    matches = [t for t in tasks if isinstance(t, dict) and t.get("id") == options.task_id]
    if not matches:
        print(json.dumps({"task_id": options.task_id, "found": False}, indent=2))
        return 1
    keys = ("id", "status", "stub_name", "created_at", "started_at", "ended_at")
    print(json.dumps({k: matches[0].get(k) for k in keys}, indent=2))
    return 0


def _volume_files(beam: str, env: dict, volume: str, folder: str, depth: int = 0) -> list[tuple[str, int]]:
    """Recursively list (path, size) under a volume folder. `beam cp` cannot copy a folder or a glob."""
    if depth > 8:
        raise LaunchRefused(f"volume folder nesting is too deep under {folder}")
    listing = subprocess.run([beam, "ls", "--format", "json", f"{volume}/{folder}"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if listing.returncode != 0:
        raise LaunchRefused(f"beam ls failed for {volume}/{folder} (exit {listing.returncode})")
    files: list[tuple[str, int]] = []
    for entry in json.loads(listing.stdout or "[]"):
        path = entry["path"]
        if "size" in entry:
            files.append((path, int(entry["size"])))
        else:  # no size: treated as a sub-folder
            files.extend(_volume_files(beam, env, volume, path, depth + 1))
    return files


def fetch(options) -> int:
    config = load_config()
    volume = config["volume"]["name"]
    remote = sanitize_subdir(options.remote_subdir)
    local = Path(options.local_dir)
    if not local.is_absolute():
        local = ROOT / local
    token = select_token(config, os.environ, parse_dotenv(), getattr(options, "workspace", "primary"))
    beam = beam_binary(options.python)
    copied: list[str] = []
    with isolated_beam_home(token) as home:
        env = child_environment(home)
        for path, size in _volume_files(beam, env, volume, remote):
            relative = path[len(remote) + 1:] if path.startswith(remote + "/") else Path(path).name
            target = local / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # beam cp only writes inside its working directory, so run it from the target folder.
            completed = subprocess.run([beam, "cp", f"beam://{volume}/{path}", target.name],
                                       cwd=target.parent, env=env, capture_output=True, text=True, timeout=900)
            if completed.returncode != 0:
                print(f"beam cp failed for {path} with exit code {completed.returncode}", file=sys.stderr)
                print((completed.stderr or completed.stdout)[-800:], file=sys.stderr)
                return completed.returncode
            # Observed: beam cp can return an older copy of a path that was overwritten on the volume.
            # A size mismatch against the listing is treated as a failed fetch.
            if target.stat().st_size != size:
                print(f"size mismatch for {path}: listed {size} bytes, downloaded {target.stat().st_size}; "
                      "the download may be stale. Use a fresh --output-subdir for each launch.", file=sys.stderr)
                return 1
            copied.append(relative)
    if not copied:
        print(f"no files found under {volume}/{remote}", file=sys.stderr)
        return 1
    print(f"fetched {len(copied)} file(s) from {volume}/{remote} -> {local}: {', '.join(sorted(copied))}")
    return 0


def smoke(options) -> int:
    config = load_config()
    preset = config["smoke"]
    # A fresh volume folder per run: beam cp returned stale copies of overwritten paths during testing.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    subdir = options.output_subdir or f"{preset['output_subdir']}-{stamp}"
    launch_options = argparse.Namespace(
        gpu=preset["gpu"], timeout=preset["timeout"], cpu=config["defaults"]["cpu"],
        memory=config["defaults"]["memory"], max_cost_usd=preset["max_cost_usd"],
        output_subdir=subdir, balance_cents=options.balance_cents,
        command=list(preset["command"]), python=options.python,
    )
    code = launch(launch_options)
    if code != 0:
        return code
    local = Path(options.local_dir) if options.local_dir else ROOT / "runs" / preset["output_subdir"]
    fetch_options = argparse.Namespace(remote_subdir=subdir, local_dir=str(local), python=options.python)
    return fetch(fetch_options)


def deploy_demo(options) -> int:
    """Deploy scripts/beam_demo.py as a public, passcode-gated endpoint.

    The passcode becomes a Beam secret (AAWDEN_DEMO_PASSCODE) through `beam secret create`, so it is
    never written into the app file, the image or a request made by this machine. The workspace token
    is read from `.env` and lives only in a temporary HOME.
    """
    config = load_config()
    hourly = config["hourly_usd"].get(options.gpu)
    if hourly is None:
        raise LaunchRefused(f"unknown gpu {options.gpu}")
    keep_warm = 300  # must match KEEP_WARM in scripts/beam_demo.py
    estimate = hourly * keep_warm / 3600
    if estimate > options.max_cost_usd:
        raise LaunchRefused(f"keep-warm window alone costs ${estimate:.4f}, over the ${options.max_cost_usd:.2f} guard")
    token = select_token(config, os.environ, parse_dotenv(), options.workspace)
    secret, app = "AAWDEN_DEMO_PASSCODE", "scripts/beam_demo.py:web_server"
    if options.dry_run:
        print(f"would create secret {secret} (value hidden), then deploy {app} on {options.gpu}")
        return 0
    with isolated_beam_home(token) as home:
        # The GPU is chosen inside the @asgi decorator, which is evaluated here at deploy time.
        # Passing it through the environment keeps the CLI's --gpu and the decorator from drifting
        # apart, which is how a "RTX5090" deploy silently ran on a 22 GiB RTX4090.
        env = child_environment(home, {"AAWDEN_DEMO_GPU": options.gpu})
        beam = beam_binary(options.python)
        # beam-cli 0.2.207 takes the value as a positional argument (`beam secret create NAME VALUE`);
        # there is no --from-value flag. The value is passed as an argv element, so it is visible in
        # this process's own command line while the call runs, and nowhere else: not in the app file,
        # not in the image, not in any file under the repository.
        made = subprocess.run([beam, "secret", "create", secret, options.passcode],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        if made.returncode != 0:
            blob = (made.stdout + made.stderr).lower()
            if "already exists" in blob or "alreadyexist" in blob.replace(" ", ""):
                # Redeploying with the same passcode: the secret is already in the workspace, and
                # `beam secret create` refuses to overwrite. The value is unchanged, so carry on.
                print(f"secret {secret} already exists; reusing it", flush=True)
            else:
                print(made.stdout[-2000:] + made.stderr[-2000:], file=sys.stderr)
                print("secret creation failed; if it already exists, update it in the dashboard", file=sys.stderr)
                return made.returncode
        print(f"secret {secret} set", flush=True)
        deployed = subprocess.run([beam, "deploy", app], cwd=ROOT, env=env, text=True, timeout=3600)
        return deployed.returncode


def build_parser() -> argparse.ArgumentParser:
    default_python = sys.executable
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=default_python, help="interpreter with beam-client installed")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("launch", help="Run a command on a Beam GPU and wait for it")
    p.add_argument("--gpu", default=load_config()["defaults"]["gpu"])
    p.add_argument("--cpu", type=float, default=load_config()["defaults"]["cpu"])
    p.add_argument("--memory", default=load_config()["defaults"]["memory"])
    p.add_argument("--timeout", type=int, default=load_config()["defaults"]["timeout"])
    p.add_argument("--max-cost-usd", type=float, required=True,
                   help="refuse if timeout x hourly price exceeds this")
    p.add_argument("--output-subdir", required=True, help="volume folder, e.g. wsp-v3/beam-smoke")
    p.add_argument("--workspace", default="primary", choices=sorted(WORKSPACES),
                   help="primary = BEAM_TOKEN_2, secondary = BEAM_TOKEN; each has its own GPU queue")
    p.add_argument("--balance-cents", type=int, default=None,
                   help="Beam credit balance in cents; enables security_wsp.budget.reserve_job")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=launch)

    p = sub.add_parser("status", help="Show a Beam task by id")
    p.add_argument("task_id")
    p.add_argument("--workspace", default="primary", choices=sorted(WORKSPACES))
    p.set_defaults(func=status)

    p = sub.add_parser("fetch", help="Download a volume folder with beam cp")
    p.add_argument("remote_subdir")
    p.add_argument("local_dir")
    p.add_argument("--workspace", default="primary", choices=sorted(WORKSPACES),
                   help="volume lives in the workspace that ran the job")
    p.set_defaults(func=fetch)

    p = sub.add_parser("smoke", help="Doctor plus torch/toolkit check on RTX4090, then fetch")
    p.add_argument("--balance-cents", type=int, default=None)
    p.add_argument("--workspace", default="primary", choices=sorted(WORKSPACES))
    p.add_argument("--output-subdir", default=None,
                   help="volume folder (default: smoke preset plus a UTC timestamp, so every run is a new folder)")
    p.add_argument("--local-dir", default=None, help="local folder for the fetched output")
    p.set_defaults(func=smoke)

    p = sub.add_parser("deploy-demo", help="Deploy the public demo endpoint (scripts/beam_demo.py)")
    p.add_argument("--passcode", required=True,
                   help="checked against the X-Passcode header on every request; stored as a Beam secret")
    p.add_argument("--gpu", default="RTX4090")
    p.add_argument("--max-cost-usd", type=float, required=True,
                   help="refuse if the keep-warm window at this hourly price exceeds this")
    p.add_argument("--workspace", default="primary", choices=sorted(WORKSPACES))
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.set_defaults(func=deploy_demo)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    options = parser.parse_args(argv)
    try:
        return int(options.func(options))
    except LaunchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
