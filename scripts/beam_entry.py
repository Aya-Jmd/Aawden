"""Beam GPU entry point for running pinned Aawden commands in a serverless container.

``run_command(args, output_subdir)`` runs one command from the synced checkout (the
container working copy is /mnt/code), streams its log to the ``aawden-v3`` volume,
copies the produced ``runs/<output_subdir>`` directory there, and returns the exit code
and a log tail. ``check`` is the in-container readiness probe.

Settings come from configs/beam.json and the two lock files it cites. The launcher
(scripts/beam_launch.py) sets AAWDEN_BEAM_GPU / _CPU / _MEMORY / _TIMEOUT before it
imports this module, so the decorator reflects the requested resources.

Never place secrets here. The Hugging Face token reaches the container as the Beam
workspace secret AAWDEN_AYA_HF_TOKEN and is exported only as HF_TOKEN in the job.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "beam.json").read_text(encoding="utf-8"))
SOURCE_LOCK = json.loads((ROOT / "dependency-sources.lock.json").read_text(encoding="utf-8"))
RESEARCH_LOCK = ROOT / "requirements-research.lock"

SECRET_NAME = CONFIG["secret_name"]
VOLUME_NAME = CONFIG["volume"]["name"]
VOLUME_MOUNT = CONFIG["volume"]["mount_path"]
JOB_ENV = dict(CONFIG["env"])
DEFAULTS = CONFIG["defaults"]
# Launcher-supplied overrides; the container never sets them, so it uses the defaults.
GPU = os.environ.get("AAWDEN_BEAM_GPU") or DEFAULTS["gpu"]
CPU = float(os.environ.get("AAWDEN_BEAM_CPU") or DEFAULTS["cpu"])
MEMORY = os.environ.get("AAWDEN_BEAM_MEMORY") or DEFAULTS["memory"]
TIMEOUT = int(os.environ.get("AAWDEN_BEAM_TIMEOUT") or DEFAULTS["timeout"])

try:
    from beam import Image, Volume, function
except ImportError:  # The SDK is needed to launch or run jobs, not for the helpers below.
    Image = Volume = function = None


CONSTRAINTS_FILE = "/tmp/aawden-constraints.txt"
PIN_PATTERN = re.compile(r"([A-Za-z0-9._-]+)==([A-Za-z0-9.+!_-]+)")


def lock_pins() -> list[str]:
    """Every exact pin in requirements-research.lock. The two sponsor git lines are installed from archives."""
    pins = []
    for raw in RESEARCH_LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or " @ " in line:
            continue
        if not PIN_PATTERN.fullmatch(line):
            raise ValueError(f"Unexpected line in requirements-research.lock: {line!r}")
        pins.append(line)
    if len(pins) < 100:
        raise ValueError("requirements-research.lock yielded too few pins")
    return pins


def constraints_command(pins: list[str]) -> str:
    """Write the lock pins to a constraints file so the sponsor installs cannot move any of them."""
    quoted = " ".join(shlex.quote(pin) for pin in pins)
    return f"printf '%s\\n' {quoted} > {CONSTRAINTS_FILE}"


def sponsor_install_commands() -> list[str]:
    """Install the pinned CircuitKIT and AuditKIT archives, verifying each SHA-256 first."""
    commands = []
    for name in sorted(SOURCE_LOCK["sources"]):
        source = SOURCE_LOCK["sources"][name]
        url, digest = source["url"], source["archive_sha256"]
        if not re.fullmatch(r"https://codeload\.github\.com/Lexsi-Labs/[A-Za-z]+/zip/[0-9a-f]{40}", url):
            raise ValueError(f"Unexpected sponsor archive URL for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Malformed SHA-256 for {name}")
        archive = f"/tmp/{name}-{source['revision'][:12]}.zip"
        commands.append(
            f"set -eu; curl -fsSL --retry 3 -o {archive} '{url}'; "
            f"echo '{digest}  {archive}' | sha256sum -c -; "
            f"python3 -m pip install --no-cache-dir -c {CONSTRAINTS_FILE} {archive}; rm -f {archive}"
        )
    return commands


def verification_command() -> str:
    """Fail the image build unless every locked pin is installed at exactly its pinned version."""
    code = (
        "import importlib.metadata as m, sys; "
        f"pins=[l.strip().split('==') for l in open('{CONSTRAINTS_FILE}') if l.strip()]; "
        "bad=[(n, v, m.version(n)) for n, v in pins if m.version(n) != v]; "
        "print('lock pin check:', len(pins), 'pins; mismatches:', bad); sys.exit(1 if bad else 0)"
    )
    return f"python3 -c {shlex.quote(code)} && (python3 -m pip check || echo 'pip check reported issues; see log')"


def build_image():
    if Image is None:
        return None
    pins = lock_pins()
    image = Image(python_version=CONFIG["python_version"])
    image.add_python_packages(pins)
    # One RUN step: the builder does not keep /tmp between separate commands, so the
    # constraints file must be written and read in the same shell.
    parts = [constraints_command(pins), *sponsor_install_commands(), verification_command()]
    image.add_commands(["set -eu; " + "; ".join(parts)])
    return image


IMAGE = build_image()
VOLUME = Volume(name=VOLUME_NAME, mount_path=VOLUME_MOUNT) if Volume is not None else None


def _sdk_function(**options):
    if function is None:
        return lambda fn: fn
    return function(**options)


@_sdk_function(
    name=CONFIG["function_name"],
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY,
    timeout=TIMEOUT,
    retries=int(DEFAULTS["retries"]),
    image=IMAGE,
    volumes=[VOLUME] if VOLUME is not None else [],
    secrets=[SECRET_NAME],
    env=JOB_ENV,
)
def run_command(args: list, output_subdir: str) -> dict:
    return execute_job(args, output_subdir)


def validate_output_subdir(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output_subdir must be a non-empty string")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise ValueError("output_subdir may contain only letters, digits, ., _, / and -")
    # Check the raw segments: PurePosixPath would silently drop "." and empty parts.
    if value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("output_subdir must be a relative path without empty, . or .. segments")
    return str(PurePosixPath(value))


def normalize_command(args) -> list[str]:
    if not isinstance(args, (list, tuple)) or not args or not all(isinstance(a, str) for a in args):
        raise ValueError("args must be a non-empty list of strings")
    # "{python}" resolves to the container interpreter, so shell strings use the image's packages.
    command = [a.replace("{python}", sys.executable) for a in args]
    if command[0] in ("python", "python3"):
        command[0] = sys.executable
    elif command[0].endswith(".py"):
        command.insert(0, sys.executable)
    return command


def execute_job(args, output_subdir, *, root: Path = ROOT, volume_root: str = VOLUME_MOUNT) -> dict:
    subdir = validate_output_subdir(output_subdir)
    command = normalize_command(args)
    volume_dir = Path(volume_root) / subdir
    scratch = root / "runs" / subdir
    volume_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    token = env.pop("AAWDEN_AYA_HF_TOKEN", None)
    if token:
        env["HF_TOKEN"] = token
    env.update({
        "AAWDEN_BEAM_OUTPUT": str(scratch),
        "AAWDEN_BEAM_VOLUME": str(volume_root),
        **JOB_ENV,
    })

    started = time.monotonic()
    tail: deque[str] = deque(maxlen=200)
    log_path = volume_dir / "job.log"
    with log_path.open("a", encoding="utf-8") as log:
        header = f"$ {' '.join(command)}\n"
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            tail.append(line)
        exit_code = process.wait()

    copied = []
    for produced in sorted(scratch.rglob("*")):
        if produced.is_file():
            target = volume_dir / produced.relative_to(scratch)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, target)
            copied.append(str(produced.relative_to(scratch)))

    result = {
        "exit_code": exit_code,
        "seconds": round(time.monotonic() - started, 1),
        "command": command,
        "output_subdir": subdir,
        "volume_path": f"{VOLUME_NAME}/{subdir}",
        "task_id": os.environ.get("TASK_ID", ""),
        "copied_files": copied,
        "log_tail": "".join(tail)[-4000:],
    }
    (volume_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def environment_report(root: Path = ROOT, volume_root: str = VOLUME_MOUNT) -> dict:
    report: dict = {
        "ok": True,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "code_root": str(root),
        "task_id": os.environ.get("TASK_ID", ""),
    }
    dotenv = sorted(str(p.relative_to(root)) for p in root.rglob(".env*") if p.is_file())
    report["dotenv_files_in_workdir"] = dotenv
    report["ok"] = report["ok"] and not dotenv
    report["secret_env_set"] = {
        SECRET_NAME: bool(os.environ.get(SECRET_NAME)),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
    }

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        torch_info = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": cuda}
        if cuda:
            props = torch.cuda.get_device_properties(0)
            torch_info.update(
                device_name=torch.cuda.get_device_name(0),
                bf16_supported=bool(torch.cuda.is_bf16_supported(including_emulation=False)),
                total_memory_gib=round(props.total_memory / 2**30, 2),
            )
        report["torch"] = torch_info
        report["ok"] = report["ok"] and cuda
    except Exception as exc:  # Report the failure rather than hiding it behind a traceback.
        report["torch"] = {"error": type(exc).__name__, "message": str(exc)[:300]}
        report["ok"] = False

    packages = {}
    # CircuitKIT must load before transformer_lens: it installs compatibility patches and
    # refuses to load once transformer_lens is already imported.
    for distribution, module in (
        ("circuitkit", "circuitkit"),
        ("auditkit", "auditkit"),
        ("transformers", "transformers"),
        ("transformer-lens", "transformer_lens"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("huggingface-hub", "huggingface_hub"),
        ("numpy", "numpy"),
    ):
        try:
            importlib.import_module(module)
            packages[distribution] = {"import": "ok", "version": importlib.metadata.version(distribution)}
        except Exception as exc:
            packages[distribution] = {"import": "failed", "error": type(exc).__name__, "message": str(exc)[:200]}
            if distribution in ("circuitkit", "auditkit", "transformers", "transformer-lens"):
                report["ok"] = False
    report["packages"] = packages

    volume = Path(volume_root)
    report["volume"] = {
        "path": str(volume),
        "exists": volume.is_dir(),
        "entries": sorted(p.name for p in volume.iterdir())[:100] if volume.is_dir() else [],
    }
    return report


def check() -> int:
    report = environment_report()
    output = Path(os.environ.get("AAWDEN_BEAM_OUTPUT") or ROOT / "runs" / "beam-check")
    output.mkdir(parents=True, exist_ok=True)
    (output / "check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def invoke(args_list: list[str], output_subdir: str) -> int:
    """Run run_command on Beam from the launcher; prints one BEAM_RESULT or BEAM_ERROR line."""
    if function is None:
        print("BEAM_ERROR the Beam SDK is not importable in this interpreter", flush=True)
        return 2
    try:
        result = run_command.remote(args=list(args_list), output_subdir=output_subdir)
    except Exception as exc:
        print(f"BEAM_ERROR task_id={getattr(exc, 'task_id', '')} type={type(exc).__name__}", flush=True)
        return 1
    print("BEAM_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("exit_code", 1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="In-container readiness probe (torch, toolkits, env files, volume)")
    inv = sub.add_parser("invoke", help="Launcher entry: run one command on Beam")
    inv.add_argument("--output-subdir", required=True)
    inv.add_argument("rest", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if options.command == "check":
        return check()
    rest = options.rest[1:] if options.rest[:1] == ["--"] else options.rest
    return invoke(rest, options.output_subdir)


if __name__ == "__main__":
    raise SystemExit(main())
