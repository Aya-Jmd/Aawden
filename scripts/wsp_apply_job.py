"""Run one WSP apply round on a GPU host, with the driver state staged into the job.

The Beam upload excludes ``runs/`` (it holds model outputs), so a job cannot see a run directory that
was created by an earlier job. The driver state is therefore staged under ``jobinput/<name>/`` which
*is* uploaded, and this wrapper moves it into the output directory before invoking the driver. The
driver then appends a new round to the same hash-chained state, and the launcher copies the whole
``runs/<subdir>`` back to the volume.

Refusals are the driver's own: this wrapper never bypasses a budget, submission or sealed-data check.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True,
                        help="staged driver state, repo-relative, e.g. jobinput/wsp-dump2")
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory the driver should use, e.g. runs/wsp-v3/wsp-dump2")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "wsp-discovery.json")
    parser.add_argument("--limit-cases", type=int, default=20)
    parser.add_argument("--wording", default="reviewer")
    args, extra = parser.parse_known_args(argv)

    stage = args.stage if args.stage.is_absolute() else ROOT / args.stage
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not (stage / "state.json").is_file():
        raise SystemExit(f"Refused: no staged driver state at {stage}/state.json")

    output.mkdir(parents=True, exist_ok=True)
    for item in sorted(stage.iterdir()):
        target = output / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print(f"[apply-job] staged {stage.name} -> {output}", flush=True)

    command = [sys.executable, "scripts/wsp_agent_round.py", "apply",
               "--config", str(args.config), "--dataset", str(args.dataset),
               "--output", str(output), "--localization", str(args.localization),
               "--spec", str(args.spec), "--limit-cases", str(args.limit_cases),
               "--wording", args.wording, "--baseline", "auto", *extra]
    print("[apply-job] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
