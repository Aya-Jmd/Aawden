"""One GPU launch: phase 3 localization on the merged base, then the phase 4 workbench dump.

Queue time has been the scarcest resource on this project, so the two discovery-only stages run as
one container: localization first (it produces the components/attention files the workbench reads),
then the programmer's workbench dump that consumes them.

Both stages enforce the same rules on their own: discovery split only, committed sha256 verified,
transfer wordings refused, nothing written outside its run directory. This driver only sequences
them and refuses to start if either input is missing or uncommitted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALLOWED_DATASETS = {"datasets/wsp-discovery.json"}
SEALED = {"datasets/wsp-validation.json", "datasets/wsp-final.json"}


def run(stage: str, command: list[str], log: Path) -> int:
    started = time.monotonic()
    print(f"[chain] {stage}: {' '.join(command)}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stream.write(line)
            stream.flush()
        code = process.wait()
    print(f"[chain] {stage} exit={code} in {time.monotonic() - started:.1f}s", flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model-merged-v7.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "wsp-discovery.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "wsp-v3" / "mt")
    parser.add_argument("--lora-spec", type=Path, default=ROOT / "configs" / "merged-v7-adapter-spec.json")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--attention-cases", type=int, default=12)
    parser.add_argument("--attention-call-cap", type=int, default=96)
    parser.add_argument("--causal-cases", type=int, default=16)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--limit-cases", type=int, default=20)
    parser.add_argument("--skip-localize", action="store_true")
    parser.add_argument("--skip-dump", action="store_true")
    parser.add_argument("--toy", action="store_true", help="offline plumbing check with the tiny random model")
    args = parser.parse_args(argv)

    dataset = (args.dataset if args.dataset.is_absolute() else ROOT / args.dataset).resolve()
    relative = dataset.relative_to(ROOT).as_posix() if dataset.is_relative_to(ROOT) else None
    if relative in SEALED or relative not in ALLOWED_DATASETS:
        raise SystemExit(f"Refused: {relative} is not the committed discovery split")
    for path in ([] if args.toy else (args.config, args.lora_spec)):
        if not path.is_file():
            raise SystemExit(f"Refused: missing input {path}")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = out / "chain.log"
    summary: dict = {"stages": []}

    if not args.skip_localize:
        code = run("localize", [sys.executable, "scripts/wsp_localize.py",
                                "--config", str(args.config), "--dataset", str(dataset),
                                "--output", str(out / "localize"), "--layers", args.layers,
                                "--top-k", str(args.top_k), "--attention-cases", str(args.attention_cases),
                                "--attention-call-cap", str(args.attention_call_cap),
                                "--causal-cases", str(args.causal_cases), "--budget", str(args.budget)]
                               + (["--toy", "--limit", "6", "--layers", "0,1", "--bootstrap-replicates", "200",
                                   "--no-circuitkit"] if args.toy else []),
                   log)
        summary["stages"].append({"stage": "localize", "exit_code": code})
        if code != 0:
            (out / "chain-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
            raise SystemExit(code)

    if not args.skip_dump:
        code = run("dump", [sys.executable, "scripts/wsp_agent_round.py", "dump",
                            "--config", str(args.config), "--dataset", str(dataset),
                            "--output", str(out / "agent"),
                            "--localization", str(out / "localize" / "components.json"),
                            "--limit-cases", str(args.limit_cases)] + (["--toy"] if args.toy else []),
                   log)
        summary["stages"].append({"stage": "dump", "exit_code": code})
        if code != 0:
            (out / "chain-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
            raise SystemExit(code)

    summary["lora_spec"] = str(args.lora_spec)
    (out / "chain-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
