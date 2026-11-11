"""Retrain chain: the dry-run plan is deterministic, matches configs/retrain-v7.json and the recorded evidence,
imports no torch, and refuses to overwrite an existing checkpoint or to exceed the job cap silently."""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retrain-v7.json"
SNAPSHOT_ZIP = ROOT / "runs" / "aya-beam-20261107" / "adapter-v7-reproduction-source.zip"
RUNS = ROOT / "runs" / "aya-beam-20261107"


def load_script():
    spec = importlib.util.spec_from_file_location("retrain_adapter_chain", ROOT / "scripts" / "retrain_adapter_chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chain():
    return load_script()


@pytest.fixture(scope="module")
def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    if not SNAPSHOT_ZIP.is_file():
        pytest.skip("v7 reproduction snapshot zip is absent")
    target = tmp_path_factory.mktemp("v7-snapshot")
    with zipfile.ZipFile(SNAPSHOT_ZIP) as archive:
        archive.extractall(target)
    return target


def evidence(*parts):
    """Load a recorded evidence file, or skip when that run directory was not synced.

    The v1-v7 run outputs live under `runs/aya-beam-20261107/`, which is committed but large; a
    working copy that synced code only (for example a scratch checkout used to run cluster jobs) has
    the config without the evidence. Absent evidence is a skip, not a failure: the check still runs
    wherever the files are present.
    """
    path = RUNS.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"recorded evidence absent: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_main(chain, argv, capsys):
    code = chain.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_config_stage_hyperparameters_match_recorded_protocols(config):
    recorded = {name: evidence(f"adapter-{name}", "study", "protocol.json") for name in ("v5", "v6", "v7")}
    stages = {stage["name"]: stage for stage in config["stages"]}
    assert list(stages) == ["v5", "v6", "v7"]
    for name, protocol in recorded.items():
        stage = stages[name]
        assert stage["epochs"] == protocol["epochs"], name
        assert stage["learning_rate"] == protocol["lr"], name
        assert stage["verdict_weight"] == protocol["verdict_weight"], name
        assert config["lora"]["rank"] == protocol["rank"]
        assert config["lora"]["alpha"] == protocol["alpha"]
        assert config["lora"]["target_modules"] == protocol["target_modules"]
        assert config["seed"] == protocol["seed"]
        assert config["conditions"] == protocol["conditions"]
        assert config["optimizer"]["accumulation"] == protocol["accumulation"]
    assert stages["v7"]["trace_prompt"] is True
    assert stages["v5"]["trace_prompt"] is False and stages["v6"]["trace_prompt"] is False


def test_config_resume_chain_matches_recorded_hashes(config):
    stages = {stage["name"]: stage for stage in config["stages"]}
    assert stages["v5"]["resume_from"] is None
    assert stages["v6"]["resume_from"] == "v5"
    assert stages["v7"]["resume_from"] == "v6"
    v5 = evidence("adapter-v5", "study", "development", "summary.json")
    v6 = evidence("adapter-v6", "study", "development", "summary.json")
    v6_protocol = evidence("adapter-v6", "study", "protocol.json")
    v7_protocol = evidence("adapter-v7", "study", "protocol.json")
    assert v6_protocol["resume_sha256"] == v5["model"]["adapter_sha256"] == stages["v5"]["recorded"]["adapter_sha256"]
    assert v7_protocol["resume_sha256"] == v6["model"]["adapter_sha256"] == stages["v6"]["recorded"]["adapter_sha256"]
    assert stages["v7"]["recorded"]["adapter_sha256"] == "e86f9475e3c8871ba3c60a63b10704954f66edcb791bd9631866a07d82a9022f"


def test_config_dataset_hashes_match_files(config):
    for name, spec in config["datasets"].items():
        path = ROOT / "datasets" / name
        assert path.is_file(), name
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"], name


def test_dry_run_plan_is_deterministic_and_matches_config(chain, config, snapshot, capsys):
    argv = ["--dry-run", "--source-root", str(snapshot), "--output", "checkpoints/retrained"]
    code, first, _ = run_main(chain, argv, capsys)
    code2, second, _ = run_main(chain, argv, capsys)
    assert code == 0 and code2 == 0
    assert first == second
    plan = json.loads(first)
    # The full three-stage chain in one process exceeds the 2100 s cap by design; per-stage jobs do not.
    assert plan["problems"] == ["all_stages_in_one_process_over_cap"]
    assert plan["source_fingerprint"]["matches"] is True
    assert plan["source_fingerprint"]["actual"] == config["source"]["fingerprint"]
    stages = plan["stages"]
    assert [s["name"] for s in stages] == ["v5", "v6", "v7"]
    assert [s["resume_from"] for s in stages] == [None, "v5", "v6"]
    # Steps are micro-steps (examples x epochs), as the recorded training.json files count them.
    assert [s["steps"] for s in stages] == [1040, 1560, 900]
    assert [s["examples_per_epoch"] for s in stages] == [260, 260, 300]
    assert [s["evaluates"] for s in stages] == [False, False, True]
    assert all(s["estimated_job_seconds"] <= config["gpu_estimate"]["timeout_cap_seconds"] for s in stages)
    assert plan["estimated_job_seconds_total"] > config["gpu_estimate"]["timeout_cap_seconds"]


def test_dry_run_does_not_import_torch(snapshot):
    code = (
        "import sys, json, io, contextlib, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('c', r'{ROOT / 'scripts' / 'retrain_adapter_chain.py'}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"buf = io.StringIO()\n"
        f"with contextlib.redirect_stdout(buf):\n"
        f"    m.main(['--dry-run', '--source-root', r'{snapshot}'])\n"
        "print('torch-loaded=' + str('torch' in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "torch-loaded=False" in result.stdout


def test_shortcut_plan_is_one_labelled_stage(chain, config, snapshot, capsys):
    code, out, _ = run_main(chain, ["--dry-run", "--shortcut", "--source-root", str(snapshot)], capsys)
    plan = json.loads(out)
    assert code == 0
    assert plan["deviation"] is True and plan["mode"] == "shortcut"
    assert [s["name"] for s in plan["stages"]] == ["v7-shortcut"]
    assert plan["stages"][0]["resume_from"] is None
    assert plan["stages"][0]["steps"] == 900
    assert plan["problems"] == []


def test_refuses_to_overwrite_existing_checkpoint(chain, snapshot, tmp_path, capsys):
    (tmp_path / "v5.safetensors").write_bytes(b"not a real adapter")
    code, _, err = run_main(chain, ["--stage", "v5", "--output", str(tmp_path), "--source-root", str(snapshot)], capsys)
    assert code == 2
    assert "refusing to overwrite" in err
    assert (tmp_path / "v5.safetensors").read_bytes() == b"not a real adapter"


def test_refuses_stage_whose_resume_file_is_missing(chain, snapshot, tmp_path, capsys):
    code, _, err = run_main(chain, ["--stage", "v7", "--output", str(tmp_path), "--source-root", str(snapshot)], capsys)
    assert code == 2
    assert "resume file missing" in err


def test_all_stages_in_one_process_refused_without_flag(chain, config, snapshot, tmp_path, capsys):
    code, _, err = run_main(chain, ["--output", str(tmp_path), "--source-root", str(snapshot)], capsys)
    assert code == 2
    assert "all_stages_in_one_process_over_cap" in err
    assert list(tmp_path.iterdir()) == []


def test_resume_file_that_disagrees_with_its_record_is_refused(chain, snapshot, tmp_path, capsys):
    (tmp_path / "v5.safetensors").write_bytes(b"bytes")
    (tmp_path / "v5.json").write_text(json.dumps({"adapter_sha256": "0" * 64}), encoding="utf-8")
    code, _, err = run_main(chain, ["--stage", "v6", "--output", str(tmp_path), "--source-root", str(snapshot)], capsys)
    assert code == 2
    assert "differs from its own record" in err


def test_source_mismatch_is_refused_by_default(chain, tmp_path, capsys):
    code, _, err = run_main(chain, ["--stage", "v5", "--output", str(tmp_path), "--source-root", str(ROOT)], capsys)
    assert code == 2
    assert "plan problem: source" in err
