"""Offline tests for the Beam launcher: cost guard, token handling, upload exclusions, pins."""
from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import beam_entry  # noqa: E402
import beam_launch  # noqa: E402
from beam_launch import LaunchRefused  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "beam.json").read_text(encoding="utf-8"))


ALL_IN_HOURLY_2VCPU_32GI = 0.69 + 2 * 0.18972 + 32 * 0.0198  # GPU + vCPU + RAM, RTX4090


def test_cost_guard_uses_all_in_serverless_rate_times_timeout():
    expected = round(ALL_IN_HOURLY_2VCPU_32GI * 1800 / 3600, 4)
    assert beam_launch.estimate_cost_usd(CONFIG, "RTX4090", 1800, 2.0, "32Gi") == pytest.approx(expected)
    assert beam_launch.check_cost_guard(CONFIG, "RTX4090", 1800, expected, 2.0, "32Gi") == pytest.approx(expected)


def test_cost_guard_refuses_when_estimate_exceeds_maximum():
    with pytest.raises(LaunchRefused, match="exceeds --max-cost-usd"):
        beam_launch.check_cost_guard(CONFIG, "RTX4090", 1800, 0.80, 2.0, "32Gi")


def test_cost_guard_counts_cpu_and_ram_not_only_gpu():
    gpu_only = beam_launch.estimate_cost_usd(CONFIG, "RTX4090", 3600, 0.0, "0.0001Gi")
    with_resources = beam_launch.estimate_cost_usd(CONFIG, "RTX4090", 3600, 2.0, "32Gi")
    assert with_resources > gpu_only + 0.9


def test_cost_guard_refuses_unpriced_gpu_and_non_gib_memory():
    with pytest.raises(LaunchRefused, match="No hourly price"):
        beam_launch.estimate_cost_usd(CONFIG, "H100", 600)
    with pytest.raises(LaunchRefused, match="Gi"):
        beam_launch.estimate_cost_usd(CONFIG, "RTX4090", 600, 2.0, "32768Mi")


def test_smoke_preset_fits_its_own_cost_cap():
    preset = CONFIG["smoke"]
    estimate = beam_launch.estimate_cost_usd(CONFIG, preset["gpu"], preset["timeout"],
                                             CONFIG["defaults"]["cpu"], CONFIG["defaults"]["memory"])
    assert estimate <= preset["max_cost_usd"] <= 0.40


def test_runtime_policy_enforces_budget_maximum_and_allowed_gpus():
    beam_launch.check_runtime_policy("RTX4090", 1800, 2.0, "32Gi")
    with pytest.raises(LaunchRefused, match="compute budget maximum"):
        beam_launch.check_runtime_policy("RTX4090", 2400, 2.0, "32Gi")
    with pytest.raises(LaunchRefused, match="allowed GPUs"):
        beam_launch.check_runtime_policy("H100", 600, 2.0, "32Gi")


def test_parser_requires_explicit_cost_cap_and_keeps_command():
    parser = beam_launch.build_parser()
    options = parser.parse_args([
        "launch", "--gpu", "RTX4090", "--timeout", "1800", "--max-cost-usd", "0.5",
        "--output-subdir", "wsp-v3/x", "--", "python", "aawden.py", "doctor",
    ])
    assert options.gpu == "RTX4090" and options.timeout == 1800 and options.max_cost_usd == 0.5
    assert options.output_subdir == "wsp-v3/x"
    assert options.command[-3:] == ["python", "aawden.py", "doctor"]
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "--output-subdir", "x", "--", "python", "aawden.py", "doctor"])


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "wsp-v3/../x", "a b", "", "x/./y"])
def test_output_subdir_rejects_unsafe_paths(bad):
    with pytest.raises(LaunchRefused):
        beam_launch.sanitize_subdir(bad)


def test_output_subdir_accepts_volume_folder():
    assert beam_launch.sanitize_subdir("wsp-v3/beam-smoke") == "wsp-v3/beam-smoke"


def test_refuses_old_token_even_when_it_is_the_only_token():
    with pytest.raises(LaunchRefused, match="is not set"):
        beam_launch.select_token(CONFIG, {"BEAM_TOKEN": "old-secret"}, {"BEAM_TOKEN": "old-secret"})


def test_refuses_when_workspace_two_token_equals_old_token():
    with pytest.raises(LaunchRefused, match="equals the old"):
        beam_launch.select_token(CONFIG, {"BEAM_TOKEN_2": "same", "BEAM_TOKEN": "same"}, {})


def test_selects_workspace_two_token_and_never_the_old_one():
    token = beam_launch.select_token(CONFIG, {"BEAM_TOKEN": "old"}, {"BEAM_TOKEN_2": "new", "BEAM_TOKEN": "old"})
    assert token == "new"
    token = beam_launch.select_token(CONFIG, {"BEAM_TOKEN_2": "env-new"}, {"BEAM_TOKEN_2": "file-new"})
    assert token == "env-new"


def test_refusal_message_does_not_echo_token_values():
    with pytest.raises(LaunchRefused) as info:
        beam_launch.select_token(CONFIG, {"BEAM_TOKEN": "super-secret-old"}, {})
    assert "super-secret-old" not in str(info.value)


def test_dotenv_parser_handles_quotes_comments_and_export(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport BEAM_TOKEN_2=\"quoted-value\"\nAAWDEN_AYA_HF_TOKEN='single'\n\nBROKEN_LINE\n",
        encoding="utf-8",
    )
    values = beam_launch.parse_dotenv(env_file)
    assert values == {"BEAM_TOKEN_2": "quoted-value", "AAWDEN_AYA_HF_TOKEN": "single"}
    assert beam_launch.parse_dotenv(tmp_path / "missing") == {}


def test_isolated_home_writes_only_workspace_two_config_and_removes_it():
    with beam_launch.isolated_beam_home("tok-test-123") as home:
        config_file = home / ".beam" / "config.ini"
        assert config_file.read_text(encoding="utf-8") == (
            "[default]\ntoken = tok-test-123\ngateway_host = gateway.beam.cloud\ngateway_port = 443\n"
        )
        assert home != Path.home()
        assert not str(home).startswith(str(Path.home() / ".beam"))
    assert not home.exists()


def test_child_environment_scrubs_beam_variables(monkeypatch):
    monkeypatch.setenv("BEAM_TOKEN", "old")
    monkeypatch.setenv("BETA9_TOKEN", "other")
    monkeypatch.setenv("CONFIG_PATH", "/somewhere")
    monkeypatch.setenv("KEEP_ME", "yes")
    env = beam_launch.child_environment(Path("/tmp/iso-home"), {"AAWDEN_BEAM_GPU": "RTX4090"})
    assert "BEAM_TOKEN" not in env and "BETA9_TOKEN" not in env and "CONFIG_PATH" not in env
    assert env["HOME"] == "/tmp/iso-home"
    assert env["KEEP_ME"] == "yes" and env["AAWDEN_BEAM_GPU"] == "RTX4090"


def test_beamignore_excludes_dotenv_files_and_covers_config_list():
    lines = {line.strip() for line in (ROOT / ".beamignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")}
    assert ".env" in lines and ".env.*" in lines
    for entry in CONFIG["excluded_from_upload"]:
        assert entry in lines, entry


def test_image_pins_are_every_research_lock_pin_and_no_git_lines():
    lock_lines = []
    for raw in (ROOT / "requirements-research.lock").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and " @ " not in line:
            lock_lines.append(line)
    assert beam_entry.lock_pins() == lock_lines
    pins = set(beam_entry.lock_pins())
    for required in ("torch==2.15.0", "transformers==5.22.1", "accelerate==1.15.0", "safetensors==0.8.0",
                     "huggingface-hub==1.33.0", "numpy==2.4.6", "tokenizers==0.23.3", "transformer-lens==3.8.0",
                     "sentence-transformers==6.2.0"):
        assert required in pins, required
    assert not any(pin.startswith(("circuitkit", "auditkit")) for pin in pins)


def test_constraints_file_and_verification_cover_every_pin():
    pins = beam_entry.lock_pins()
    constraints = beam_entry.constraints_command(pins)
    assert constraints.startswith("printf ")
    assert beam_entry.CONSTRAINTS_FILE in constraints
    assert "torch==2.15.0" in constraints
    verify = beam_entry.verification_command()
    assert beam_entry.CONSTRAINTS_FILE in verify and "sys.exit(1 if bad else 0)" in verify


def test_sponsor_installs_use_constraints_so_pins_cannot_move():
    for command in beam_entry.sponsor_install_commands():
        assert f"-c {beam_entry.CONSTRAINTS_FILE}" in command
        assert "sha256sum -c -" in command


def test_sponsor_install_uses_lock_archives_and_sha256():
    lock = json.loads((ROOT / "dependency-sources.lock.json").read_text(encoding="utf-8"))["sources"]
    commands = "\n".join(beam_entry.sponsor_install_commands())
    for name, source in lock.items():
        assert source["url"] in commands
        assert f"echo '{source['archive_sha256']}" in commands
    assert commands.count("sha256sum -c") == len(lock)


def test_normalize_command_maps_python_and_placeholders():
    assert beam_entry.normalize_command(["python", "aawden.py", "doctor"])[0] == sys.executable
    assert beam_entry.normalize_command(["aawden.py", "doctor"])[0] == sys.executable
    shell = beam_entry.normalize_command(["sh", "-c", "{python} aawden.py doctor"])
    assert shell[0] == "sh" and sys.executable in shell[2]
    with pytest.raises(ValueError):
        beam_entry.normalize_command([])


def test_execute_job_runs_locally_maps_secret_and_copies_outputs(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "runs").mkdir(parents=True)
    volume = tmp_path / "volume"
    monkeypatch.setenv("AAWDEN_AYA_HF_TOKEN", "hf-test-value")
    script = (
        "import os, pathlib\n"
        "print('HF_TOKEN_SET', 'HF_TOKEN' in os.environ, 'SECRET_NAME_VISIBLE', 'AAWDEN_AYA_HF_TOKEN' in os.environ)\n"
        "out = pathlib.Path(os.environ['AAWDEN_BEAM_OUTPUT'])\n"
        "(out / 'result.txt').write_text('ok')\n"
    )
    result = beam_entry.execute_job(
        [sys.executable, "-c", script], "wsp-v3/unit", root=root, volume_root=str(volume),
    )
    assert result["exit_code"] == 0
    assert (volume / "wsp-v3" / "unit" / "result.txt").read_text() == "ok"
    log = (volume / "wsp-v3" / "unit" / "job.log").read_text()
    assert "HF_TOKEN_SET True SECRET_NAME_VISIBLE False" in log
    assert "hf-test-value" not in log and "hf-test-value" not in json.dumps(result)
    assert "result.txt" in result["copied_files"]


def test_execute_job_reports_failure_exit_code(tmp_path):
    result = beam_entry.execute_job(
        [sys.executable, "-c", "import sys; sys.exit(3)"], "unit/fail",
        root=tmp_path, volume_root=str(tmp_path / "vol"),
    )
    assert result["exit_code"] == 3


def test_environment_report_flags_dotenv_in_workdir(tmp_path):
    (tmp_path / ".env").write_text("BEAM_TOKEN_2=do-not-upload\n", encoding="utf-8")
    report = beam_entry.environment_report(root=tmp_path, volume_root=str(tmp_path / "missing"))
    assert report["ok"] is False
    assert ".env" in report["dotenv_files_in_workdir"]
    assert report["volume"]["exists"] is False
    assert "do-not-upload" not in json.dumps(report)


def test_launch_refuses_over_cap_before_any_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no subprocess may start for a refused launch")

    monkeypatch.setattr(beam_launch.subprocess, "Popen", forbidden)
    monkeypatch.setattr(beam_launch.subprocess, "run", forbidden)
    options = Namespace(gpu="RTX4090", timeout=1800, cpu=2.0, memory="32Gi", max_cost_usd=0.10,
                        output_subdir="wsp-v3/refused", command=["--", "python", "aawden.py", "doctor"],
                        balance_cents=None, python=sys.executable)
    with pytest.raises(LaunchRefused, match="exceeds --max-cost-usd"):
        beam_launch.launch(options)


def test_launch_refuses_when_balance_cannot_cover_floor(monkeypatch):
    monkeypatch.setattr(beam_launch.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    options = Namespace(gpu="RTX4090", timeout=600, cpu=2.0, memory="32Gi", max_cost_usd=0.40,
                        output_subdir="wsp-v3/poor", command=["--", "python", "aawden.py", "doctor"],
                        balance_cents=1000, python=sys.executable)
    with pytest.raises(LaunchRefused, match="Budget reservation refused"):
        beam_launch.launch(options)


def test_reserve_job_accepts_when_balance_covers_floor_and_margin():
    record = beam_launch.reserve_with_balance(5000, "RTX4090", 900, 2.0, "32Gi")
    assert record["passed"] is True
    assert record["maximum_job_reserve_cents"] > 0


def test_fetch_refuses_unsafe_remote_path(tmp_path):
    options = Namespace(remote_subdir="../secrets", local_dir=str(tmp_path), python=sys.executable)
    with pytest.raises(LaunchRefused):
        beam_launch.fetch(options)


def test_launcher_help_runs_without_network():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "beam_launch.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0
    assert "launch" in completed.stdout and "smoke" in completed.stdout


def test_probe_imports_circuitkit_before_transformer_lens():
    source = (ROOT / "scripts" / "beam_entry.py").read_text(encoding="utf-8")
    assert source.index('("circuitkit", "circuitkit")') < source.index('("transformer-lens", "transformer_lens")')


class _FakeBeam:
    """Stands in for `beam ls` and `beam cp`, writing the bytes it was told to."""

    def __init__(self, listing, payload_size):
        self.listing = listing
        self.payload_size = payload_size

    def __call__(self, args, **kwargs):
        if args[1] == "ls":
            return subprocess.CompletedProcess(args, 0, json.dumps(self.listing), "")
        if args[1] == "cp":
            Path(kwargs["cwd"], args[3]).write_bytes(b"x" * self.payload_size)
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


def _fetch_options(tmp_path):
    return Namespace(remote_subdir="wsp-v3/unit-fetch", local_dir=str(tmp_path / "out"), python=sys.executable)


def test_fetch_fails_when_download_size_differs_from_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_TOKEN_2", "workspace-two-test-token")
    # Every beam call is faked below, so the binary lookup is faked too; no beam install is needed.
    monkeypatch.setattr(beam_launch, "beam_binary", lambda python: "beam")
    listing = [{"path": "wsp-v3/unit-fetch/check.json", "size": "1215"}]
    monkeypatch.setattr(beam_launch.subprocess, "run", _FakeBeam(listing, payload_size=1445))
    assert beam_launch.fetch(_fetch_options(tmp_path)) == 1


def test_fetch_succeeds_when_sizes_match_and_recurses_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_TOKEN_2", "workspace-two-test-token")
    # Every beam call is faked below, so the binary lookup is faked too; no beam install is needed.
    monkeypatch.setattr(beam_launch, "beam_binary", lambda python: "beam")
    listing_by_folder = {
        "wsp-v3/unit-fetch": [{"path": "wsp-v3/unit-fetch/check.json", "size": "12"},
                              {"path": "wsp-v3/unit-fetch/sub"}],
        "wsp-v3/unit-fetch/sub": [{"path": "wsp-v3/unit-fetch/sub/a.txt", "size": "12"}],
    }

    class Fake(_FakeBeam):
        def __call__(self, args, **kwargs):
            if args[1] == "ls":
                folder = args[-1].split("/", 1)[1]
                return subprocess.CompletedProcess(args, 0, json.dumps(listing_by_folder[folder]), "")
            return super().__call__(args, **kwargs)

    monkeypatch.setattr(beam_launch.subprocess, "run", Fake([], payload_size=12))
    assert beam_launch.fetch(_fetch_options(tmp_path)) == 0
    assert (tmp_path / "out" / "check.json").stat().st_size == 12
    assert (tmp_path / "out" / "sub" / "a.txt").stat().st_size == 12
