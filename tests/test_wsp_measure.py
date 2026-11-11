import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wsp_measure.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True)


def test_toy_run_writes_summary_and_decision(tmp_path):
    out = tmp_path / "baseline"
    r = run("--toy", "--limit", "4", "--output", str(out),
            "--dataset", "datasets/wsp-discovery.json", "--dataset", "datasets/wsp-validation.json")
    assert r.returncode == 0, r.stderr
    rows = json.loads((out / "results.json").read_text())
    # 3 conditional arms x 8 cases x 3 conditions + A_raw neutral x 8
    assert len(rows) == 3 * 8 * 3 + 8
    decision = json.loads((out / "decision.json").read_text())
    assert decision["route"] in ("a_original_aya", "b_merged_adapter")
    assert "validation_strong_prompts" in decision
    prefilled = [r for r in rows if r["method"] == "A_pf"]
    assert all(r["messages"][-1] == {"role": "assistant", "content": '{"verdict":"'} for r in prefilled)
    assert all(r["messages"][-1]["role"] != "assistant" for r in rows if r["method"] == "A_raw")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["evidence_kind"] == "constructed_toy"


def test_resume_is_identical_and_changed_identity_rejected(tmp_path):
    out = tmp_path / "baseline"
    base = ("--toy", "--limit", "2", "--output", str(out), "--dataset", "datasets/wsp-discovery.json")
    assert run(*base).returncode == 0
    first = (out / "results.json").read_text()
    assert run(*base).returncode == 0
    assert (out / "results.json").read_text() == first
    r = run(*base, "--arms", "A_pf")
    assert r.returncode != 0 and "Resume rejected" in r.stderr


def test_sealed_final_split_is_refused(tmp_path):
    r = run("--toy", "--output", str(tmp_path / "x"), "--dataset", "datasets/wsp-final.json")
    assert r.returncode != 0 and "sealed" in (r.stderr + r.stdout)


def test_toy_prefilled_answers_parse_strictly(tmp_path):
    out = tmp_path / "b"
    assert run("--toy", "--limit", "3", "--output", str(out), "--dataset", "datasets/wsp-discovery.json").returncode == 0
    rows = json.loads((out / "results.json").read_text())
    assert rows and all(r["parse_error"] is None for r in rows), [r["parse_error"] for r in rows][:3]


def test_final_only_wording_refused(tmp_path):
    r = run("--toy", "--wording", "ciso", "--output", str(tmp_path / "x"), "--dataset", "datasets/wsp-discovery.json")
    assert r.returncode != 0 and "reserved for final" in (r.stderr + r.stdout)


def test_uncommitted_or_modified_dataset_refused(tmp_path):
    copy = ROOT / "datasets" / "wsp-discovery.json"
    tampered = tmp_path / "wsp-discovery.json"
    data = json.loads(copy.read_text())
    data["cases"][0]["oracle_verdict"] = ("vulnerable" if data["cases"][0]["oracle_verdict"] != "vulnerable"
                                          else "not_vulnerable_for_target")
    tampered.write_text(json.dumps(data))
    r = run("--toy", "--output", str(tmp_path / "x"), "--dataset", str(tampered))
    assert r.returncode != 0


def test_merged_label_requires_lora_spec(tmp_path):
    r = run("--toy", "--label", "merged-adapter", "--output", str(tmp_path / "x"),
            "--dataset", "datasets/wsp-discovery.json")
    assert r.returncode != 0 and "--label" in (r.stderr + r.stdout)
