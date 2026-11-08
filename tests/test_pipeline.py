from pathlib import Path
import pytest
from security_wsp.artifacts import read_json, verify_run
from security_wsp.demo import run_demo
from security_wsp.sponsors import enable_vendor, audit_records

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_demo_exports_reloads_and_remains_explicitly_toy(tmp_path):
    protocol = dict(read_json(ROOT / "configs/protocol.json"), bootstrap_replicates=100)
    summary = run_demo(tmp_path / "run", protocol, use_auditkit=False)
    assert summary["evidence_kind"] == "constructed_toy"
    assert summary["scientific_success"] is False
    assert summary["oracle_passed"] == 50
    assert summary["reload_verified"]
    assert summary["candidate_trials"] == 6
    for condition in protocol["conditions"]:
        assert summary["metrics"]["N"][condition]["balanced_accuracy"] == summary["metrics"]["B"][condition]["balanced_accuracy"]
    assert verify_run(tmp_path / "run")["valid"]
    assert run_demo(tmp_path / "run", protocol, use_auditkit=False)["run_id"] == summary["run_id"]


def test_actual_auditkit_scores_strict_raw_json_and_retains_invalid_errors(tmp_path):
    enable_vendor()
    pytest.importorskip("auditkit", reason="Pinned AuditKIT source/install absent; core tests do not imply sponsor verification")
    rows = []
    for n, raw in enumerate(('{"verdict":"vulnerable","category":"sql_injection","evidence_lines":[2],"explanation":"Source evidence","confidence":0.9}', "not JSON")):
        rows.append({"request_id": str(n), "method": "A", "oracle_verdict": "vulnerable", "raw_output": raw,
                     "category": "sql_injection", "source_line_count": 3})
    result = audit_records(rows, tmp_path / "audit.json")
    assert result["status"] == "executed" and result["samples"] == 2
    report = read_json(tmp_path / "audit.json")
    assert report["failed_count"] == 0
    assert report["stats"]["aawden_strict_verdict"]["mean"] == .5

