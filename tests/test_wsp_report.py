"""Offline tests for scripts/wsp_report.py: the judge-facing Markdown report.

The run directory built here is a FIXTURE. It copies the real shape of
runs/wsp-v3/baseline-A2/summary.json but every value in it is fabricated, and every label says so.
It is not evidence about any model, and nothing here loads a model or touches a GPU.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wsp_report.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import digest, file_hash  # noqa: E402

FIXTURE_LABEL = "FIXTURE-NOT-REAL-DATA"
FAKE_MARK = "FABRICATED FIXTURE VALUES — NOT MEASURED FROM ANY MODEL"


def _script():
    spec = importlib.util.spec_from_file_location("wsp_report_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wsp_report = _script()


# --------------------------------------------------------------------------------------------------
# fixture: the real summary shape, fabricated values
# --------------------------------------------------------------------------------------------------


def block(n=8, tp=2, tn=1, fp=3, fn=2, bacc=0.250, invalid=1, tokens=(120, 60)):
    """One condition block in the shape statistics.metrics returns."""
    return {"n": n, "n_vulnerable": tp + fn, "n_clean": tn + fp, "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn, "balanced_accuracy": bacc, "accuracy": 0.375,
            "false_positive_rate": 0.75, "false_negative_rate": 0.5, "miss_rate": 0.5, "coverage": 0.875,
            "invalid_count": invalid, "generation_errors": 0,
            "input_tokens": tokens[0], "output_tokens": tokens[1], "latency_seconds": 12.5,
            "explanation_grounding": None, "general_coding_accuracy": None}


def fixture_by_arm_split():
    """Both splits, both arms, fabricated values in the producer's exact key layout."""
    return {
        "A_pf": {
            "wsp_discovery": {
                "neutral": block(bacc=0.125, tp=1, tn=1, fp=3, fn=3, invalid=2),
                "correct_suggestion": block(bacc=0.875, tp=6, tn=1, fp=0, fn=0, invalid=1),
                "incorrect_suggestion": block(bacc=0.000, tp=0, tn=0, fp=4, fn=4, invalid=0),
            },
            "wsp_validation": {
                "neutral": block(n=6, bacc=0.250, tp=2, tn=0, fp=2, fn=2, invalid=2),
                "incorrect_suggestion": block(n=6, bacc=0.000, tp=0, tn=0, fp=3, fn=3, invalid=0),
            },
        },
        "B_trace_pf": {
            "wsp_discovery": {
                "neutral": block(bacc=0.375, tp=3, tn=0, fp=4, fn=1, invalid=0),
                "correct_suggestion": block(bacc=0.625, tp=4, tn=1, fp=0, fn=0, invalid=0),
                "incorrect_suggestion": block(bacc=0.500, tp=3, tn=1, fp=2, fn=2, invalid=0),
            },
            "wsp_validation": {
                "neutral": block(n=6, bacc=0.500, tp=3, tn=0, fp=2, fn=1, invalid=0),
                "incorrect_suggestion": block(n=6, bacc=0.250, tp=1, tn=1, fp=2, fn=2, invalid=0),
            },
        },
    }


def fixture_summary():
    """Real key layout taken from runs/wsp-v3/baseline-A2/summary.json; all values fabricated."""
    return {
        "model": {"model_id": FAKE_MARK, "adapter": "fixture", "revision": "0" * 40},
        "label": FIXTURE_LABEL,
        "datasets": {"datasets/fixture-discovery.json": "0" * 64},
        "summary": {
            "by_arm": {"A_pf": {"neutral": block(bacc=0.125)}},
            "by_arm_split": fixture_by_arm_split(),
            "paired": {
                "B_trace_pf-A_pf": {
                    "incorrect_suggestion": {
                        "left": "B_trace_pf", "right": "A_pf", "condition": "incorrect_suggestion",
                        "difference": 0.500, "ci95": [0.250, 0.750], "unit": "family_id",
                        "families": 4, "cases": 8, "replicates": 100,
                        "interval": "paired percentile cluster bootstrap", "difference_unit": "proportion"},
                    "neutral": {},
                }
            },
        },
    }


def build_fixture_run(tmp_path: Path, name: str = "baseline-fixture") -> Path:
    """A verified wsp_measure-shaped run directory: 3 artifacts plus a manifest."""
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text("[]\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(fixture_summary(), indent=2) + "\n", encoding="utf-8")
    (out / "decision.json").write_text(json.dumps({
        "A_pf_neutral_ba": 0.125, "A_pf_misleading_ba": 0.0, "A_pf_drop": 0.125,
        "route": "fixture-route", "rule": FAKE_MARK, "deference_drop_meets_screen": True,
        "validation_strong_prompts": {}, "prompt_only_solution": False}, indent=2) + "\n", encoding="utf-8")
    write_manifest(out)
    return out


def write_manifest(run: Path, *, identity=None, status="completed", artifacts=None):
    identity = identity if identity is not None else {"format": FAKE_MARK, "arms": 2}
    names = artifacts if artifacts is not None else ["results.json", "summary.json", "decision.json"]
    manifest = {
        "run_id": digest(identity)[:16], "parent_run": None, "identity_sha256": digest(identity),
        "identity": identity, "stage": "fixture", "status": status, "evidence_kind": "fabricated_fixture",
        "created_at": "2026-01-01T00:00:00+00:00", "environment": {}, "failure_reason": None,
        "artifact_hashes": {name: file_hash(run / name) for name in names},
        "expected_predictions": 0,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def report_for(run: Path) -> str:
    args = wsp_report.main.__globals__["argparse"].Namespace(
        baseline=run, merged=None, localize=None, rounds=None, final=None, out=None)
    return wsp_report.build(args)


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True)


# --------------------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------------------


def test_report_contains_the_fixture_numbers(tmp_path):
    run = build_fixture_run(tmp_path)
    text = report_for(run)

    assert FIXTURE_LABEL in text
    # Straight from summary.json: balanced accuracy, per split, with n and the confusion counts.
    assert "| `A_pf` | neutral | 8 | 0.125 | 1 | 1 | 3 | 3 | 2 |" in text
    assert "| `A_pf` | correct_suggestion | 8 | 0.875 | 6 | 1 | 0 | 0 | 1 |" in text
    assert "| `A_pf` | incorrect_suggestion | 8 | 0.000 | 0 | 0 | 4 | 4 | 0 |" in text
    # The validation split appears separately, never pooled with discovery.
    assert "**Split `wsp_validation`** — n = 6 per condition block." in text
    assert "**Split `wsp_discovery`** — n = 8 per condition block." in text
    # Paired interval, printed verbatim from the summary's `paired` block.
    assert "[0.250, 0.750]" in text
    assert "`B_trace_pf-A_pf`" in text
    # The stronger-prompt comparison is a real number, not an interpolation: A_pf's misleading
    # accuracy is 0.000 and B_trace_pf's is 0.500 on the same split, so the delta is +0.500.
    assert "| `wsp_discovery` | `B_trace_pf` | incorrect_suggestion | 0.500 | 0.500 | MISSING |" in text
    assert "| `wsp_discovery` | `A_pf` | incorrect_suggestion | 0.000 | MISSING | B_trace_pf: -0.500 |" in text
    # Token and latency cost come from the same blocks.
    assert "| `A_pf` | neutral | 8 | 120 | 60 | 12.50 | 0 |" in text
    # The fixture's own provenance is named.
    assert "summary.json" in text


def test_absent_stages_are_missing_not_zero(tmp_path):
    run = build_fixture_run(tmp_path)
    text = report_for(run)

    missing_section = text.split("## 7. MISSING", 1)[1].split("## 8.", 1)[0]
    assert "Localization: MISSING" in missing_section
    assert "Agent rounds: MISSING" in missing_section
    assert "Sealed final evaluation: MISSING" in missing_section
    assert "Shuffled-weight control: MISSING" in missing_section
    # A control with no artifact must never be rendered as a measured pass or a zero.
    assert "| No-op edit | must reproduce the baseline | MISSING |" in text
    assert "| Shuffled weights | separates reading weights from prior knowledge | MISSING |" in text
    # The no-prefill arm is absent from this fixture, so it is MISSING, not 0.000.
    assert "No-prefill control: MISSING" in text


def test_tampered_artifact_is_refused(tmp_path):
    run = build_fixture_run(tmp_path)
    # Flip one measured value after the manifest recorded the hash.
    payload = json.loads((run / "summary.json").read_text())
    payload["summary"]["by_arm_split"]["A_pf"]["wsp_discovery"]["neutral"]["balanced_accuracy"] = 0.999
    (run / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(wsp_report.ReportError) as excinfo:
        report_for(run)
    assert "integrity failure" in str(excinfo.value)
    assert "summary.json" in str(excinfo.value)

    # The CLI path must fail loudly and write no report.
    out = tmp_path / "should-not-exist.md"
    result = run_cli("--baseline", str(run), "--out", str(out))
    assert result.returncode == 2
    assert "REFUSED" in result.stderr
    assert not out.exists()


def test_incomplete_run_is_refused(tmp_path):
    run = build_fixture_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["status"] = "running"
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(wsp_report.ReportError):
        report_for(run)


def test_output_is_byte_deterministic(tmp_path):
    run = build_fixture_run(tmp_path)
    first, second = tmp_path / "a.md", tmp_path / "b.md"
    for path in (first, second):
        result = run_cli("--baseline", str(run), "--out", str(path))
        assert result.returncode == 0, result.stderr
    assert first.read_bytes() == second.read_bytes()
    # No wall-clock value leaks into the report.
    text = first.read_text()
    assert "generated_at" not in text
    assert "2026-" not in text


def test_no_inputs_produces_a_report_of_missing(tmp_path):
    out = tmp_path / "empty.md"
    result = run_cli("--out", str(out))
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    missing_section = text.split("## 7. MISSING", 1)[1].split("## 8.", 1)[0]
    assert "Baseline: MISSING — no directory supplied." in missing_section
    assert "Merged-base measurement: MISSING" in missing_section
    # Reproduction commands are still printed, so a reader knows how to fill the gaps.
    assert "python scripts/wsp_measure.py" in text


def test_report_imports_no_gpu_code():
    # The module must stay importable on a CPU-only box: no torch, no transformers.
    source = SCRIPT.read_text()
    for banned in ("import torch", "from transformers", "cuda"):
        assert banned not in source


# --------------------------------------------------------------------------------------------------
# localization and rounds: synthetic directories, same integrity rules
# --------------------------------------------------------------------------------------------------


def _manifest(run: Path, names: list[str]):
    return write_manifest(run, artifacts=names)


def build_fixture_localize(tmp_path: Path) -> Path:
    run = tmp_path / "localize-fixture"
    run.mkdir(parents=True, exist_ok=True)
    (run / "components.json").write_text(json.dumps({
        "ranking_rule": FAKE_MARK, "claim": "exploratory localization, not a causal claim",
        "ranked": [{"component": "resid_post.l0", "site": "resid_post", "layer": 0, "rank": 1,
                    "score": 0.420, "separation_statistic": 0.900, "random_baseline_mean": 0.480,
                    "separation_p_value": 0.010, "cases": 20, "degenerate": False}]}, indent=2) + "\n")
    (run / "attention.json").write_text("{}\n")
    (run / "localization.json").write_text(json.dumps({
        "causal": {"no_op_passed": True, "random_seeds": [17, 29, 43],
                   "components": [{"component": "resid_post.l0",
                                   "no_op_control": {"max_absolute_margin_delta": 0.0,
                                                     "tolerance": 1e-5, "passed": True},
                                   "effect": {"mean_margin_delta": 0.310, "sign_flip": {"p_value": 0.002}},
                                   "largest_random_absolute_effect": 0.090,
                                   "label_mapping_swap": {"status": "consistent"},
                                   "supported_by_controls": False,
                                   "conclusion": "exploratory localization, not a causal claim"}]}},
        indent=2) + "\n")
    _manifest(run, ["components.json", "attention.json", "localization.json"])
    return run


def build_fixture_rounds(tmp_path: Path) -> Path:
    root = tmp_path / "rounds-fixture"
    round_dir = root / "rounds" / "round-01"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "spec.json").write_text("{}\n")
    (round_dir / "summary.json").write_text(json.dumps({
        "format": "aawden-wsp-agent-round-v1", "round": 1, "kind": "proposal", "seed": None,
        "misleading_balanced_accuracy": 0.420, "neutral_balanced_accuracy": 0.400,
        "correct_suggestion_balanced_accuracy": 0.860, "changed_scalars": 1234, "update_norm": 0.51,
        "per_class_counts": {"incorrect_suggestion": {"n": 20}},
        "budget": {"forward_units_used": 17, "forward_units_remaining": 33}}, indent=2) + "\n")
    artifacts = {n: file_hash(round_dir / n) for n in ("summary.json", "spec.json")}
    (root / "state.json").write_text(json.dumps({
        "format": "aawden-wsp-agent-state-v1",
        "budget": {"forward_units_used": 17, "forward_units_remaining": 33, "forward_pass_budget": 50},
        "rounds": [{"round": 1, "artifact_dir": "rounds/round-01", "artifacts": artifacts}]}, indent=2) + "\n")
    return root


def test_localization_is_exploratory_without_control_support(tmp_path):
    run = build_fixture_run(tmp_path)
    locate = build_fixture_localize(tmp_path)
    args = wsp_report.main.__globals__["argparse"].Namespace(
        baseline=run, merged=None, localize=locate, rounds=None, final=None, out=None)
    text = wsp_report.build(args)

    # supported_by_controls is False in the fixture, so the causal claim must be withheld.
    assert "**Claim status: exploratory localization, not a causal claim.**" in text
    assert "| 1 | `resid_post.l0` | resid_post | 0 | 20 | 0.900 | 0.480 | 0.420 | no |" in text
    assert "| `resid_post.l0` | 0.310 | 0.002 | 0.090 | consistent | no |" in text
    # The no-op passes and is read from the causal block, not invented.
    assert "passed=yes" in text
    # Shuffled weights are still absent, so still MISSING.
    assert "| Shuffled weights | separates reading weights from prior knowledge | MISSING |" in text


def test_rounds_table_reports_budget_and_result(tmp_path):
    run = build_fixture_run(tmp_path)
    rounds = build_fixture_rounds(tmp_path)
    args = wsp_report.main.__globals__["argparse"].Namespace(
        baseline=run, merged=None, localize=None, rounds=rounds, final=None, out=None)
    text = wsp_report.build(args)

    assert "used 17, remaining 33, total 50" in text
    assert "| 1 | proposal | -- | 0.420 | 0.400 | 0.860 | 17 | 1234 | 0.510 |" in text
    assert "round 1 = 20" in text


def test_tampered_round_artifact_is_refused(tmp_path):
    run = build_fixture_run(tmp_path)
    rounds = build_fixture_rounds(tmp_path)
    # Rewrite a round summary after state.json recorded its hash.
    (rounds / "rounds" / "round-01" / "summary.json").write_text("{}\n")
    args = wsp_report.main.__globals__["argparse"].Namespace(
        baseline=run, merged=None, localize=None, rounds=rounds, final=None, out=None)
    with pytest.raises(wsp_report.ReportError) as excinfo:
        wsp_report.build(args)
    assert "integrity check" in str(excinfo.value)


def test_pipes_are_escaped_in_table_cells(tmp_path):
    run = build_fixture_run(tmp_path)
    locate = build_fixture_localize(tmp_path)
    args = wsp_report.main.__globals__["argparse"].Namespace(
        baseline=run, merged=None, localize=locate, rounds=None, final=None, out=None)
    text = wsp_report.build(args)
    # A raw "|Δmargin|" would split the controls table into extra columns.
    assert "\\|Δmargin\\|" in text
    for line in text.splitlines():
        if line.startswith("| No-op edit"):
            assert line.count(" | ") == 3
