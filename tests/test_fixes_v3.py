import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_report_module():
    spec = importlib.util.spec_from_file_location("report_aya_progress", ROOT / "scripts/report_aya_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(balanced_accuracy, threshold_exceeded):
    best = {"study": "adapter-v7", "stage": "confirmation", "method": "adapted",
            "metrics": {"balanced_accuracy": balanced_accuracy, "n": 20},
            "family_cluster_ci95": [.75, 1.0], "bootstrap": {"replicates": 10000, "seed": 42, "families": 10}}
    return {"scope": "scope", "threshold": .75, "threshold_exceeded": threshold_exceeded,
            "best_observed_confirmation": best, "records": [], "limitations": ["limitation"]}


def test_point_estimate_sentence_follows_threshold_outcome():
    report = _load_report_module()
    met = report.render(_report(0.9, True))
    assert "The point estimate meets the goal; the interval does not establish" in met
    assert "does not meet the goal" not in met

    unmet = report.render(_report(0.7, False))
    assert "The point estimate does not meet the goal; the interval does not establish" in unmet
    assert "The point estimate meets the goal" not in unmet
