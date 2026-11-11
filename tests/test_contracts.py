import json
import pytest
from security_wsp.artifacts import RunStore, atomic_json, read_json, verify_run
from security_wsp.data import CLEAN, VULNERABLE, ABSTAIN
from security_wsp.interventions import TrialRegistry, project_vector, load_toy_checkpoint, write_toy_checkpoint
from security_wsp.parsing import parse_response
from security_wsp.statistics import metrics, paired_cluster_interval


def payload():
    return dict(verdict=VULNERABLE, category="sql_injection", evidence_lines=[2], explanation="Line 2 uses unbound interpolation.", confidence=.9)


@pytest.mark.parametrize("text", [
    chr(96)*3+"json\n{}\n"+chr(96)*3, '{"verdict":"vulnerable","verdict":"not_vulnerable_for_target"}',
    json.dumps({"verdict": VULNERABLE}), "[]", "null", "{broken",
    json.dumps(dict(payload(), confidence=True)), json.dumps(dict(payload(), confidence=float("nan"))),
    json.dumps(dict(payload(), evidence_lines=[True])), json.dumps(dict(payload(), evidence_lines=[99])),
    json.dumps(dict(payload(), evidence_lines=[2, 2])), json.dumps(dict(payload(), category="other")),
    json.dumps(dict(payload(), verdict="probably vulnerable")), json.dumps(dict(payload(), extra="unexpected")),
    json.dumps(dict(payload(), evidence_lines=[{}])),
])
def test_ambiguous_or_invalid_outputs_are_not_recovered(text):
    parsed, error = parse_response(text, category="sql_injection", line_count=3)
    assert parsed is None and error


def test_valid_and_abstaining_outputs_are_distinct():
    parsed, error = parse_response(json.dumps(payload()), category="sql_injection", line_count=3)
    assert error is None and parsed["verdict"] == VULNERABLE
    parsed, error = parse_response(json.dumps(dict(payload(), verdict=ABSTAIN)), category="sql_injection", line_count=3)
    assert error is None and parsed["verdict"] == ABSTAIN


def row(case, label, prediction, method="B", family=None):
    return dict(case_id=case, family_id=family or case, oracle_verdict=label, predicted_verdict=prediction,
                method=method, condition="incorrect_suggestion", correct=label==prediction, parse_error=None)


def test_metric_denominators_include_invalid_and_abstaining_outputs():
    rows = [row("a", VULNERABLE, VULNERABLE), row("b", VULNERABLE, None),
            row("c", CLEAN, ABSTAIN), row("d", CLEAN, CLEAN)]
    result = metrics(rows)
    assert result["balanced_accuracy"] == .5
    assert result["coverage"] == .5
    assert result["miss_rate"] == .5
    assert result["false_negative_rate"] == 0


def test_cluster_bootstrap_retains_pairing_and_perfect_noop_difference():
    rows = []
    for family in range(4):
        for label in (VULNERABLE, CLEAN):
            for method in ("B", "D"):
                rows.append(row(str(family)+label, label, label, method, str(family)))
    result = paired_cluster_interval(rows, replicates=100)
    assert result["ci95"] == [0, 0] and result["families"] == 4
    with pytest.raises(ValueError, match="identical"):
        paired_cluster_interval(rows[:-1], replicates=100)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_cluster_interval(rows+[rows[0]], replicates=100)


def test_candidate_budget_counts_distinct_recipes(tmp_path):
    registry = TrialRegistry(tmp_path / "trials.json", {"model": "fixed"}, budget=2)
    first = registry.reserve({"strength": .5})
    assert registry.reserve({"strength": .5})["candidate_id"] == first["candidate_id"]
    registry.reserve({"strength": 1})
    with pytest.raises(ValueError, match="budget"):
        registry.reserve({"strength": .25})
    with pytest.raises(ValueError, match="different"):
        TrialRegistry(tmp_path / "trials.json", {"model": "changed"}, budget=2)


def test_projection_and_checkpoint_count_actual_stored_changes(tmp_path):
    original = [1., 1.25]
    assert project_vector(original, [0, 1], 0) == original
    edited = project_vector(original, [0, 1], 1)
    assert edited == [1, 0] and original == [1, 1.25]
    record = write_toy_checkpoint(tmp_path / "edit.json", original, edited, {"type": "projection"})
    assert record["changed_entries"] == 1 and record["touched_entries"] == 2
    assert load_toy_checkpoint(tmp_path / "edit.json") == edited
    tampered = read_json(tmp_path / "edit.json")
    tampered["weights"][0] += .1
    atomic_json(tmp_path / "edit.json", tampered)
    with pytest.raises(ValueError, match="integrity"):
        load_toy_checkpoint(tmp_path / "edit.json")


def test_resumption_and_artifact_tampering(tmp_path):
    identity = {"dataset": "fixed", "prompt": "v1"}
    run = RunStore(tmp_path, identity, stage="test", evidence_kind="test")
    run.start()
    run.save_prediction("a", {"correct": True})
    run.fail("interrupted", interrupted=True)
    resumed = RunStore(tmp_path, identity, stage="test", evidence_kind="test")
    assert resumed.load_prediction("a")["correct"]
    with pytest.raises(ValueError, match="Resume rejected"):
        RunStore(tmp_path, dict(identity, prompt="v2"), stage="test", evidence_kind="test")
    atomic_json(tmp_path / "results.json", {"correct": 1})
    resumed.finish(["a"], ["results.json"])
    assert verify_run(tmp_path)["valid"]
    atomic_json(tmp_path / "results.json", {"correct": 2})
    with pytest.raises(ValueError, match="integrity"):
        verify_run(tmp_path)



def test_tolerant_json_recovers_fenced_answer():
    from security_wsp.parsing import tolerant_json
    text = '```json\n{\n  "answer": 30\n}\n```\n\nHere is why: x[0] * 10 + x[-1] = 30.'
    assert tolerant_json(text) == {"answer": 30}


def test_extract_json_object_ignores_python_set_literals():
    # Aya's explanations contain balanced braces that are not JSON (regressions2 cases 17 and 22).
    from security_wsp.parsing import tolerant_json
    text = '```json\n{"answer": 4}\n```\nThen `s ^ t` = {1, 2, 4} and `{n % 3 for n in range(10)}` is a set.'
    assert tolerant_json(text) == {"answer": 4}


def test_extract_json_object_refuses_ambiguity_and_absence():
    import pytest
    from security_wsp.parsing import extract_json_object
    with pytest.raises(ValueError):
        extract_json_object('{"answer": 1} or {"answer": 2}')
    with pytest.raises(ValueError):
        extract_json_object("no object here (1, 2)")


def test_verdict_parsing_stays_strict():
    # The security benchmark measures format obedience, so a fenced verdict is still invalid.
    text = '```json\n' + json.dumps(payload()) + '\n```'
    parsed, error = parse_response(text, category="sql_injection", line_count=3)
    assert parsed is None and "Expecting value" in error
