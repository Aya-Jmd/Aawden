from dataclasses import replace
from pathlib import Path

from security_wsp.artifacts import read_json
from security_wsp.calibrated_review import bias_score
from security_wsp.consensus_review import select_answers
from security_wsp.data import CLEAN, VULNERABLE, oracle_cases, validate_cases
from security_wsp.rigorous_review import (FLOW_VARIANTS, MAP_VARIANTS, POLICY_VARIANTS, SCENARIO_VARIANTS, VARIANTS,
                                         check_request, checked_dataset, error_report, rigorous_request)
from security_wsp.policy_notes import compact_projection, counterfactual_notes, policy_notes
from security_wsp.source_notes import source_notes
from security_wsp.symbolic_review import flow_notes


ROOT = Path(__file__).resolve().parents[1]


def test_rigorous_splits_are_balanced_canonical_and_disjoint():
    protocol = read_json(ROOT / "configs/rigorous-review.json")
    development = checked_dataset(ROOT, protocol["development"])
    confirmation = checked_dataset(ROOT, protocol["confirmation"])
    followup = read_json(ROOT / "configs/rigorous-review-v2.json")
    fresh = checked_dataset(ROOT, followup["confirmation"])
    challenge = read_json(ROOT / "configs/rigorous-review-v4.json")
    harder = checked_dataset(ROOT, challenge["confirmation"])
    calibration = read_json(ROOT / "configs/rigorous-review-v7.json")
    newest = checked_dataset(ROOT, calibration["confirmation"])
    projection = read_json(ROOT / "configs/rigorous-review-v8.json")
    projected = checked_dataset(ROOT, projection["confirmation"])
    validate_cases(list(oracle_cases().values()))
    assert len(development) == 16
    assert len(confirmation) == 32
    assert len(fresh) == 32
    assert len(harder) == 32
    assert len(newest) == 32
    assert len(projected) == 32 and all(compact_projection(case) for case in projected)
    for cases in (development, confirmation, fresh, harder, newest, projected):
        for family in {case.family_id for case in cases}:
            pair = [case for case in cases if case.family_id == family]
            assert len(pair) == 2
            assert {case.oracle_verdict for case in pair} == {VULNERABLE, CLEAN}
            assert pair[0].threat_model == pair[1].threat_model


def test_model_prompts_do_not_change_when_oracle_metadata_changes():
    case = next(case for case in oracle_cases().values()
                if case.split == "rigorous_confirmation" and case.target_category == "access_control")
    changed = replace(case, split="hidden", oracle_verdict=CLEAN if case.oracle_verdict == VULNERABLE else VULNERABLE)
    for variant in (*VARIANTS, *MAP_VARIANTS, *FLOW_VARIANTS, *POLICY_VARIANTS, *SCENARIO_VARIANTS):
        assert rigorous_request(case, variant) == rigorous_request(changed, variant)


def test_source_notes_show_guards_and_arguments_without_running_the_source():
    notes = source_notes(
        "def review(conn, name, allowed):\n"
        "    if not allowed:\n"
        "        return None\n"
        "    return conn.execute('SELECT ? ', (name,))\n"
    )
    assert "line 2 (falsehood of (allowed)) is false" in notes
    assert "with SQL text 'SELECT ? ' and bindings (name,)" in notes


def test_symbolic_paths_use_only_active_literal_choice_and_preserve_binding():
    notes = flow_notes(
        "def lookup(conn, name):\n"
        "    options = (('SELECT ?', (name,)), (f'SELECT {name}', ()))\n"
        "    key = 0\n"
        "    query, values = options[key]\n"
        "    return conn.execute(query, values)\n"
    )
    assert "returns conn.execute('SELECT ?', (name,))" in notes
    assert "SELECT {name}" not in notes


def test_non_owner_projection_simplifies_a_guard_without_executing_the_fixture():
    cases = [case for case in oracle_cases().values() if case.family_id.endswith("approval_reset")]
    notes = {case.oracle_verdict: policy_notes(case) for case in cases}
    assert "returns records.get(record_id)" not in notes[CLEAN]
    assert "returns records.get(record_id)" in notes[VULNERABLE]


def test_symbolic_sql_projection_preserves_reversing_quote_escapes():
    cases = [case for case in oracle_cases().values() if case.family_id.endswith("escape_then_recover")]
    notes = {case.oracle_verdict: policy_notes(case) for case in cases}
    assert "username.replace" not in notes[VULNERABLE]
    assert "username.replace" in notes[CLEAN]


def test_consistency_review_receives_only_the_draft_explanation():
    case = next(case for case in oracle_cases().values()
                if case.split == "rigorous_confirmation" and case.target_category == "access_control")
    draft = {"parsed": {"explanation": "A branch binds the input.", "verdict": "hidden_draft_decision"},
             "oracle_verdict": "hidden_fixture_label", "raw_output": "unused_raw_draft"}
    request = check_request(case, "check", draft)
    user = request.messages[1]["content"]
    assert "A branch binds the input." in user
    assert "hidden_draft_decision" not in user
    assert "hidden_fixture_label" not in user
    assert "unused_raw_draft" not in user


def test_symbolic_scenarios_preserve_quote_binding_and_non_owner_guards():
    cases = {case.family_id.rsplit("_", 1)[-1]: case for case in oracle_cases().values()
             if case.oracle_verdict == CLEAN and case.family_id.endswith(("quote_count_dispatch", "compound_exception"))}
    sql = counterfactual_notes(cases["dispatch"])
    access = counterfactual_notes(cases["exception"])
    assert "SELECT name FROM users WHERE name = ?" in sql
    assert "same tenant" in access and "different tenant" in access
    assert "returns records.get(record_id)" not in access


def test_compact_projection_falls_back_for_mutating_access_policy():
    cases = {case.family_id.rsplit("_", 1)[-1]: case for case in oracle_cases().values()
             if case.oracle_verdict == CLEAN and case.family_id.endswith(("set_gate", "compound_exception"))}
    assert compact_projection(cases["gate"]) == ""
    assert "returns" in compact_projection(cases["exception"])


def test_false_alarms_misses_invalids_and_abstentions_are_separate():
    rows = []
    for index, (expected, predicted, parse_error) in enumerate((
        (CLEAN, VULNERABLE, None), (VULNERABLE, CLEAN, None), (VULNERABLE, None, "parse error"),
        (CLEAN, "insufficient_information", None), (VULNERABLE, VULNERABLE, None), (CLEAN, CLEAN, None),
    )):
        rows.append({"case_id": str(index), "family_id": str(index // 2), "storage_id": str(index),
                     "category": "sql_injection", "oracle_verdict": expected, "predicted_verdict": predicted,
                     "generation_error": None, "parse_error": parse_error, "correct": expected == predicted})
    report = error_report(rows)
    assert [row["kind"] for row in report["errors"]] == ["false_alarm", "missed_bug", "invalid_output", "abstention"]
    assert report["overall"]["accuracy"] == 2 / 6
    assert report["pairs_correct"] == 1


def test_verdict_bias_uses_exposed_scores_and_counts_ties_as_uncertain():
    rows = [
        {"positive": 8.0, "negative": 7.5, "other": 5.0, "expected": CLEAN},
        {"positive": 9.0, "negative": 7.0, "other": 6.0, "expected": VULNERABLE},
    ]
    assert bias_score(rows, 0) == (1, 1)
    assert bias_score(rows, -.5) == (1, 0)
    assert bias_score(rows, -1) == (2, 0)


def test_second_opinion_keeps_raw_aya_answers_and_preserves_invalids():
    primary = [{"case_id": "a", "predicted_verdict": VULNERABLE}, {"case_id": "b", "predicted_verdict": None},
               {"case_id": "c", "predicted_verdict": VULNERABLE}, {"case_id": "d", "predicted_verdict": CLEAN}]
    second = [{"case_id": "a", "predicted_verdict": CLEAN}, {"case_id": "c", "predicted_verdict": None}]
    selected = select_answers(primary, second)
    assert selected == [second[0], primary[1], primary[2], primary[3]]
    assert selected[0] is second[0]
