import hashlib
from dataclasses import replace
import pytest
from security_wsp.data import build_cases, run_oracle, validate_cases, split_by_family, CLEAN, VULNERABLE
from security_wsp.prompts import render


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c.case_id)
def test_every_label_has_executable_behavioral_evidence(case):
    result = run_oracle(case)
    assert result["passed"], result
    assert result["observed_verdict"] == case.oracle_verdict
    assert any("violation" in check for check in result["checks"])


def test_balanced_screen_and_real_source_hashes():
    cases = build_cases()
    assert len(cases) == 50
    assert sum(c.oracle_verdict == CLEAN for c in cases) == 25
    assert sum(c.oracle_verdict == VULNERABLE for c in cases) == 25
    for case in cases:
        assert case.source_sha256 == hashlib.sha256(case.code.encode()).hexdigest()
    validate_cases(cases)


def test_family_and_source_leakage_are_rejected():
    cases = build_cases()
    with pytest.raises(ValueError, match="leakage"):
        validate_cases([cases[0], replace(cases[1], split="validation")])
    split = split_by_family(cases)
    validate_cases(split)
    assert {c.split for c in split} == {"discovery", "validation", "exploratory_test"}
    assert split == split_by_family(cases)
    assert split != split_by_family(cases, seed=12)


def test_shared_exact_templates_stay_in_one_statistical_cluster():
    import ast
    groups = {}
    for c in build_cases():
        key = ast.dump(ast.parse(c.code), include_attributes=False)
        groups.setdefault(key, set()).add(c.family_id)
    assert all(len(group) == 1 for group in groups.values())


def test_prompts_exclude_oracle_metadata_and_preserve_source():
    case = build_cases()[0]
    requests = [render(case, condition) for condition in ("neutral", "correct_suggestion", "incorrect_suggestion")]
    for request in requests:
        prompt = str(request.messages)
        for forbidden in ("oracle_verdict", "oracle_test_ids", "reviewer_id", case.family_id, case.case_id, case.source_sha256):
            assert forbidden not in prompt
        assert request.messages[-1]["content"].split("<source>\n")[1].split("</source>")[0] == case.code
    assert len({r.request_id for r in requests}) == 3


def test_imported_code_cannot_be_executed_as_an_oracle():
    case = replace(build_cases()[0], code="raise RuntimeError('untrusted')")
    with pytest.raises(ValueError, match="canonical"):
        run_oracle(case)

