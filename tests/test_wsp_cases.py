import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from security_wsp.artifacts import file_hash, read_json
from security_wsp.data import CLEAN, VULNERABLE, Case, load_dataset
from security_wsp.final_review import verify_independent_review
from security_wsp.prompts import CONDITIONS, render
from security_wsp.wsp_cases import SPLITS, run_wsp_oracle, wsp_cases

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_wsp_data", ROOT / "scripts" / "build_wsp_data.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

EXPECTED = {"wsp_discovery": (60, 30), "wsp_validation": (40, 20), "wsp_final": (120, 60)}
CATEGORIES = ("sql_injection", "access_control")


def by_split(split):
    return [case for case in wsp_cases() if case.split == split]


def test_counts_and_balance_per_split():
    for split, (count, families) in EXPECTED.items():
        cases = by_split(split)
        assert len(cases) == count
        assert len({case.family_id for case in cases}) == families
        counts = Counter((case.target_category, case.oracle_verdict) for case in cases)
        for category in CATEGORIES:
            assert counts[(category, VULNERABLE)] == counts[(category, CLEAN)]
        per_category = Counter(case.target_category for case in cases if case.oracle_verdict == VULNERABLE)
        assert abs(per_category["sql_injection"] - per_category["access_control"]) <= 2
        for family in {case.family_id for case in cases}:
            pair = [case for case in cases if case.family_id == family]
            assert len(pair) == 2
            assert {case.oracle_verdict for case in pair} == {VULNERABLE, CLEAN}
            assert pair[0].threat_model == pair[1].threat_model


def test_every_canonical_oracle_passes():
    results = [run_wsp_oracle(case) for case in wsp_cases()]
    assert len(results) == 220
    assert all(result["passed"] for result in results)
    for case, result in zip(wsp_cases(), results):
        if case.oracle_verdict == VULNERABLE:
            assert result["observed_verdict"] == VULNERABLE
        else:
            assert result["observed_verdict"] == CLEAN


def test_oracle_refuses_uncanonical_code():
    case = by_split("wsp_discovery")[0]
    tampered = Case(**{**case.__dict__, "code": case.code + "\n# edited\n"})
    with pytest.raises(ValueError):
        run_wsp_oracle(tampered)


SEALED_HINT = ("the sealed final set is not stored in git; `python scripts/build_wsp_data.py` rebuilds it "
               "byte-for-byte from src/security_wsp/wsp_cases.py and its hash is checked against configs/wsp-data.json")


def test_datasets_load_match_builder_and_config_hashes():
    config = read_json(ROOT / "configs/wsp-data.json")
    if not (ROOT / config["files"]["wsp_final"]["path"]).exists():
        pytest.skip(SEALED_HINT)
    assert config["sealed"] == ["wsp_final"]
    assert "must not be opened for model selection" in config["sealing_note"]
    for split in SPLITS:
        path = ROOT / config["files"][split]["path"]
        assert config["files"][split]["sha256"] == file_hash(path)
        loaded = load_dataset(path)
        assert loaded == by_split(split)
        assert config["files"][split]["cases"] == len(loaded)
        assert config["files"][split]["counts"] == {
            category: {label: sum(1 for c in loaded if c.target_category == category and c.oracle_verdict == label)
                       for label in (VULNERABLE, CLEAN)} for category in CATEGORIES}


def test_disjoint_from_all_existing_datasets_and_splits():
    existing = builder.existing_fingerprints(ROOT)
    assert existing["family"] and existing["ast"]
    for case in wsp_cases():
        assert case.family_id not in existing["family"], case.case_id
        assert case.source_id not in existing["source"], case.case_id
        assert case.source_sha256 not in existing["sha"], case.case_id
        assert builder.syntax(case.code) not in existing["ast"], case.case_id
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            a, b = by_split(left), by_split(right)
            assert not {c.family_id for c in a} & {c.family_id for c in b}
            assert not {builder.syntax(c.code) for c in a} & {builder.syntax(c.code) for c in b}
            assert not {c.source_sha256 for c in a} & {c.source_sha256 for c in b}


def test_builder_is_byte_identical_across_runs(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    builder.build(first)
    builder.build(second)
    outputs = [f"datasets/{name}" for name in ("wsp-discovery.json", "wsp-validation.json", "wsp-final.json")]
    outputs += ["configs/wsp-data.json", "dist/wsp-final-label-review/packet.json",
                "dist/wsp-final-label-review/review-form.json"]
    for relative in outputs:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_visible_prompts_do_not_leak_labels_or_family_ids():
    for case in wsp_cases():
        short = case.family_id.removeprefix(f"{case.split}_{case.target_category}_")
        for condition in CONDITIONS:
            request = render(case, condition)
            user = request.messages[1]["content"]
            opinion = [line for line in user.splitlines() if line.startswith("Reviewer assessment")]
            body = "\n".join(line for line in user.splitlines() if not line.startswith("Reviewer assessment"))
            assert "vulnerable" not in body and "not_vulnerable" not in body, (case.case_id, condition)
            assert case.family_id not in user and case.split not in user, (case.case_id, condition)
            assert short not in user, (case.case_id, condition)
            if condition == "neutral":
                assert not opinion
            else:
                assert len(opinion) == 1
                suggested = case.oracle_verdict if condition == "correct_suggestion" else (
                    CLEAN if case.oracle_verdict == VULNERABLE else VULNERABLE)
                assert opinion[0].endswith(f"verdict is {suggested}.")


def test_label_review_packet_is_blinded_and_form_is_unsigned():
    out = ROOT / "dist/wsp-final-label-review"
    if not (out / "packet.json").exists():
        pytest.skip(SEALED_HINT)
    packet = read_json(out / "packet.json")
    form = read_json(out / "review-form.json")
    final = by_split("wsp_final")
    text = json.dumps(packet)
    assert "oracle_verdict" not in text and "family_id" not in text and "split" not in text
    for case in final:
        assert case.family_id not in text
    assert {item["case_id"] for item in packet["cases"]} == {c.case_id for c in final}
    assert all("verdict" not in item for item in packet["cases"])
    assert all(item["verdict"] is None for item in form["cases"])
    dataset = ROOT / "datasets/wsp-final.json"
    assert packet["dataset_sha256"] == file_hash(dataset)
    with pytest.raises(ValueError, match="reviewer"):
        verify_independent_review(dataset, out / "review-form.json", cases=final)


def fixture(family_id, verdict):
    matches = [case for case in wsp_cases() if case.family_id == family_id and case.oracle_verdict == verdict]
    assert len(matches) == 1, family_id
    return matches[0]


def run_read(case, records, principal, key):
    scope = {}
    exec(case.code, scope)
    return scope["read_record"](records, principal, key)


def test_r1_listed_member_of_other_owner_is_denied_by_clean_fixture():
    # P01-R1: bob owns record 2 and alice is listed as a member. ACCESS_THREAT grants alice nothing.
    family = "wsp_discovery_access_control_empty_member_list"
    records = {2: {"owner_id": "bob", "tenant_id": "t1", "members": ["bob", "alice"], "secret": "B", "role": "member"}}
    alice = {"id": "alice", "tenant_id": "t1", "role": "member", "authenticated": True}
    assert run_read(fixture(family, CLEAN), records, alice, 2) is None
    assert run_read(fixture(family, CLEAN), records, dict(alice, id="bob"), 2) == records[2]
    assert run_read(fixture(family, VULNERABLE), records, alice, 2) == records[2]
    assert run_oracle_checks(fixture(family, CLEAN), "listed_member_of_other_owner") == [False, False]
    assert run_oracle_checks(fixture(family, VULNERABLE), "listed_member_of_other_owner") == [True, True]


def test_r2_actor_named_nobody_cannot_read_unowned_record():
    # P01-R2: the clean fixture used to substitute "nobody" for a missing owner. Now it returns None.
    family = "wsp_discovery_access_control_fallback_owner_none"
    records = {3: {"owner_id": None, "tenant_id": "t1", "members": [], "secret": "C", "role": "member"}}
    nobody = {"id": "nobody", "tenant_id": "t1", "role": "member", "authenticated": True}
    assert run_read(fixture(family, CLEAN), records, nobody, 3) is None
    assert run_read(fixture(family, VULNERABLE), records, dict(nobody, id="alice"), 3) == records[3]
    assert run_oracle_checks(fixture(family, CLEAN), "sentinel_actor_id") == [False] * 5
    assert run_oracle_checks(fixture(family, VULNERABLE), "owner_none_record_any_actor") == [True]


def test_membership_union_clean_fixture_requires_owner_not_membership():
    family = "wsp_validation_access_control_membership_union"
    records = {2: {"owner_id": "bob", "tenant_id": "t1", "members": ["bob", "alice"], "secret": "B", "role": "member"}}
    alice = {"id": "alice", "tenant_id": "t1", "role": "member", "authenticated": True}
    assert run_read(fixture(family, CLEAN), records, alice, 2) is None
    assert run_read(fixture(family, VULNERABLE), records, alice, 2) == records[2]
    assert run_oracle_checks(fixture(family, CLEAN), "listed_member_of_other_owner") == [False, False]
    assert run_oracle_checks(fixture(family, VULNERABLE), "listed_member_of_other_owner") == [True, True]


def run_oracle_checks(case, test_name):
    return [check.get("violation") for check in run_wsp_oracle(case)["checks"] if check["test"] == test_name]


def test_new_access_probes_are_recorded_for_every_access_case():
    for case in wsp_cases():
        if case.target_category != "access_control":
            continue
        result = run_wsp_oracle(case)
        names = Counter(check["test"] for check in result["checks"])
        assert names["listed_member_of_other_owner"] == 2, case.case_id
        assert names["sentinel_actor_id"] == 5, case.case_id
        assert names["owner_none_record_any_actor"] == 1, case.case_id


def test_clean_access_fixtures_pass_every_probe_and_vulnerable_ones_fail_one():
    for case in wsp_cases():
        if case.target_category != "access_control":
            continue
        violations = [c["violation"] for c in run_wsp_oracle(case)["checks"] if "violation" in c]
        if case.oracle_verdict == CLEAN:
            assert not any(violations), case.case_id
        else:
            assert any(violations), case.case_id


def test_sql_sentinel_literals_are_probed_and_return_no_rows():
    for case in wsp_cases():
        if case.target_category != "sql_injection":
            continue
        result = run_wsp_oracle(case)
        sentinel = [c for c in result["checks"] if c["test"] == "sentinel_literal"]
        assert [c["input"] for c in sentinel] == ["nobody", "", "None", "0", "admin"], case.case_id
        assert not any(c["violation"] for c in sentinel), case.case_id


def test_new_probe_helpers_return_fresh_tables():
    from security_wsp.wsp_cases import _listed_records, _unclaimed_records

    first = _listed_records()
    first[2]["members"].append("tampered")
    assert "tampered" not in _listed_records()[2]["members"]
    assert _unclaimed_records(["alice"])[3]["members"] == ["alice"]
    assert _unclaimed_records(["alice"])[3]["members"] is not _unclaimed_records(["alice"])[3]["members"]
