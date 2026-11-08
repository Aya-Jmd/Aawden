"""Evaluator-only handoff and content checks for independent final-case review."""
from __future__ import annotations

from pathlib import Path
from .artifacts import atomic_json, file_hash, read_json
from .data import CLEAN, VULNERABLE, load_dataset


def final_cases(dataset: Path):
    cases = load_dataset(dataset)
    if any(case.split != "final_test" for case in cases):
        raise ValueError("The evaluator packet requires final_test cases only")
    if {case.oracle_verdict for case in cases} != {CLEAN, VULNERABLE}:
        raise ValueError("The final set must include both oracle verdicts")
    return cases


def review_cases(dataset: Path, development: bool):
    if not development:
        return final_cases(dataset)
    cases = load_dataset(dataset)
    if any(case.split == "final_test" for case in cases):
        raise ValueError("The development review must stay separate from final_test")
    return cases


def prepare_review_packet(dataset: Path, packet: Path, *, development=False):
    """Give the reviewer code and test IDs while keeping author labels private."""
    cases = review_cases(dataset, development)
    content = {
        "schema_version": 1,
        "dataset_sha256": file_hash(dataset),
        "cases": [
            {"case_id": case.case_id, "source_sha256": case.source_sha256,
             "required_test_ids": case.oracle_test_ids, **case.visible()}
            for case in cases
        ],
    }
    atomic_json(packet, content)
    return {"cases": len(cases), "dataset_sha256": content["dataset_sha256"], "packet": str(packet)}


def verify_independent_review(dataset: Path, review: Path, cases=None, *, development=False):
    cases = review_cases(dataset, development) if cases is None else cases
    document = read_json(review)
    dataset_hash = file_hash(dataset)
    if document.get("schema_version") != 1 or document.get("dataset_sha256") != dataset_hash:
        raise ValueError("The independent review is not bound to this dataset")
    reviewer = document.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("The independent reviewer must supply a stable reviewer ID")
    for flag in ("independent_of_author_and_edit_selection", "blinded_to_oracle_labels", "blinded_to_model_outputs"):
        if document.get(flag) is not True:
            raise ValueError(f"Missing independent review attestation: {flag}")
    records = document.get("cases")
    if (not isinstance(records, list) or len(records) != len(cases)
            or not all(isinstance(record, dict) for record in records)):
        raise ValueError("The independent review must cover every case once")
    by_id = {item.get("case_id"): item for item in records}
    if set(by_id) != {case.case_id for case in cases} or len(by_id) != len(records):
        raise ValueError("The independent review contains missing, duplicate, or unexpected cases")
    for case in cases:
        record = by_id[case.case_id]
        if record.get("source_sha256") != case.source_sha256:
            raise ValueError(f"Reviewer used a different source version: {case.case_id}")
        if record.get("verdict") != case.oracle_verdict:
            raise ValueError(f"Reviewer/oracle disagreement must be resolved before freeze: {case.case_id}")
        tests = record.get("passed_test_ids")
        if (not isinstance(tests, list) or not all(isinstance(test, str) for test in tests)
                or not case.oracle_test_ids or sorted(tests) != sorted(case.oracle_test_ids)):
            raise ValueError(f"Independent behavioral checks are incomplete: {case.case_id}")
        if not isinstance(record.get("note"), str) or not record["note"].strip():
            raise ValueError(f"Reviewer did not document the evidence: {case.case_id}")
    return {
        "status": "evaluator_attested_and_content_verified",
        "reviewer_id": reviewer,
        "cases": len(cases),
        "dataset_sha256": dataset_hash,
        "review_sha256": file_hash(review),
        "scope": "Evidence and attestations checked; reviewer identity and operational separation are externally enforced.",
    }
