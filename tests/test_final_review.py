from dataclasses import asdict, replace
from pathlib import Path
import pytest

from security_wsp.artifacts import atomic_json, file_hash, read_json
from security_wsp.data import build_cases
from security_wsp.final_review import prepare_review_packet, verify_independent_review
from security_wsp.workflow import final_evaluate


def final_inputs(tmp_path):
    # Existing development fixtures are used only as unit-test data.
    cases = [replace(c, split="final_test") for c in build_cases()[:2]]
    dataset = tmp_path / "fixture.json"
    atomic_json(dataset, {"cases": [asdict(case) for case in cases]})
    review = tmp_path / "review.json"
    contents = {
        "schema_version": 1, "dataset_sha256": file_hash(dataset), "reviewer_id": "test-reviewer",
        "independent_of_author_and_edit_selection": True, "blinded_to_oracle_labels": True,
        "blinded_to_model_outputs": True,
        "cases": [{"case_id": c.case_id, "source_sha256": c.source_sha256, "verdict": c.oracle_verdict,
                   "passed_test_ids": list(c.oracle_test_ids), "note": "Test fixture only."} for c in cases],
    }
    atomic_json(review, contents)
    return cases, dataset, review, contents


def test_review_packet_blinds_labels_and_binds_the_exact_case_file(tmp_path):
    _, dataset, review, _ = final_inputs(tmp_path)
    packet = tmp_path / "packet.json"
    prepare_review_packet(dataset, packet)
    document = read_json(packet)
    assert document["dataset_sha256"] == file_hash(dataset)
    assert all("oracle_verdict" not in row and "reviewer_id" not in row for row in document["cases"])
    assert verify_independent_review(dataset, review)["cases"] == 2


@pytest.mark.parametrize("change, error", [
    (lambda r: r.update(blinded_to_model_outputs=False), "attestation"),
    (lambda r: r["cases"][0].update(verdict="insufficient_information"), "disagreement"),
    (lambda r: r["cases"][0].update(passed_test_ids=[]), "behavioral checks"),
    (lambda r: r["cases"][0].update(source_sha256="stale"), "source version"),
    (lambda r: r["cases"].__setitem__(0, r["cases"][1]), "duplicate"),
])
def test_independent_review_rejects_unresolved_or_incomplete_evidence(tmp_path, change, error):
    _, dataset, review, contents = final_inputs(tmp_path)
    change(contents)
    atomic_json(review, contents)
    with pytest.raises(ValueError, match=error):
        verify_independent_review(dataset, review)


def test_final_evaluation_rejects_uncommitted_review_before_loading_model(tmp_path):
    _, dataset, review, _ = final_inputs(tmp_path)
    frozen = tmp_path / "frozen.json"
    atomic_json(frozen, {"final_dataset_sha256": file_hash(dataset), "final_review_sha256": "0"*64})
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="pre-freeze commitment"):
        final_evaluate(frozen, dataset, output, review)
    assert not output.exists()
