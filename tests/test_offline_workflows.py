import json
from dataclasses import asdict, replace

import pytest

from security_wsp import workflow
from security_wsp.artifacts import atomic_json, file_hash, read_json, source_fingerprint, verify_run
from security_wsp.data import build_cases
from security_wsp.models import GenerationResult
from security_wsp.prompts import PROMPT_VERSION, STRONG_VARIANTS
from security_wsp.regressions import score_grounding, write_grounding_packet


class FixtureAdapter:
    def __init__(self, config, cases, fail_at=None):
        self.config = config
        self.metadata = {"adapter": "offline_test_fixture"}
        self.cases = {case.case_id: case for case in cases}
        self.fail_at = fail_at
        self.requests = []

    def generate(self, requests):
        request, = requests
        self.requests.append(request)
        if len(self.requests) == self.fail_at:
            raise KeyboardInterrupt
        if request.condition == "coding_regression":
            text = '{"answer":true}' if self.config["model_id"] == "fixture/original" else '{"answer":1}'
        else:
            case = self.cases[request.case_id]
            verdict = case.oracle_verdict
            if request.condition == "incorrect_suggestion" and request.messages[0]["content"] == STRONG_VARIANTS["trace"]:
                verdict = "not_vulnerable_for_target" if verdict == "vulnerable" else "vulnerable"
            text = json.dumps({"verdict": verdict, "category": case.target_category,
                               "evidence_lines": [1], "explanation": "Test fixture.", "confidence": .5})
        return [GenerationResult(request.request_id, text, 10, 5, "stop")]


def test_prompt_choice_uses_validation_and_freezes_selected_wording(monkeypatch, tmp_path):
    cases = [replace(case, split="validation") for case in build_cases()[:2]]
    cases.append(replace(build_cases()[2], split="discovery"))
    config = {"model_id": "fixture/original", "max_new_tokens": 32, "seed": 42}
    adapter = FixtureAdapter(config, cases)
    monkeypatch.setattr(workflow, "require_research_runtime", lambda: None)
    monkeypatch.setattr(workflow, "HuggingFaceAdapter", lambda _: adapter)
    protocol = {"strong_prompt_candidates": ["trace", "verify"], "neutral_noninferiority_margin": .02}
    result = workflow.select_strong_prompt(cases, protocol, config, tmp_path)
    assert result["selected_protocol"]["strong_prompt_variant"] == "verify"
    assert {r.case_id for r in adapter.requests} == {c.case_id for c in cases[:2]}
    assert verify_run(tmp_path)["verified_artifacts"] == 14


def test_candidate_validation_resumes_missing_answers_and_reuses_completed_results(tmp_path):
    cases = build_cases()[:2]
    interrupted = FixtureAdapter({"model_id": "fixture/original"}, cases, fail_at=2)
    with pytest.raises(KeyboardInterrupt):
        workflow.run_validation(cases, interrupted, "B", {}, tmp_path, "fixed identity")
    resumed = FixtureAdapter(interrupted.config, cases)
    result = workflow.run_validation(cases, resumed, "B", {}, tmp_path, "fixed identity")
    assert len(result["rows"]) == 6
    assert len(resumed.requests) == 5
    assert resumed.requests[0].request_id == interrupted.requests[1].request_id
    completed = FixtureAdapter(interrupted.config, cases)
    assert workflow.run_validation(cases, completed, "B", {}, tmp_path, "fixed identity") == result
    assert completed.requests == []
    assert [a["status"] for a in read_json(tmp_path / "manifest.json")["attempts"]] == ["interrupted", "completed"]


def test_frozen_run_keeps_authority_coding_and_blinded_review_separate(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    cases = [replace(case, split="final_test") for case in build_cases()[:2]]
    dataset, labels, coding = [tmp_path / name for name in ("final.json", "labels.json", "coding.json")]
    atomic_json(dataset, {"cases": [asdict(case) for case in cases]})
    atomic_json(labels, {"schema_version": 1, "dataset_sha256": file_hash(dataset), "reviewer_id": "fixture",
                         "independent_of_author_and_edit_selection": True, "blinded_to_oracle_labels": True,
                         "blinded_to_model_outputs": True,
                         "cases": [{"case_id": c.case_id, "source_sha256": c.source_sha256,
                                    "verdict": c.oracle_verdict, "passed_test_ids": c.oracle_test_ids,
                                    "note": "Fixture evidence."} for c in cases]})
    atomic_json(coding, {"cases": [{"case_id": "coding", "code": "result = 2 > 1", "question": "What is result?", "expected": True}]})
    checkpoint = tmp_path / "edited"
    atomic_json(checkpoint / "aawden_edit.json", {"fixture": True})
    original = {"model_id": "fixture/original", "max_new_tokens": 32, "seed": 42}
    edited = dict(original, model_id=str(checkpoint))
    protocol = {"bootstrap_replicates": 100, "seed": 42, "minimum_gain": .05, "neutral_noninferiority_margin": .02,
                "screen_minimum_neutral_ba": .75, "screen_minimum_suggestion_drop": .10, "test_authority_wording": True}
    freeze = tmp_path / "freeze.json"
    atomic_json(freeze, {"protocol": protocol, "original_config": original, "edited_config": edited,
                         "random_control_recipes": [], "identity": {"config": original}, "selection_rule": "Fixture rule.",
                         "development_family_ids": [], "development_source_ids": [], "development_code_hashes": [],
                         "development_template_hashes": [], "final_dataset_sha256": file_hash(dataset),
                         "final_review_sha256": file_hash(labels), "coding_dataset_sha256": file_hash(coding),
                         "source_sha256": source_fingerprint(), "prompt_version": PROMPT_VERSION,
                         "edit_manifest_sha256": file_hash(checkpoint / "aawden_edit.json"), "checkpoint_hashes": {}})
    monkeypatch.setattr(workflow, "require_research_runtime", lambda: None)
    monkeypatch.setattr(workflow, "HuggingFaceAdapter", lambda config: FixtureAdapter(config, cases))

    def audit_fixture(rows, output):
        result = {"status": "offline_test_fixture", "samples": len(rows)}
        atomic_json(output, result)
        return result

    monkeypatch.setattr(workflow, "audit_records", audit_fixture)
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="Coding dataset differs"):
        workflow.final_evaluate(freeze, dataset, run, labels, dataset)
    assert not run.exists()
    summary = workflow.final_evaluate(freeze, dataset, run, labels, coding)
    assert summary["standard_prompt_comparisons"]["neutral"]["left"] == "C"
    assert len(read_json(run / "authority-results.json")) == 4
    assert summary["coding_regression"]["accuracy_difference"] == -1
    assert len(read_json(run / "results.json")) == 30
    packet_file = tmp_path / "packet.json"
    write_grounding_packet(run, packet_file)
    packet = read_json(packet_file)
    assert len(packet["explanations"]) == 12
    assert all("method" not in item and "oracle_verdict" not in item for item in packet["explanations"])
    review = tmp_path / "grounding.json"
    document = {"schema_version": 1, "packet_sha256": packet["packet_sha256"], "reviewer_id": "fixture",
                "independent_of_edit_selection": True, "blinded_to_model_arm": True,
                "explanations": [{"review_id": row["review_id"], "supported": True, "note": "Fixture decision."}
                                 for row in packet["explanations"]]}
    atomic_json(review, document)
    assert score_grounding(run, review)["arms"]["B"]["supported"] == 6
    document["explanations"].pop()
    atomic_json(review, document)
    with pytest.raises(ValueError, match="cover every explanation"):
        score_grounding(run, review)
