from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from security_wsp import neutral_study
from security_wsp.artifacts import file_hash, read_json, verify_run
from security_wsp.confirmation_data import confirmation_cases
from security_wsp.data import CLEAN, VULNERABLE, build_cases, load_dataset, run_oracle, write_dataset
from security_wsp.evidence_prompts import CANDIDATES, render_evidence
from security_wsp.models import GenerationResult

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_data_matches_commitment_and_executable_oracles():
    path = ROOT / "datasets/neutral-confirmation.json"
    protocol = read_json(ROOT / "configs/neutral-study.json")
    cases = load_dataset(path)
    assert file_hash(path) == protocol["confirmation_sha256"]
    assert tuple(cases) == confirmation_cases()
    assert Counter(case.oracle_verdict for case in cases) == {VULNERABLE: 20, CLEAN: 20}
    assert all(run_oracle(case)["passed"] for case in cases)


def test_second_confirmation_is_fresh_and_has_executable_oracles():
    import ast
    from security_wsp.confirmation_v2 import confirmation_cases_v2
    fresh = confirmation_cases_v2()
    exposed = (*build_cases(), *confirmation_cases())
    syntax = lambda case: ast.dump(ast.parse(case.code), include_attributes=False)
    assert len(fresh) == 40
    assert Counter(case.oracle_verdict for case in fresh) == {VULNERABLE: 20, CLEAN: 20}
    assert not {syntax(case) for case in exposed} & {syntax(case) for case in fresh}
    assert all(run_oracle(case)["passed"] for case in fresh)


def test_evidence_prompt_prefill_cannot_expose_the_neutral_label():
    case = build_cases()[0]
    for candidate in CANDIDATES:
        neutral = render_evidence(case, "neutral", candidate)
        assert neutral.messages == render_evidence(replace(case, oracle_verdict=CLEAN), "neutral", candidate).messages
        assert neutral.messages[-1] == {"role": "assistant", "content": '{"category":"sql_injection","explanation":"'}
        source = neutral.messages[1]["content"].split("<source>\n")[1].split("</source>")[0]
        assert "\n".join(line.split(" | ", 1)[1] for line in source.splitlines()) + "\n" == case.code


def test_frozen_adapter_prompts_preserve_schema_and_do_not_prefill_a_verdict():
    from security_wsp.adapter_review import VARIANTS, request_for
    case = build_cases()[0]
    for variant in VARIANTS:
        request = request_for(case, variant)
        assert [message["role"] for message in request.messages] == ["system", "user"]
        assert request.messages == request_for(replace(case, oracle_verdict=CLEAN), variant).messages
        assert "exactly: verdict, category, evidence_lines, explanation, confidence" in request.messages[0]["content"]


def test_simulation_inputs_have_no_expected_outcomes_or_reference_labels():
    pytest.importorskip("torch")
    from security_wsp.mental_review import simulation_request, final_request
    case = next(case for case in build_cases() if case.target_category == "access_control")
    request = simulation_request(case)
    assert request.messages == simulation_request(replace(case, oracle_verdict=CLEAN)).messages
    assert case.case_id not in str(request.messages)
    assert "oracle_verdict" not in str(request.messages)
    assert "mentally" in request.messages[0]["content"].lower()
    final = final_request(case, "Unverified example output")
    assert "Unverified example output" in final.messages[-1]["content"]
    assert "unverified working notes" in final.messages[-1]["content"]


def test_previous_answers_require_matching_model_hardware_and_messages(tmp_path):
    from types import SimpleNamespace
    import pytest
    from security_wsp.adapter_review import prior_plain, request_for
    from security_wsp.artifacts import RunStore, atomic_json
    cases = build_cases()[:2]
    metadata = {"adapter_sha256": "fixed", "gpu": {"name": "fixture"},
                "environment": {"packages": {"torch": {"version": "fixture"}}}}
    store = RunStore(tmp_path, {"fixture": True}, stage="fixture", evidence_kind="cohere_local_checkpoint")
    rows = [{"case_id": case.case_id, "messages": request_for(case, "plain").messages} for case in cases]
    atomic_json(tmp_path / "summary.json", {"model": metadata})
    atomic_json(tmp_path / "results.json", rows)
    store.finish([], ["summary.json", "results.json"])
    assert prior_plain(tmp_path, SimpleNamespace(metadata=metadata), cases) == rows
    changed = dict(metadata, gpu={"name": "different"})
    assert prior_plain(tmp_path, SimpleNamespace(metadata=changed), cases) is None
    changed = dict(metadata, adapter_sha256="different")
    with pytest.raises(ValueError, match="different adapter"):
        prior_plain(tmp_path, SimpleNamespace(metadata=changed), cases)


def test_failed_development_never_opens_confirmation(monkeypatch, tmp_path):
    class AlwaysVulnerable:
        metadata = {"test_model": True}

        def generate(self, requests):
            return [GenerationResult(request.request_id, request.messages[-1]["content"] +
                                     'Claim", "evidence_lines":[1],"verdict":"vulnerable","confidence":1}',
                                     1, 1, "stop") for request in requests]

    monkeypatch.setattr(neutral_study, "require_research_runtime", lambda: None)
    monkeypatch.setattr(neutral_study, "HuggingFaceAdapter", lambda _: AlwaysVulnerable())
    development = tmp_path / "development.json"
    write_dataset(development, build_cases()[:2])
    output = tmp_path / "study"
    result = neutral_study.run_neutral_study({"model_id": "fixture"}, read_json(ROOT / "configs/neutral-study.json"),
                                             development, tmp_path / "absent.json", output)
    assert result["development_balanced_accuracy"] == .5
    assert result["confirmation_completed"] is False
    assert verify_run(output / "development")["valid"]


def test_changed_study_data_is_rejected_before_model_loading(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from security_wsp import adapter_review, mental_review
    data = tmp_path / "changed.json"
    data.write_text("[]")
    protocol = {"development": str(data), "development_sha256": "wrong", "threshold": .75,
                "variants": list(adapter_review.VARIANTS)}
    for module in (adapter_review, mental_review):
        monkeypatch.setattr(module, "require_research_runtime", lambda: None)
        def forbidden(*args):
            pytest.fail("Invalid data must be rejected without buying model-loading time")
        monkeypatch.setattr(module, "HuggingFaceAdapter", forbidden)
        with pytest.raises(ValueError, match="Development"):
            getattr(module, "run_" + module.__name__.rsplit(".", 1)[-1])(
                {}, protocol, tmp_path / "checkpoint", tmp_path / "previous", tmp_path / module.__name__)
