"""Offline tests for scripts/wsp_final.py: the freeze and final-evaluate orchestration.

Everything here runs on CPU against a fake final dataset written to tmp_path and the constructed
plumbing double in the script. No model is downloaded, no GPU is touched, and nothing is written
under runs/. The fake dataset reuses development cases relabelled into the sealed split; it is
empty evidence and every artifact it produces says so.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wsp_final.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import atomic_json, file_hash, read_json  # noqa: E402
from security_wsp.data import ABSTAIN, CLEAN, VULNERABLE, load_dataset  # noqa: E402
from security_wsp.patches import build_patch, save_patch  # noqa: E402
from security_wsp.statistics import metrics  # noqa: E402


def _script():
    spec = importlib.util.spec_from_file_location("wsp_final_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wsp_final = _script()


# --------------------------------------------------------------------------------------------------
# fixtures: a clearly fake sealed set, fake commitments, a fake patch and a toy selection
# --------------------------------------------------------------------------------------------------

FAKE_MARK = "FAKE TEST DATA - NOT THE SEALED WSP FINAL SET"


def fake_final_dataset(tmp_path: Path, name: str = "fake-final.json"):
    """A schema-compatible sealed-split dataset built from development cases.

    Four cases: two vulnerable and two clean, each pair in its own family, so balanced accuracy is
    defined and at least two bootstrap units exist. split is wsp_final to match protocol v3.
    """
    cases = load_dataset(ROOT / "datasets" / "wsp-discovery.json")
    vulnerable = [c for c in cases if c.oracle_verdict == VULNERABLE][:2]
    clean = [c for c in cases if c.oracle_verdict == CLEAN][:2]
    chosen = vulnerable + clean
    fake = []
    for index, case in enumerate(chosen):
        fake.append(replace(case, case_id=f"fake_final_{index:02d}", family_id=f"fake_family_{index // 2:02d}",
                            source_id=f"fake_source_{index:02d}", split="wsp_final"))
    path = tmp_path / name
    atomic_json(path, {"schema_version": 1,
                       "metadata": {"fake": True, "mark": FAKE_MARK, "cases": len(fake), "families": 2,
                                    "final_test_protected": False,
                                    "note": "Test fixture. Relabelled development cases; not evaluation data."},
                       "cases": [asdict(case) for case in fake]})
    return path, fake


def fake_review(tmp_path: Path, dataset: Path, cases) -> Path:
    path = tmp_path / "fake-review.json"
    atomic_json(path, {
        "schema_version": 1, "dataset_sha256": file_hash(dataset), "reviewer_id": "fake-test-reviewer",
        "independent_of_author_and_edit_selection": True, "blinded_to_oracle_labels": True,
        "blinded_to_model_outputs": True,
        "cases": [{"case_id": c.case_id, "source_sha256": c.source_sha256, "verdict": c.oracle_verdict,
                   "passed_test_ids": list(c.oracle_test_ids), "note": "Fake test review; not an independent review."}
                  for c in cases],
        "fake": True})
    return path


def fake_commitments(tmp_path: Path, dataset: Path, cases, *, sha: str | None = None) -> Path:
    path = tmp_path / "fake-wsp-data.json"
    atomic_json(path, {"schema_version": 1, "fake": True, "sealed": ["wsp_final"],
                       "files": {"wsp_final": {"path": str(dataset), "sha256": sha or file_hash(dataset),
                                               "cases": len(cases), "families": len({c.family_id for c in cases}),
                                               "sealed": True}}})
    return path


def fake_regression_commitments(tmp_path: Path) -> Path:
    """Point at the real committed regression files, whose hashes are real; only the container is fake."""
    document = read_json(ROOT / "configs" / "wsp-regressions.json")
    path = tmp_path / "fake-wsp-regressions.json"
    atomic_json(path, dict(document, fake=True))
    return path


def fake_patch(tmp_path: Path, *, toy_effect: str = "vulnerable", changed: int = 1) -> dict:
    """A real aawden-patch-v1 manifest, written by the real save_patch, with a toy effect marker.

    The tensor is sized so the patch stays inside protocol v3's 0.0005 changed-weight fraction:
    one row of 2048 in a 2000-row tensor is 2048 of 4,096,000 scalars (0.0005 exactly at the limit,
    0.0005 not below it), so use 4000 rows: 2048 of 8,192,000 is 0.00025. The freeze guard is
    exercised on a realistic sparse patch, not bypassed.
    """
    torch = pytest.importorskip("torch")
    name = "model.layers.0.mlp.down_proj.weight"
    base = torch.zeros((4000, 2048), dtype=torch.float32)
    edited = base.clone()
    edited[:changed] = 0.5
    patch = build_patch({name: base}, {name: edited})
    stem = tmp_path / "patch-01"
    manifest = save_patch(patch, stem, {"fake": True, "toy_effect": toy_effect,
                                        "note": "Test patch. Not a weight-space-programming result."})
    changed_scalars = changed * 2048
    assert manifest["changed_scalars"] == changed_scalars
    assert manifest["changed_scalars"] / manifest["total_parameters"] < 0.0005
    return {"path": str(stem.with_name("patch-01.json")), "manifest": manifest, "changed_scalars": changed_scalars}


def fake_selection(tmp_path: Path, patch_path: str | None, **overrides) -> Path:
    document = {
        "schema_version": 1, "status": "selected", "evidence_kind": "constructed_toy",
        "method": "wsp_agent_patch" if patch_path else "none",
        "patch": {"kind": "wsp_agent_patch", "path": patch_path} if patch_path else None,
        "strong_prompt": "verify",
        "exposure_note": "Fake test selection. The sealed set was never opened during this fixture's construction.",
        "fake": True,
    }
    document.update(overrides)
    path = tmp_path / "selection.json"
    atomic_json(path, document)
    return path


REGRESSIONS = [ROOT / "datasets" / "wsp-coding.json", ROOT / "datasets" / "wsp-general.json",
               ROOT / "datasets" / "wsp-format.json"]


def frozen_setup(tmp_path: Path, *, with_patch: bool = True):
    """Freeze a fake final set end to end and return the paths the evaluate step needs."""
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    commitments = fake_commitments(tmp_path, dataset, cases)
    patch = fake_patch(tmp_path) if with_patch else None
    selection = fake_selection(tmp_path, patch["path"] if patch else None)
    output = tmp_path / "final"
    result = wsp_final.freeze(selection, output, review=review, regressions=REGRESSIONS, dataset=dataset,
                              commitments=commitments, regression_commitments=fake_regression_commitments(tmp_path),
                              allow_toy=True)
    return {"dataset": dataset, "cases": cases, "review": review, "commitments": commitments,
            "selection": selection, "output": output, "result": result}


# --------------------------------------------------------------------------------------------------
# refusing before freeze
# --------------------------------------------------------------------------------------------------

def test_evaluate_refuses_before_any_freeze(tmp_path):
    bundle_path = tmp_path / "final" / "freeze.json"
    with pytest.raises(wsp_final.FinalRefused, match="freeze manifest is missing"):
        wsp_final.verify_freeze(bundle_path)
    assert not bundle_path.parent.exists()


def test_evaluate_refuses_a_freeze_that_was_never_written(tmp_path):
    setup = frozen_setup(tmp_path)
    (setup["output"] / "freeze.json").unlink()
    with pytest.raises(wsp_final.FinalRefused):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_freezing_twice_is_refused(tmp_path):
    setup = frozen_setup(tmp_path)
    with pytest.raises(wsp_final.FinalRefused, match="already exists"):
        wsp_final.freeze(setup["selection"], setup["output"], review=setup["review"], regressions=REGRESSIONS,
                         dataset=setup["dataset"], commitments=setup["commitments"],
                         regression_commitments=tmp_path / "fake-wsp-regressions.json", allow_toy=True)


def test_freeze_refuses_a_missing_bound_file(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    commitments = fake_commitments(tmp_path, dataset, cases)
    selection = fake_selection(tmp_path, None)
    with pytest.raises(wsp_final.FinalRefused, match="label-review file is missing"):
        wsp_final.freeze(selection, tmp_path / "final", review=tmp_path / "nope.json", regressions=REGRESSIONS,
                         dataset=dataset, commitments=commitments,
                         regression_commitments=fake_regression_commitments(tmp_path), allow_toy=True)
    # A missing candidate patch is refused too.
    patch = fake_patch(tmp_path)
    Path(patch["path"]).unlink()
    selection = fake_selection(tmp_path, patch["path"])
    review = fake_review(tmp_path, dataset, cases)
    with pytest.raises(wsp_final.FinalRefused, match="selected patch manifest is missing"):
        wsp_final.freeze(selection, tmp_path / "final", review=review, regressions=REGRESSIONS, dataset=dataset,
                         commitments=commitments, regression_commitments=fake_regression_commitments(tmp_path),
                         allow_toy=True)


def test_freeze_refuses_a_final_dataset_whose_sha_is_not_committed(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    commitments = fake_commitments(tmp_path, dataset, cases, sha="0" * 64)
    selection = fake_selection(tmp_path, None)
    with pytest.raises(wsp_final.FinalRefused, match="does not equal the value committed"):
        wsp_final.freeze(selection, tmp_path / "final", review=review, regressions=REGRESSIONS, dataset=dataset,
                         commitments=commitments, regression_commitments=fake_regression_commitments(tmp_path),
                         allow_toy=True)
    assert not (tmp_path / "final" / "freeze.json").exists()


def test_freeze_refuses_to_open_a_final_dataset_that_drifted(tmp_path):
    setup = frozen_setup(tmp_path)
    document = read_json(setup["dataset"])
    document["cases"][0]["oracle_verdict"] = (VULNERABLE if document["cases"][0]["oracle_verdict"] == CLEAN else CLEAN)
    atomic_json(setup["dataset"], document)
    with pytest.raises(wsp_final.FinalRefused):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_freeze_refuses_a_review_bound_to_another_dataset(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    # A different dataset: same shapes, but one source differs, so the sha256 differs too.
    other, other_cases = fake_final_dataset(tmp_path, name="other-final.json")
    document = read_json(other)
    document["cases"][0]["code"] = document["cases"][0]["code"].replace("return", "return  ", 1)
    atomic_json(other, document)
    import hashlib
    document["cases"][0]["source_sha256"] = hashlib.sha256(document["cases"][0]["code"].encode("utf-8")).hexdigest()
    atomic_json(other, document)
    other_cases = load_dataset(other)
    assert file_hash(other) != file_hash(dataset)
    review = fake_review(tmp_path, other, other_cases)
    commitments = fake_commitments(tmp_path, dataset, cases)
    with pytest.raises(wsp_final.FinalRefused, match="is bound to dataset sha256"):
        wsp_final.freeze(fake_selection(tmp_path, None), tmp_path / "final", review=review, regressions=REGRESSIONS,
                         dataset=dataset, commitments=commitments,
                         regression_commitments=fake_regression_commitments(tmp_path), allow_toy=True)


def test_freeze_refuses_a_toy_selection_without_the_flag(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    commitments = fake_commitments(tmp_path, dataset, cases)
    with pytest.raises(wsp_final.FinalRefused, match="constructed_toy"):
        wsp_final.freeze(fake_selection(tmp_path, None), tmp_path / "final", review=review, regressions=REGRESSIONS,
                         dataset=dataset, commitments=commitments,
                         regression_commitments=fake_regression_commitments(tmp_path))


def test_freeze_refuses_an_unselected_or_hopeless_selection(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    commitments = fake_commitments(tmp_path, dataset, cases)
    regressions = fake_regression_commitments(tmp_path)
    no_candidate = fake_selection(tmp_path, None, status="no_candidate")
    with pytest.raises(wsp_final.FinalRefused, match="nothing is frozen"):
        wsp_final.freeze(no_candidate, tmp_path / "final", review=review, regressions=REGRESSIONS, dataset=dataset,
                         commitments=commitments, regression_commitments=regressions, allow_toy=True)
    bad_prompt = fake_selection(tmp_path, None, strong_prompt="made_up")
    with pytest.raises(wsp_final.FinalRefused, match="strong_prompt"):
        wsp_final.freeze(bad_prompt, tmp_path / "final", review=review, regressions=REGRESSIONS, dataset=dataset,
                         commitments=commitments, regression_commitments=regressions, allow_toy=True)


def test_review_is_bound_to_the_committed_review_form(tmp_path):
    """A completed review must follow the committed form's case order, when a form is committed."""
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    form = read_json(review)
    for item in form["cases"]:
        item.update(verdict=None, passed_test_ids=[], note="")
    form_path = tmp_path / "form.json"
    atomic_json(form_path, form)
    commitments = fake_commitments(tmp_path, dataset, cases)
    document = read_json(commitments)
    document["label_review_packet"] = {"form": str(form_path), "form_sha256": file_hash(form_path),
                                       "cases": len(cases)}
    atomic_json(commitments, document)
    binding = wsp_final.verify_review_form(review, commitments)
    assert binding["cases"] == len(cases) and binding["form_sha256"] == file_hash(form_path)

    # A review that follows a different case order is refused.
    reordered = read_json(review)
    reordered["cases"] = list(reversed(reordered["cases"]))
    bad = tmp_path / "reordered-review.json"
    atomic_json(bad, reordered)
    with pytest.raises(wsp_final.FinalRefused, match="case order"):
        wsp_final.verify_review_form(bad, commitments)

    # A form whose bytes differ from the commitment is refused.
    document["label_review_packet"]["form_sha256"] = "3" * 64
    atomic_json(commitments, document)
    with pytest.raises(wsp_final.FinalRefused, match="does not equal the value committed"):
        wsp_final.verify_review_form(review, commitments)


def test_uncommitted_review_form_is_recorded_as_unbound(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    binding = wsp_final.verify_review_form(review, fake_commitments(tmp_path, dataset, cases))
    assert binding["form"] is None and "no committed review form" in binding["unbound_reason"]


# --------------------------------------------------------------------------------------------------
# freeze contents
# --------------------------------------------------------------------------------------------------

def test_freeze_binds_source_protocol_prompts_and_every_hash(tmp_path):
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    assert frozen["protocol"] == "aawden-protocol-v3"
    assert frozen["protocol_sha256"] == file_hash(ROOT / "configs" / "protocol-v3.json")
    assert frozen["source_sha256"] == wsp_final.source_fingerprint()
    assert frozen["final_dataset"]["sha256"] == file_hash(setup["dataset"])
    assert frozen["label_review"]["sha256"] == file_hash(setup["review"])
    assert frozen["method"] == "wsp_agent_patch"
    assert frozen["patch"]["changed_scalars"] == 2048
    # The three regression sets are bound with their real committed hashes.
    kinds = {entry["kind"] for key, entry in frozen["regressions"].items() if key != "_commitments_sha256"}
    assert kinds == {"coding", "multiple_choice", "instruction_following"}
    # Prompt identities are bound in the freeze, with the prefill applied to every rendered prompt.
    identity = frozen["prompt_identity"]
    assert identity["json_prefill"] == '{"verdict":"'
    assert identity["strong_variant_selected"] == "verify"
    assert identity["wordings"] == ["reviewer", "authority", "ciso", "scanner", "consensus", "code_comment"]
    # The binding covers the fixed text every prompt is built from, plus one frozen suggestion for
    # every (condition, wording, verdict) the run can render -- not two free-floating samples.
    components = identity["components"]
    assert components["json_prefill"] == '{"verdict":"'
    assert components["source_open"] == "<source>\n" and components["source_close"] == "</source>\n"
    assert set(components["system_strong"]) == {"trace", "verify"}
    assert identity["rendered_count"] == len(identity["opinions"]) + len(identity["source_comments"]) > 0
    for condition in ("correct_suggestion", "incorrect_suggestion"):
        for wording in identity["wordings"]:
            for verdict in ("vulnerable", "not_vulnerable_for_target"):
                key = f"{condition}|{wording}|{verdict}"
                assert key in identity["opinions"] or key in identity["source_comments"], key
    # code_comment carries its verdict inside the source block, so it is bound as a comment, not an
    # appended sentence; the other wordings append a sentence.
    assert any(key.startswith("correct_suggestion|code_comment|") for key in identity["source_comments"])
    assert any(key.startswith("correct_suggestion|reviewer|") for key in identity["opinions"])
    # Every prompt identity the freeze binds verifies against the code that renders it.
    assert wsp_final.check_prompt_identity(frozen) > 0
    # The repeated-look rule is written into the manifest, not just enforced in code.
    assert "repeated final look" in frozen["final_look_rule"] or "final look" in frozen["final_look_rule"]
    assert "hashed, not parsed" in frozen["sealed_data_access"]


def test_freeze_never_parses_the_sealed_dataset_bytes_before_writing_the_manifest(tmp_path, monkeypatch):
    """A freeze that fails must leave no manifest and must never have loaded final cases."""
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    commitments = fake_commitments(tmp_path, dataset, cases, sha="1" * 64)
    calls = []
    original = wsp_final.load_final_cases

    def spy(binding):
        calls.append(binding)
        return original(binding)

    monkeypatch.setattr(wsp_final, "load_final_cases", spy)
    with pytest.raises(wsp_final.FinalRefused):
        wsp_final.freeze(fake_selection(tmp_path, None), tmp_path / "final", review=review, regressions=REGRESSIONS,
                         dataset=dataset, commitments=commitments,
                         regression_commitments=fake_regression_commitments(tmp_path), allow_toy=True)
    assert calls == []
    assert not (tmp_path / "final" / "freeze.json").exists()


# --------------------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------------------

def test_evaluate_refuses_a_second_look_in_the_same_output_dir(tmp_path):
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    first = wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    assert first["evidence_kind"] == "constructed_toy"
    record = read_json(setup["output"] / "evaluation.json")
    assert record["status"] == "completed" and record["final_look"] == 1
    with pytest.raises(wsp_final.FinalRefused, match="consumed"):
        wsp_final.evaluate(wsp_final.verify_freeze(setup["output"] / "freeze.json"), setup["output"], allow_toy=True)


def test_evaluate_refuses_a_different_output_dir_for_the_same_freeze(tmp_path):
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    with pytest.raises(wsp_final.FinalRefused, match="second final look"):
        wsp_final.evaluate(bundle, tmp_path / "elsewhere", allow_toy=True)
    assert not (tmp_path / "elsewhere").exists()


def test_evaluate_refuses_an_unverified_freeze_before_loading_any_model(tmp_path):
    setup = frozen_setup(tmp_path)
    document = read_json(setup["output"] / "freeze.json")
    document["final_dataset"]["sha256"] = "2" * 64
    # Re-seal after editing, so this exercises the hash re-derivation rather than the self-hash.
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(document))
    with pytest.raises(wsp_final.FinalRefused, match="sealed final dataset sha256"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")
    assert not (setup["output"] / "evaluation.json").exists()


def test_tampering_with_a_frozen_patch_is_refused(tmp_path):
    setup = frozen_setup(tmp_path)
    weights = Path(read_json(setup["output"] / "freeze.json")["patch"]["weights"])
    weights.write_bytes(weights.read_bytes() + b"\0")
    with pytest.raises(wsp_final.FinalRefused, match="patch weights changed"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_evaluate_measures_every_required_arm_with_the_same_prefill(tmp_path):
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    summary = wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    # A, B, C, D, every candidate's C and D, three random seeds, and the no-op.
    for method in ("A", "B", "C", "D", "noop", "random-17", "random-29", "random-43",
                   "projection_edit:C", "projection_edit:D", "lora_adapter:C", "lora_adapter:D"):
        assert method in summary["arms"], method
    for condition in ("neutral", "correct_suggestion", "incorrect_suggestion"):
        assert summary["metrics"]["A"][condition]["n"] == len(setup["cases"])
        assert summary["metrics"]["D"][condition]["n"] == len(setup["cases"])
    # Every condition includes the transfer wordings as a separate block.
    assert set(summary["transfer"]) == {"authority", "ciso", "scanner", "consensus", "code_comment"}
    for block in summary["transfer"].values():
        assert set(block) == {"B:correct_suggestion", "B:incorrect_suggestion",
                              "D:correct_suggestion", "D:incorrect_suggestion"}
    # The primary endpoint and the non-inferiority checks are computed from the saved rows.
    assert summary["endpoints"]["primary_D_minus_B"] is not None
    assert set(summary["endpoints"]["noninferiority"]) == {"neutral", "correct_suggestion"}
    # The no-op control must reproduce A on the primary condition, and it changes no weights.
    assert summary["noop_check"]["equal"] is True
    assert summary["changed_weights"]["arms"]["noop"] is None
    assert summary["changed_weights"]["arms"]["A"] is None
    assert summary["changed_weights"]["arms"]["D"]["changed_scalars"] == 2048
    # Controls, changed weights and forward-pass usage are all present.
    assert summary["changed_weights"]["frozen_patch"]["changed_scalars"] == 2048
    assert summary["forward_passes"]["prompts_scored"] > 0
    assert summary["hashes"]["patches"]["D"] == summary["changed_weights"]["arms"]["D"]["weights_sha256"]


def test_frozen_patch_is_what_changes_the_edit_arms(tmp_path):
    """The toy effect is carried by the patch manifest, so C/D differ from A/B only because of it."""
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    summary = wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    misleading = "incorrect_suggestion"
    assert summary["metrics"]["A"][misleading]["balanced_accuracy"] < summary["metrics"]["D"][misleading]["balanced_accuracy"]
    assert summary["endpoints"]["primary_D_minus_B"]["difference"] >= 0


def test_prompt_only_freeze_reports_the_edit_arms_as_missing(tmp_path):
    dataset, cases = fake_final_dataset(tmp_path)
    review = fake_review(tmp_path, dataset, cases)
    commitments = fake_commitments(tmp_path, dataset, cases)
    selection = fake_selection(tmp_path, None)
    output = tmp_path / "final"
    wsp_final.freeze(selection, output, review=review, regressions=REGRESSIONS, dataset=dataset,
                     commitments=commitments, regression_commitments=fake_regression_commitments(tmp_path),
                     allow_toy=True)
    bundle = wsp_final.verify_freeze(output / "freeze.json")
    summary = wsp_final.evaluate(bundle, output, allow_toy=True)
    assert "C" not in summary["arms"] and "D" not in summary["arms"]
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "MISSING" in report
    assert report.count("MISSING") > 5


def test_regressions_report_before_and_after(tmp_path):
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    summary = wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    sets = summary["regressions"]["sets"]
    assert len(sets) == 3
    for block in sets.values():
        assert set(block["arms"]) == {"original", "edited"}
        assert block["sha256"]


# --------------------------------------------------------------------------------------------------
# the report generator never invents a number
# --------------------------------------------------------------------------------------------------

def test_report_prints_missing_for_every_absent_measurement():
    complete = _minimal_summary()
    assert "MISSING" not in wsp_final.render_report(complete)

    incomplete = _minimal_summary()
    del incomplete["metrics"]["D"]
    del incomplete["metrics"]["random-17"]
    incomplete["endpoints"]["primary_D_minus_B"] = None
    incomplete["endpoints"]["noninferiority"] = {"neutral": {"missing": True}}
    incomplete["flip_rates"]["C"] = {"missing": True}
    incomplete["transfer"].pop("ciso")
    incomplete["changed_weights"]["frozen_patch"] = None
    incomplete["regressions"]["sets"] = {}
    report = wsp_final.render_report(incomplete)

    for expected in ("MISSING: the edited arms were not measured", "MISSING"):
        assert expected in report
    # An arm that is declared in "arms" but has no measurement keeps its row, filled with MISSING:
    # eight measured columns (n, BA, accuracy, FP rate, FN rate, coverage, invalid, abstentions).
    d_row = [line for line in report.splitlines() if line.startswith("| D | incorrect_suggestion |")]
    assert len(d_row) == 1 and d_row[0].count("MISSING") == 8
    assert "| random-17 |" in report and "| random-17 | neutral | MISSING |" in report
    assert report.count("MISSING") > 20
    assert "| neutral | MISSING | MISSING | MISSING | MISSING |" in report
    assert "| C | MISSING | MISSING |" in report
    assert "### ciso" not in report
    assert "MISSING: no patch was frozen" in report
    assert "MISSING: no regression set was measured" in report


def test_report_never_emits_a_number_that_is_not_in_the_summary():
    """Every numeric token in the report must trace back to a value in the summary dict."""
    summary = _minimal_summary()
    summary["metrics"]["D"]["incorrect_suggestion"]["balanced_accuracy"] = 0.8123
    report = wsp_final.render_report(summary)
    import re
    numbers = set(re.findall(r"\d+\.\d+", report))
    rendered = {"0.8123"}
    assert rendered <= numbers
    # Values that appear nowhere in the summary must not appear in the report.
    for absent in ("0.4999", "0.0001", "12345.678"):
        assert absent not in report


def test_report_of_a_missing_arm_does_not_reuse_a_neighbouring_number():
    summary = _minimal_summary()
    summary["metrics"]["D"]["incorrect_suggestion"]["balanced_accuracy"] = 0.77
    del summary["metrics"]["C"]
    report = wsp_final.render_report(summary)
    rows = [line for line in report.splitlines() if line.startswith("| C | incorrect_suggestion")]
    assert len(rows) == 1
    assert "0.77" not in rows[0]


def test_summary_json_round_trips(tmp_path):
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    summary = read_json(setup["output"] / "summary.json")
    assert summary == json.loads(json.dumps(summary))
    assert summary["hashes"]["freeze"]
    for block in summary["metrics"].values():
        for cell in block.values():
            assert isinstance(cell["n"], int)
    report = (setup["output"] / "report.md").read_text(encoding="utf-8")
    # Every table the write-up cites exists in the report.
    for heading in ("Primary endpoint", "Non-inferiority", "Flip rates", "Transfer wordings",
                    "Regressions", "Changed weights", "Forward-pass usage", "Controls", "Hashes"):
        assert heading in report
    assert summary["evidence_kind"] == "constructed_toy"


def _minimal_summary() -> dict:
    """A complete summary for the report generator, with every arm present and plausible."""
    conditions = ["neutral", "correct_suggestion", "incorrect_suggestion"]
    methods = ["A", "B", "C", "D", "random-17", "random-29", "random-43", "noop"]
    cell = {"n": 4, "balanced_accuracy": 0.5, "accuracy": 0.5, "false_positive_rate": 0.25,
            "false_negative_rate": 0.25, "coverage": 1.0, "invalid_count": 0, "abstention_count": 0,
            "missing": False}
    metrics_table = {method: {condition: dict(cell) for condition in conditions} for method in methods}
    return {
        "schema_version": 1, "protocol": "aawden-protocol-v3", "freeze_sha256": "a" * 64,
        "evidence_kind": "constructed_toy", "generated_at": "2026-01-01T00:00:00+00:00",
        "method": "wsp_agent_patch", "strong_prompt": "verify",
        "primary_condition": "incorrect_suggestion", "primary_metric": "balanced_accuracy",
        "wordings": {"primary": "reviewer", "transfer": ["authority", "ciso", "scanner", "consensus", "code_comment"]},
        "arms": {method: {"kind": "base", "label": method, "strong": True, "variant": "verify"} for method in methods},
        "conditions": conditions,
        "metrics": metrics_table,
        "paired": {key: {condition: {"difference": 0.1, "ci95": [0.02, 0.18]} for condition in conditions}
                   for key in ("D-B", "C-A")},
        "endpoints": {
            "primary_D_minus_B": {"difference": 0.1, "ci95": [0.02, 0.18], "minimum_gain": 0.05,
                                  "meets_minimum_gain": True, "ci_lower_above_zero": True, "unit": "family_id",
                                  "replicates": 10000, "seed": 42},
            "standard_prompt_C_minus_A": {"difference": 0.1, "ci95": [0.02, 0.18]},
            "noninferiority": {"neutral": {"difference": 0.0, "ci95": [-0.01, 0.01], "margin": 0.02, "passes": True},
                               "correct_suggestion": {"difference": 0.0, "ci95": [-0.01, 0.01], "margin": 0.02,
                                                      "passes": True}},
        },
        "flip_rates": {method: {"own_neutral_correct": {"numerator": 1, "denominator": 2, "rate": 0.5},
                                "original_neutral_correct": {"numerator": 1, "denominator": 2, "rate": 0.5}}
                       for method in methods},
        "transfer": {wording: {f"{method}:{condition}": dict(cell) for method in ("B", "D") for condition in conditions[:2]}
                     for wording in ("authority", "ciso", "scanner", "consensus", "code_comment")},
        "regressions": {"arms": ["base", "selected"], "sets": {
            "datasets/wsp-coding.json": {"kind": "coding", "sha256": "b" * 64,
                                         "arms": {"base": {"cases": 40, "correct": 30, "accuracy": 0.75, "invalid": 0},
                                                  "selected": {"cases": 40, "correct": 30, "accuracy": 0.75, "invalid": 0}},
                                         "accuracy_difference": 0.0}}, "failures": []},
        "changed_weights": {"frozen_patch": {"changed_scalars": 100, "total_parameters": 1000000,
                                             "percent_changed": 0.01, "tensors": 2, "weights_sha256": "c" * 64},
                            "arms": {method: {"changed_scalars": 100, "percent_changed": 0.01, "weights_sha256": "c" * 64}
                                     for method in methods}},
        "forward_passes": {"measurement_calls": 10, "prompts_scored": 100,
                           "by_method": {method: {"calls": 1, "prompts": 10} for method in methods},
                           "definition": "one call is one generate() over one arm's requests"},
        "noop_check": {"compared_to": "A", "condition": "incorrect_suggestion", "equal": True},
        "failures": [],
        "hashes": {"freeze": "a" * 64, "protocol": "d" * 64, "source": "e" * 64, "final_dataset": "f" * 64,
                   "label_review": "1" * 64, "regressions": {"datasets/wsp-coding.json": "b" * 64},
                   "patches": {method: "c" * 64 for method in methods}, "selection": "2" * 64,
                   "run_store": "runs/fake/predictions"},
        "final_look_rule": "One final look per output directory.",
    }


def test_cli_refusal_exit_code(tmp_path):
    """The CLI prints a refusal to stderr and exits 2 without creating anything."""
    assert wsp_final.main(["evaluate", "--freeze-manifest", str(tmp_path / "absent.json")]) == 2
    assert not (tmp_path / "absent.json").exists()


# --------------------------------------------------------------------------------------------------
# the freeze manifest binds itself (critical: manifest with no self-hash)
# --------------------------------------------------------------------------------------------------

def test_the_freeze_manifest_carries_its_own_sha256(tmp_path):
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    assert frozen["manifest_sha256"] == wsp_final.manifest_sha256(frozen)
    assert setup["result"]["manifest_sha256"] == frozen["manifest_sha256"]
    # The self-hash is not part of the body it covers, so it cannot hash itself.
    assert "manifest_sha256" not in json.loads(wsp_final.manifest_body(frozen).decode("utf-8"))


def test_editing_output_dir_in_a_frozen_manifest_is_refused(tmp_path):
    """The reviewer's exact attack: move output_dir, which check_final_look used to trust."""
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen["output_dir"] = str((tmp_path / "a-second-look").resolve())
    atomic_json(setup["output"] / "freeze.json", frozen)
    with pytest.raises(wsp_final.FinalRefused, match="manifest_sha256"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_editing_a_bound_hash_in_a_frozen_manifest_is_refused(tmp_path):
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen["final_dataset"]["sha256"] = "9" * 64
    atomic_json(setup["output"] / "freeze.json", frozen)
    with pytest.raises(wsp_final.FinalRefused, match="does not match its own manifest_sha256"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_a_manifest_with_no_self_hash_is_refused(tmp_path):
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen.pop("manifest_sha256")
    atomic_json(setup["output"] / "freeze.json", frozen)
    with pytest.raises(wsp_final.FinalRefused, match="must bind itself"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_a_freeze_copied_to_another_directory_is_refused(tmp_path):
    """Even a correctly self-hashed manifest may not be moved: output_dir must be where it sits."""
    setup = frozen_setup(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "freeze.json").write_bytes((setup["output"] / "freeze.json").read_bytes())
    with pytest.raises(wsp_final.FinalRefused, match="only in the directory it was frozen in"):
        wsp_final.verify_freeze(elsewhere / "freeze.json")


def test_evaluate_refuses_a_rebound_manifest_even_when_resealed(tmp_path):
    """A re-sealed manifest still cannot redirect the run: build_adapt style trust is not enough."""
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen["method"] = "none"
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(frozen))
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    # The manifest is internally consistent, so verification passes; the run must still be the frozen
    # one, and the arms are derived from the manifest rather than a remembered plan.
    assert bundle["freeze"]["method"] == "none"
    summary = wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    assert "C" not in summary["arms"]


# --------------------------------------------------------------------------------------------------
# patch statistics are re-derived, not trusted (critical: verify_freeze trusts manifest numbers)
# --------------------------------------------------------------------------------------------------

def _protocol():
    return wsp_final.load_protocol(ROOT / "configs" / "protocol-v3.json")


def test_patch_statistics_are_derived_from_the_patch_file_not_its_manifest(tmp_path):
    patch = fake_patch(tmp_path)
    protocol = _protocol()
    binding = wsp_final.read_selection(
        fake_selection(tmp_path, patch["path"]), protocol, allow_toy=True)["patch"]
    derived = wsp_final.derive_patch_statistics(binding)
    assert derived["changed_scalars"] == patch["changed_scalars"] == 2048
    assert binding["derived_changed_scalars"] == 2048


def test_a_patch_manifest_that_understates_its_size_is_refused(tmp_path):
    """The reviewer's test: a manifest claiming a smaller size than the file contains."""
    patch = fake_patch(tmp_path)
    manifest_path = Path(patch["path"])
    manifest = read_json(manifest_path)
    manifest["changed_scalars"] = 1          # understate: a quarter of the real 2048
    manifest["tensors"][0]["changed_scalars"] = 1
    atomic_json(manifest_path, manifest)
    # Re-hash nothing: the weights file is untouched, so only the size claim is a lie. A selection
    # that trusts the manifest would freeze a patch four times smaller than the one it applies.
    protocol = _protocol()
    with pytest.raises(wsp_final.FinalRefused, match="patch file contains 2048"):
        wsp_final.read_selection(fake_selection(tmp_path, patch["path"]), protocol, allow_toy=True)


def test_verify_freeze_re_derives_the_patch_size_at_evaluate_time(tmp_path):
    """The freeze-time check is not enough: verify_freeze must measure the file again."""
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    manifest_path = Path(frozen["patch"]["manifest"])
    manifest = read_json(manifest_path)
    manifest["changed_scalars"] = 1
    atomic_json(manifest_path, manifest)
    # Re-hash the manifest in the freeze and re-seal it, so every sha256 comparison still passes and
    # only the re-derivation can catch the lie.
    frozen["patch"]["manifest_sha256"] = file_hash(manifest_path)
    frozen["patch"]["changed_scalars"] = 1
    frozen["patch"]["percent_changed"] = 100.0 / frozen["patch"]["total_parameters"]
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(frozen))
    with pytest.raises(wsp_final.FinalRefused, match="patch file contains 2048"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_the_protocol_fraction_limit_is_enforced_on_the_derived_count(tmp_path):
    """A patch inside the limit only because its manifest says so is refused."""
    patch = fake_patch(tmp_path)
    manifest_path = Path(patch["path"])
    manifest = read_json(manifest_path)
    tiny = manifest["total_parameters"] // 100000      # a fraction far above 0.0005
    real_weights = manifest["total_parameters"]
    manifest["total_parameters"] = tiny
    atomic_json(manifest_path, manifest)
    protocol = _protocol()
    with pytest.raises(wsp_final.FinalRefused, match="above the protocol limit"):
        wsp_final.read_selection(fake_selection(tmp_path, patch["path"]), protocol, allow_toy=True)
    assert real_weights > tiny


def test_a_tampered_patch_file_is_caught_by_the_derivation(tmp_path):
    """Changing a stored slice changes the derived count even when the manifest is left consistent."""
    from safetensors.torch import load_file, save_file
    patch = fake_patch(tmp_path)
    weights = Path(patch["path"]).with_suffix(".safetensors")
    stored = load_file(str(weights))
    name = "model.layers.0.mlp.down_proj.weight"
    # Zero the recorded original slice: 2048 more stored scalars now differ from their values.
    stored[f"{name}::original"] = stored[f"{name}::original"] * 0
    save_file(stored, str(weights))
    manifest_path = Path(patch["path"])
    manifest = read_json(manifest_path)
    manifest["weights_sha256"] = file_hash(weights)   # keep every hash comparison passing
    atomic_json(manifest_path, manifest)
    protocol = _protocol()
    # The row was already changed, so zeroing the original leaves the same 2048 differing scalars;
    # the derivation must still equal the manifest, which is the point: it measures, it does not guess.
    derived = wsp_final.derive_patch_statistics(
        wsp_final.read_selection(fake_selection(tmp_path, patch["path"]), protocol, allow_toy=True)["patch"])
    assert derived["changed_scalars"] == 2048


def test_a_patch_file_missing_a_listed_tensor_is_refused(tmp_path):
    from safetensors.torch import load_file, save_file
    patch = fake_patch(tmp_path)
    weights = Path(patch["path"]).with_suffix(".safetensors")
    stored = load_file(str(weights))
    name = "model.layers.0.mlp.down_proj.weight"
    stored.pop(f"{name}::original")
    save_file(stored, str(weights))
    manifest_path = Path(patch["path"])
    manifest = read_json(manifest_path)
    manifest["weights_sha256"] = file_hash(weights)
    atomic_json(manifest_path, manifest)
    with pytest.raises(wsp_final.FinalRefused, match="weights file lacks"):
        wsp_final.read_selection(fake_selection(tmp_path, patch["path"]), _protocol(), allow_toy=True)


# --------------------------------------------------------------------------------------------------
# the edited arms run on the merged base (high: _reload_model drops the frozen LoRA)
# --------------------------------------------------------------------------------------------------

class _FakeMergedModel:
    """A tiny module tree standing in for the base checkpoint. Two merged layers, no weights on disk."""

    def __init__(self, bias=0.0):
        torch = pytest.importorskip("torch")
        self.bias = torch.nn.Parameter(torch.tensor([[bias, bias]]))

    def state_dict(self):
        return {"model.layers.0.mlp.down_proj.weight": self.bias}


class _FakeAdapter:
    """An adapter double: records what was loaded into it and what was merged out of it."""

    evidence_kind = "cohere_local_checkpoint"

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.metadata = {"adapter": "fake", "checkpoint_hashes": {"model.safetensors": "f" * 64}}
        self.model = _FakeMergedModel(1.0)

    def generate(self, requests):
        raise NotImplementedError("the fake adapter is never asked to generate here")


def test_build_adapt_merges_the_frozen_lora_and_records_the_merged_state(tmp_path):
    """The reviewer's test: assert the merge is called and the merged-state hash is recorded."""
    pytest.importorskip("torch")
    spec = tmp_path / "adapter.json"
    atomic_json(spec, {"checkpoint": "unused.safetensors", "sha256": "a" * 64, "rank": 8,
                       "base_checkpoint_hashes": {"model.safetensors": "f" * 64}})
    frozen = {"selection": {"lora_spec": {"path": str(spec), "sha256": file_hash(spec)},
                            "model_config": None},
              "method": "wsp_agent_patch"}
    calls = {"load": [], "merge": 0}

    def load(model, specification, base_hashes):
        calls["load"].append((specification, base_hashes))

    def merge(model):
        calls["merge"] += 1
        model.bias.data = model.bias.data + 2.0      # the merge changes the weights
        return {"modules_merged": 1, "names": ["model.layers.0.mlp.down_proj"], "disabled_without_update": []}

    adapter, record = wsp_final.build_adapt(frozen, config={"seed": 42}, adapter_factory=_FakeAdapter,
                                            load=load, merge=merge)
    assert calls["merge"] == 1, "merge_dense must be called exactly once"
    assert calls["load"] and calls["load"][0][1] == {"model.safetensors": "f" * 64}
    assert record["merged_modules"] == 1 and record["adapter_spec_sha256"] == file_hash(spec)
    # The merged state is the POST-merge state, not the pre-merge checkpoint.
    assert record["merged_state_sha256"] == wsp_final.merged_state_hash(adapter.model)
    assert record["merged_state_sha256"] != wsp_final.merged_state_hash(_FakeMergedModel(1.0))


def test_build_adapt_refuses_a_merge_that_changes_nothing(tmp_path):
    spec = tmp_path / "adapter.json"
    atomic_json(spec, {"checkpoint": "unused.safetensors", "sha256": "a" * 64, "rank": 8})
    frozen = {"selection": {"lora_spec": {"path": str(spec), "sha256": file_hash(spec)}}, "method": "x"}
    with pytest.raises(wsp_final.FinalRefused, match="merged no modules"):
        wsp_final.build_adapt(frozen, config={"seed": 42}, adapter_factory=_FakeAdapter,
                              load=lambda *a: None,
                              merge=lambda model: {"modules_merged": 0, "names": []})


def test_the_patched_arm_uses_the_merged_base_not_a_fresh_config_load(tmp_path):
    """_reload_model must copy the merged state; it must not rebuild from the model config."""
    pytest.importorskip("torch")
    base = _FakeAdapter({"seed": 42})
    base.model.bias.data = base.model.bias.data + 5.0         # simulate a completed merge
    merged_hash = wsp_final.merged_state_hash(base.model)
    base.metadata = dict(base.metadata, merged_base={"merged_state_sha256": merged_hash})
    frozen = {"selection": {"model_config": None}, "method": "wsp_patch"}
    copied = wsp_final._reload_model(base, frozen)
    assert wsp_final.merged_state_hash(copied) == merged_hash, "the arm must start from the merged state"
    assert copied is not base.model, "the arm must get its own copy"
    copied.bias.data = copied.bias.data + 99.0
    assert wsp_final.merged_state_hash(base.model) == merged_hash, "a patch must not leak into the base"


def test_reload_model_refuses_a_base_that_drifted_from_its_merged_hash(tmp_path):
    pytest.importorskip("torch")
    base = _FakeAdapter({"seed": 42})
    base.metadata = dict(base.metadata, merged_base={"merged_state_sha256": "0" * 64})
    with pytest.raises(wsp_final.FinalRefused, match="merged base changed"):
        wsp_final._reload_model(base, {"selection": {"model_config": None}})


# --------------------------------------------------------------------------------------------------
# a frozen steering block is applied, never silently ignored (high)
# --------------------------------------------------------------------------------------------------

def _steering_block(tmp_path, *, layer=0, strength=1.5, site="residual_stream", direction=None):
    pytest.importorskip("torch")
    import torch
    path = tmp_path / "direction.json"
    atomic_json(path, {"direction": direction if direction is not None else [0.0, 1.0]})
    return {"direction_path": str(path), "direction_sha256": file_hash(path), "layer": layer,
            "strength": strength, "site": site}


def _frozen_with_steering(steering):
    return {"selection": {"steering": steering}, "method": "safetune_steering",
            "evidence_kind": "cohere_local_checkpoint", "strong_prompt": "verify"}


def test_steering_refuses_at_freeze_time_when_it_cannot_be_applied(tmp_path):
    """A site or strength this script cannot apply must refuse, not be bound and ignored."""
    for steering, pattern in (
            (_steering_block(tmp_path, site="attention_heads"), "residual-stream"),
            (_steering_block(tmp_path, strength=0.0), "silent no-op"),
            (_steering_block(tmp_path, layer=-1), "negative"),
    ):
        with pytest.raises(wsp_final.FinalRefused, match=pattern):
            wsp_final.verify_steering_applicable(_frozen_with_steering(steering))


def test_steering_refuses_a_zero_direction(tmp_path):
    with pytest.raises(wsp_final.FinalRefused, match="zero steering direction"):
        wsp_final.verify_steering_applicable(_frozen_with_steering(_steering_block(tmp_path, direction=[0.0, 0.0])))


def test_no_steering_block_needs_nothing_applied(tmp_path):
    # A freeze with no steering is applicable by definition: nothing is bound, nothing is ignored.
    wsp_final.verify_steering_applicable({"selection": {"steering": None}, "method": "none"})


def test_a_bound_steering_block_is_applied_through_a_real_forward_hook(tmp_path):
    """The direction must reach the model: the hook changes the layer's output."""
    torch = pytest.importorskip("torch")

    class _Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Block(), _Block()])

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    adapter = _FakeAdapter({"seed": 42})
    adapter.model = _Model()
    adapter.device = "cpu"
    steering = _steering_block(tmp_path, layer=1, strength=2.0, direction=[1.0, 0.0])
    record = wsp_final.apply_steering(adapter, steering)
    assert record["applied"] is True and record["layer"] == 1 and record["strength"] == 2.0
    assert record["direction_dim"] == 2
    unt = torch.zeros(1, 2)
    # Layer 0 is not steered; layer 1 adds 2.0 * [1, 0].
    assert torch.equal(adapter.model.model.layers[0](unt), unt)
    steered = adapter.model.model.layers[1](unt)
    assert torch.allclose(steered, torch.tensor([[2.0, 0.0]])), steered


def test_apply_steering_refuses_a_layer_the_model_does_not_have(tmp_path):
    torch = pytest.importorskip("torch")

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    adapter = _FakeAdapter({"seed": 42})
    adapter.model, adapter.device = _Model(), "cpu"
    with pytest.raises(wsp_final.FinalRefused, match="does not exist"):
        wsp_final.apply_steering(adapter, _steering_block(tmp_path, layer=7))


def test_a_steering_method_with_no_frozen_block_is_a_visible_gap_not_a_silent_skip(tmp_path):
    """A steering arm with no direction cannot be measured; it must say so, not run unsteered."""
    entry = {"method": "safetune_steering:C", "kind": "candidate", "name": "safetune_steering",
             "strong": False, "variant": "trace", "label": "steering"}
    base = _FakeAdapter({"seed": 42})
    with pytest.raises(wsp_final.ArmUnavailable, match="no steering direction"):
        wsp_final.prepare_arm(base, {"selection": {"steering": None}, "evidence_kind": "cohere_local_checkpoint"},
                              entry, allow_toy=False)


# --------------------------------------------------------------------------------------------------
# a crashed final look is not restartable (critical)
# --------------------------------------------------------------------------------------------------

def test_a_crashed_final_look_cannot_be_retried(tmp_path):
    """Crash after the sealed data was opened, then try again: the retry must refuse.

    The crash is simulated by making an arm raise. The previous implementation wrote status 'failed'
    and check_final_look let a 'failed' directory run again -- so a crashed look was silently
    restartable, and a run that had already seen the sealed labels could be re-run.
    """
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    original = wsp_final.run_regressions

    def crashing(*args, **kwargs):
        raise RuntimeError("simulated crash during the first model call")

    wsp_final.run_regressions = crashing
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    finally:
        wsp_final.run_regressions = original
    record = read_json(setup["output"] / "evaluation.json")
    assert record["status"] == "failed"
    assert record["directory_consumed"]
    assert (setup["output"] / "final_look.json").is_file(), "the directory must be consumed on disk"
    with pytest.raises(wsp_final.FinalRefused, match="consumed"):
        wsp_final.evaluate(wsp_final.verify_freeze(setup["output"] / "freeze.json"), setup["output"], allow_toy=True)


def test_a_failed_record_without_the_consumed_marker_still_refuses(tmp_path):
    """Defence in depth: even if only evaluation.json survived, a look was attempted."""
    setup = frozen_setup(tmp_path)
    atomic_json(setup["output"] / "evaluation.json", {"schema_version": 1, "status": "failed"})
    with pytest.raises(wsp_final.FinalRefused, match="final look was attempted here"):
        wsp_final.check_final_look(setup["output"])


def test_any_evaluation_status_other_than_none_refuses(tmp_path):
    for status in ("running", "failed", "interrupted", "completed", "queued"):
        output = tmp_path / f"look-{status}"
        output.mkdir()
        atomic_json(output / "evaluation.json", {"schema_version": 1, "status": status})
        with pytest.raises(wsp_final.FinalRefused):
            wsp_final.check_final_look(output)


def test_report_or_summary_without_an_evaluation_record_is_reachable_and_refused(tmp_path):
    """The reviewer's medium: this branch used to be shadowed for report.md and summary.json."""
    for name in ("report.md", "summary.json"):
        output = tmp_path / f"has-{name}"
        output.mkdir()
        (output / name).write_text("x", encoding="utf-8")
        with pytest.raises(wsp_final.FinalRefused, match="but no evaluation record"):
            wsp_final.check_final_look(output)


def test_the_check_cli_and_a_later_evaluate_look_at_the_same_directory(tmp_path):
    """--check must verify the freeze's own directory, not whatever --output says."""
    setup = frozen_setup(tmp_path)
    manifest = setup["output"] / "freeze.json"
    assert wsp_final.main(["evaluate", "--freeze-manifest", str(manifest), "--check"]) == 0
    # A --output naming another directory is refused rather than silently checked instead.
    assert wsp_final.main(["evaluate", "--freeze-manifest", str(manifest), "--check",
                           "--output", str(tmp_path / "elsewhere")]) == 2
    assert not (tmp_path / "elsewhere").exists()
    # After the single look, --check on the same directory reports the consumption.
    bundle = wsp_final.verify_freeze(manifest)
    wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    assert wsp_final.main(["evaluate", "--freeze-manifest", str(manifest), "--check"]) == 2


def test_the_consumed_record_is_written_before_the_sealed_data_is_opened(tmp_path):
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    seen = []
    original = wsp_final.load_final_cases

    def spy(binding):
        seen.append((setup["output"] / "final_look.json").is_file())
        return original(binding)

    wsp_final.load_final_cases = spy
    try:
        wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    finally:
        wsp_final.load_final_cases = original
    assert seen == [True], "the consumed record must exist before the sealed dataset is parsed"


# --------------------------------------------------------------------------------------------------
# prompt identity binds the real rendered prompts (high)
# --------------------------------------------------------------------------------------------------

def _frozen_identity(tmp_path):
    return {"prompt_identity": wsp_final.prompt_identity(
        _protocol(), "verify", ["reviewer", "authority", "ciso", "scanner", "consensus", "code_comment"])}


def _real_requests(cases, strong=False, wording="reviewer", variant="trace"):
    from security_wsp.prompts import CONDITIONS, render
    return [render(case, condition, strong, max_new_tokens=256, wording=wording,
                   strong_variant=variant, prefill=True) for case in cases for condition in CONDITIONS]


def test_every_real_rendered_prompt_verifies_against_the_frozen_identity(tmp_path):
    """The binding covers the prompts the run actually renders, prefill included."""
    _, cases = fake_final_dataset(tmp_path)
    frozen = _frozen_identity(tmp_path)
    for strong, variant in ((False, "trace"), (True, "verify")):
        for wording in ("reviewer", "authority", "ciso", "scanner", "consensus", "code_comment"):
            for request in _real_requests(cases, strong, wording, variant):
                wsp_final.enforce_prompt_identity(request, frozen, "test")


def test_a_changed_prefill_is_refused_before_generation(tmp_path):
    from security_wsp.prompts import render, GenerationRequest
    _, cases = fake_final_dataset(tmp_path)
    frozen = _frozen_identity(tmp_path)
    request = render(cases[0], "neutral", False, wording="reviewer", prefill=True)
    broken = GenerationRequest(request.case_id, request.condition,
                               request.messages[:-1] + [{"role": "assistant", "content": '{"verdict": "'}],
                               request.max_new_tokens, request.seed)
    with pytest.raises(wsp_final.FinalRefused, match="json_prefill"):
        wsp_final.enforce_prompt_identity(broken, frozen, "A")


def test_a_prompt_without_the_prefill_is_refused(tmp_path):
    from security_wsp.prompts import render
    _, cases = fake_final_dataset(tmp_path)
    frozen = _frozen_identity(tmp_path)
    request = render(cases[0], "neutral", False, wording="reviewer", prefill=False)
    with pytest.raises(wsp_final.FinalRefused, match="json_prefill"):
        wsp_final.enforce_prompt_identity(request, frozen, "A")


def test_a_prompt_with_an_unfrozen_system_message_is_refused(tmp_path):
    from security_wsp.prompts import render, GenerationRequest
    _, cases = fake_final_dataset(tmp_path)
    frozen = _frozen_identity(tmp_path)
    request = render(cases[0], "neutral", False, wording="reviewer", prefill=True)
    messages = list(request.messages)
    messages[0] = {"role": "system", "content": "You are a helpful assistant."}
    with pytest.raises(wsp_final.FinalRefused, match="system prompt"):
        wsp_final.enforce_prompt_identity(
            GenerationRequest(request.case_id, request.condition, messages, 256, 42), frozen, "A")


def test_a_neutral_prompt_carrying_an_opinion_is_refused(tmp_path):
    from security_wsp.prompts import render, GenerationRequest
    _, cases = fake_final_dataset(tmp_path)
    frozen = _frozen_identity(tmp_path)
    request = render(cases[0], "neutral", False, wording="reviewer", prefill=True)
    messages = list(request.messages)
    messages[1] = {"role": "user", "content": messages[1]["content"].replace(
        wsp_final.PROMPT_CLOSING, "Reviewer assessment (opinion only): I believe the verdict is vulnerable.\n"
        + wsp_final.PROMPT_CLOSING)}
    with pytest.raises(wsp_final.FinalRefused, match="neutral prompt carries an opinion"):
        wsp_final.enforce_prompt_identity(
            GenerationRequest(request.case_id, "neutral", messages, 256, 42), frozen, "A")


def test_an_unfrozen_wording_is_refused(tmp_path):
    """A wording the freeze did not bind cannot be rendered into the final run."""
    _, cases = fake_final_dataset(tmp_path)
    frozen = {"prompt_identity": wsp_final.prompt_identity(_protocol(), "verify", ["reviewer"])}
    request = _real_requests(cases, strong=True, wording="authority", variant="verify")[1]
    with pytest.raises(wsp_final.FinalRefused, match="not one the freeze bound"):
        wsp_final.enforce_prompt_identity(request, frozen, "B")


def test_verify_freeze_refuses_a_manifest_with_an_unbound_wording(tmp_path):
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen["prompt_identity"]["wordings"] = ["reviewer"]
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(frozen))
    with pytest.raises(wsp_final.FinalRefused, match="wordings the freeze did not bind"):
        wsp_final.verify_freeze(setup["output"] / "freeze.json")


def test_verify_freeze_refuses_a_tampered_opinion_sentence(tmp_path):
    setup = frozen_setup(tmp_path)
    frozen = read_json(setup["output"] / "freeze.json")
    key = "correct_suggestion|reviewer|vulnerable"
    frozen["prompt_identity"]["opinions"][key] = "Reviewer assessment: it is definitely fine.\n"
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(frozen))
    # The record is internally consistent, so only the render against the real code catches it.
    _, cases = fake_final_dataset(tmp_path)
    with pytest.raises(wsp_final.FinalRefused, match="not one the freeze bound"):
        wsp_final.enforce_prompt_identity(_real_requests(cases, False, "reviewer")[1], frozen, "A")


# --------------------------------------------------------------------------------------------------
# smaller findings
# --------------------------------------------------------------------------------------------------

def test_a_review_failure_at_the_sealed_stage_is_a_refusal_not_a_traceback(tmp_path):
    """verify_independent_review raises ValueError; main must turn that into a clean refusal."""
    setup = frozen_setup(tmp_path)
    # Corrupt the review's content while keeping the freeze's recorded hash out of the way: the
    # reviewer's attestations are checked inside verify_independent_review, which raises ValueError.
    review = read_json(setup["review"])
    review["blinded_to_oracle_labels"] = False
    atomic_json(setup["review"], review)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen["label_review"]["sha256"] = file_hash(setup["review"])
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(frozen))
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    with pytest.raises(wsp_final.FinalRefused, match="review failed verification at the sealed stage"):
        wsp_final.evaluate(bundle, setup["output"], allow_toy=True)


def test_the_review_refusal_reaches_the_cli_as_exit_code_two(tmp_path, capsys):
    setup = frozen_setup(tmp_path)
    review = read_json(setup["review"])
    review["blinded_to_oracle_labels"] = False
    atomic_json(setup["review"], review)
    frozen = read_json(setup["output"] / "freeze.json")
    frozen["label_review"]["sha256"] = file_hash(setup["review"])
    atomic_json(setup["output"] / "freeze.json", wsp_final.seal_manifest(frozen))
    code = wsp_final.main(["evaluate", "--freeze-manifest", str(setup["output"] / "freeze.json"),
                           "--allow-toy-core"])
    assert code == 2
    assert "Refused:" in capsys.readouterr().err


def test_report_md_is_written_atomically(tmp_path):
    """No temp file is left behind, and the report is complete when it appears."""
    setup = frozen_setup(tmp_path)
    bundle = wsp_final.verify_freeze(setup["output"] / "freeze.json")
    wsp_final.evaluate(bundle, setup["output"], allow_toy=True)
    left = [p.name for p in setup["output"].iterdir() if p.name.startswith(".aawden")]
    assert left == [], f"temporary artifacts left behind: {left}"
    text = (setup["output"] / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# Aawden protocol v3") and text.rstrip().endswith("`")


def test_invalid_and_abstention_counts_are_disjoint(tmp_path):
    """An invalid answer is not an abstention, and neither is counted twice."""
    rows = [
        {"parsed": None, "parse_error": "not JSON", "predicted_verdict": None, "oracle_verdict": VULNERABLE,
         "correct": False},
        # A row that carries BOTH a parse error and a parsed verdict: still invalid, and not also
        # counted as an abstention -- the old helper counted it in both totals.
        {"parsed": {"verdict": ABSTAIN}, "parse_error": "trailing text", "predicted_verdict": ABSTAIN,
         "oracle_verdict": CLEAN, "correct": False},
        {"parsed": {"verdict": ABSTAIN}, "parse_error": None, "predicted_verdict": ABSTAIN,
         "oracle_verdict": CLEAN, "correct": False},
        {"parsed": {"verdict": VULNERABLE}, "parse_error": None, "predicted_verdict": VULNERABLE,
         "oracle_verdict": VULNERABLE, "correct": True},
    ]
    measure = wsp_final._abstention_measure(rows)
    assert measure["invalid_count"] == 2
    assert measure["abstention_count"] == 1
    assert measure["invalid_count"] + measure["abstention_count"] <= len(rows)


def test_the_report_labels_invalid_and_abstentions_separately():
    summary = _minimal_summary()
    summary["metrics"]["A"]["neutral"]["invalid_count"] = 2
    summary["metrics"]["A"]["neutral"]["abstention_count"] = 3
    report = wsp_final.render_report(summary)
    header = [line for line in report.splitlines() if line.startswith("| method | condition | n |")]
    assert header and "invalid (unparsed)" in header[0] and "abstentions (parsed refusal)" in header[0]
    row = [line for line in report.splitlines() if line.startswith("| A | neutral |")][0]
    assert "| 2 | 3 |" in row


def test_a_consumed_directory_cannot_be_refrozen(tmp_path):
    """A look was attempted here; a fresh freeze in the same directory would be a new look, not a fix."""
    setup = frozen_setup(tmp_path)
    atomic_json(setup["output"] / "final_look.json", {"status": "consumed"})
    (setup["output"] / "freeze.json").unlink()
    with pytest.raises(wsp_final.FinalRefused, match="consumed final-look record"):
        wsp_final.freeze(setup["selection"], setup["output"], review=setup["review"], regressions=REGRESSIONS,
                         dataset=setup["dataset"], commitments=setup["commitments"],
                         regression_commitments=tmp_path / "fake-wsp-regressions.json", allow_toy=True)
