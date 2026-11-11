"""Offline tests for scripts/wsp_localize.py: toy end-to-end, refusals, determinism, statistics.

Nothing here touches a checkpoint, a GPU or the network. The end-to-end runs use the script's own tiny
random Cohere model on CPU (about two seconds each); the unit tests use a tiny transformer config.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wsp_localize.py"

SPEC = importlib.util.spec_from_file_location("wsp_localize", SCRIPT)
localize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(localize)

from security_wsp.data import CLEAN, VULNERABLE, Case  # noqa: E402

MINIMAL = ["--toy", "--limit", "4", "--layers", "0", "--top-k", "1", "--attention-cases", "2",
           "--attention-call-cap", "4", "--causal-cases", "3", "--bootstrap-replicates", "100"]


def run(*args, timeout=600):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


def toy_run(tmp_path, name="localize", *extra):
    output = tmp_path / name
    result = run(*MINIMAL, "--output", str(output), *extra)
    assert result.returncode == 0, result.stderr[-4000:]
    return output


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------------------------
def test_toy_run_produces_every_artifact_and_a_verifiable_manifest(tmp_path):
    out = toy_run(tmp_path)
    for name in ("localization.json", "components.json", "attention.json", "report.md", "manifest.json"):
        assert (out / name).is_file(), name
    manifest = read(out / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["evidence_kind"] == "constructed_toy"
    assert manifest["failure_reason"] is None
    # Re-verifying with the library entry point checks the identity digest and every artifact hash.
    from security_wsp.artifacts import verify_run
    verified = verify_run(out)
    assert verified["valid"] is True
    assert verified["verified_artifacts"] == len(manifest["artifact_hashes"])
    assert (out / "circuitkit.json").is_file()


def test_artifacts_carry_the_required_sections(tmp_path):
    out = toy_run(tmp_path)
    components = read(out / "components.json")
    assert components["sites"] == ["resid_post", "mlp_out", "attn_out"]
    assert components["claim"] == localize.CLAIM_POLICY
    assert components["top_k"] and all(row["component"] in [r["component"] for r in components["ranked"]]
                                       for row in components["ranked"][:1])
    for row in components["ranked"]:
        assert {"component", "site", "layer", "separation_statistic", "separation_p_value",
                "random_baseline_mean", "score", "cosine_to_mean_direction"} <= set(row)
    attention = read(out / "attention.json")
    assert attention["call_cap"] == 4 and attention["calls_used"] <= attention["call_cap"]
    assert "match_failures" in attention and "reasons" in attention["match_failures"]
    assert attention["heads_ranked"], "the toy must produce per-head attention rows"
    localization = read(out / "localization.json")
    assert localization["claim_policy"]["exploratory"] == localize.CLAIM_POLICY
    assert localization["condition"] == "incorrect_suggestion"
    assert localization["cost"]["tool_units_used"] <= localization["cost"]["budget"]
    assert localization["circuitkit_status"] in ("absent", "disabled_by_flag", "unsupported_api",
                                                 "available_but_not_run", "import_error")
    for component in localization["causal"]["components"]:
        assert component["no_op_control"]["passed"] is True
        assert component["no_op_control"]["max_absolute_margin_delta"] <= localize.NOOP_TOLERANCE
        assert set(component["controls"]) == {"mean_direction", "opposite_direction",
                                             "random-17", "random-29", "random-43"}
        # The projection is sign-symmetric, so the opposite direction must be the same edit.
        assert component["projection_idempotence_check"]["matches_mean_direction_within_1e-5"] is True
        assert component["label_mapping_swap"]["status"] in ("consistent", "disagree")
        assert component["label_mapping_swap"]["raw_shift_by_continuation"]
        assert set(component["per_verdict_effect"]) == {"vulnerable", "not_vulnerable_for_target"}
        assert component["conclusion"] in (localize.CLAIM_POLICY, localize.SUPPORTED)
    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## 1. Activation contrast ranking", "## 2. Attention",
                    "## 3. Causal projection check", "## 4. CircuitKIT cross-check", "## Claim policy"):
        assert heading in report
    assert localize.CLAIM_POLICY in report


def test_causal_stage_meters_every_forward_it_spends(tiny_setup):
    """The causal stage does not go through WspTools, so its cost is only visible via UnitMeter.

    The exact count is forced by the construction: for every case, instrument and continuation
    continuation_scores makes exactly one forward. Baseline is cases * instruments * continuations; each of
    the 5 arms adds the same again. The no-op arm and the opposite-direction arm are real forwards and are
    charged too.
    """
    model, tokenizer, cases, specs = tiny_setup
    cases = cases[:2]
    continuations = sum(len(spec["continuations"]) for spec in specs.values())
    per_table = len(cases) * continuations

    stores = []

    class _Store:
        manifest = {"run_id": "test"}

        def save_prediction(self, request_id, record):
            stores.append(request_id)

    component = {"component": "mlp_out-L00", "site": "mlp_out", "layer": 0, "rank": 1, "score": 0.0}
    directions = {"mlp_out-L00": {"direction": torch_direction(model.config.hidden_size).tolist()}}
    seeds = (17, 29, 43)
    stage = localize.causal_stage(model, tokenizer, cases, [component], directions, "incorrect_suggestion",
                                  specs, 4096, 0.0, seeds, 100, _Store())
    expected = per_table * (1 + 3 + len(seeds))  # baseline + no-op + mean + opposite + one per random seed
    assert stage["cost"]["units"] == expected, (stage["cost"]["units"], expected)
    assert stage["cost"]["baseline_units"] == per_table
    assert stage["cost"]["components"] == 1
    assert "UnitMeter" in stage["cost"]["metered_by"]
    # Every arm of the one component is charged after the baseline.
    assert stage["cost"]["units"] - stage["cost"]["baseline_units"] == per_table * (3 + len(seeds))
    assert stores and "margin-baseline" in stores


def test_cost_block_sums_all_three_stages():
    """The total is activation + attention + causal, and only two of them are charged to the tool budget."""
    cost = localize.cost_block(2048, 100, activation_units=40,
                               attention={"tool_units_used": 7, "tool_units_planned": 8},
                               causal={"cost": {"units": 900}})
    assert cost["stages"]["activation"]["units"] == 40
    assert cost["stages"]["attention"]["units"] == 7
    assert cost["stages"]["causal"]["units"] == 900
    assert cost["total_units_used"] == 947
    assert cost["tool_units_used"] == 47, "the tool budget covers activation and attention only"
    assert cost["tool_units_used"] <= cost["budget"]
    assert cost["total_units_used"] != cost["tool_units_used"], \
        "reporting the tool total as the total is the bug: the causal stage would vanish"
    assert localize.total_units(cost["stages"]["activation"], cost["stages"]["attention"],
                                cost["stages"]["causal"]) == 947
    # A missing stage contributes nothing rather than raising or inventing a number.
    empty = localize.cost_block(10, 1, activation_units=0, attention={}, causal={})
    assert empty["total_units_used"] == 0 and empty["tool_units_used"] == 0


def test_cost_block_counts_every_stage_not_just_activation(tmp_path):
    """The cost block must add activation + attention + causal units, and the report must say so.

    Before the fix localization.json reported only the activation units under the name tool_units_used,
    attention units were computed on fresh WspTools instances and discarded, and the causal stage -- which
    scores continuations directly -- was never counted at all, although the docs claimed it was.
    """
    out = toy_run(tmp_path)
    localization = read(out / "localization.json")
    cost = localization["cost"]
    stages = cost["stages"]
    assert set(stages) == {"activation", "attention", "causal"}
    # The causal stage really did run, so a zero there would mean it was left uncounted.
    assert stages["causal"]["units"] > 0, "the causal stage must be metered"
    assert cost["total_units_used"] == sum(stage["units"] for stage in stages.values())
    assert cost["total_units_used"] > stages["activation"]["units"], \
        "the total must exceed the activation count, or attention/causal are missing from it"
    # Only activation and attention are charged to the tool budget, and that total is the budget check.
    assert cost["tool_units_used"] == stages["activation"]["units"] + stages["attention"]["units"]
    assert cost["tool_units_used"] <= cost["budget"]
    assert stages["causal"]["charged_to_tool_budget"] is False
    assert stages["activation"]["charged_to_tool_budget"] is True
    # The causal stage reports its own count too, so both artifacts agree.
    assert stages["causal"]["units"] == localization["causal"]["cost"]["units"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## 5. Cost" in report
    assert str(cost["total_units_used"]) in report
    assert "causal" in report and "attention" in report


def test_report_is_written_atomically(tmp_path):
    """report.md goes through the atomic helper, so a crash cannot leave a truncated file to be hashed."""
    from security_wsp.artifacts import file_hash, verify_run
    out = toy_run(tmp_path)
    # The helper itself: replaces atomically and leaves no temp file behind.
    target = tmp_path / "atomic" / "report.md"
    localize.atomic_write_text(target, "first\n")
    localize.atomic_write_text(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    assert [p.name for p in target.parent.iterdir()] == ["report.md"], "no temp file may survive a write"
    # The script never uses a bare write_text for a hashed artifact.
    source = SCRIPT.read_text(encoding="utf-8")
    assert '.write_text(' not in source, "report.md must not be written with a bare write_text"
    assert 'atomic_write_text(args.output / "report.md"' in source
    assert "report.md" in read(out / "manifest.json")["artifact_hashes"]
    assert verify_run(out)["valid"] is True
    assert len(file_hash(out / "report.md")) == 64


def test_predictions_behind_every_component_are_hashed(tmp_path):
    out = toy_run(tmp_path)
    manifest = read(out / "manifest.json")
    names = set(manifest["artifact_hashes"])
    components = read(out / "components.json")
    for row in components["ranked"]:
        assert f"predictions/contrast-{row['component']}.json" in names
    localization = read(out / "localization.json")
    for component in localization["causal"]["components"]:
        assert f"predictions/causal-{component['component']}.json" in names
    assert "predictions/margin-baseline.json" in names
    # A tampered artifact must be caught by verify_run.
    from security_wsp.artifacts import verify_run
    path = out / "components.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_run(out)


def test_toy_ranking_is_deterministic(tmp_path):
    first, second = toy_run(tmp_path, "a"), toy_run(tmp_path, "b")
    for name in ("components.json", "attention.json", "localization.json", "report.md", "circuitkit.json"):
        assert (first / name).read_text(encoding="utf-8") == (second / name).read_text(encoding="utf-8"), name
    ranked = [row["component"] for row in read(first / "components.json")["ranked"]]
    assert ranked == [row["component"] for row in read(second / "components.json")["ranked"]]


def test_resume_rejects_a_changed_identity(tmp_path):
    out = toy_run(tmp_path)
    before = (out / "components.json").read_text(encoding="utf-8")
    # Identical configuration resumes and reproduces the artifacts.
    assert run(*MINIMAL, "--output", str(out)).returncode == 0
    assert (out / "components.json").read_text(encoding="utf-8") == before
    changed = run(*MINIMAL, "--top-k", "2", "--output", str(out))
    assert changed.returncode != 0
    assert "Resume rejected" in (changed.stderr + changed.stdout)


def test_resume_rejects_a_changed_budget(tmp_path):
    """--budget changes the cost block and the report, so it is part of the run identity.

    Before the fix the identity omitted --budget, so a second run with a different budget was accepted and
    silently rewrote localization.json and report.md under the original manifest.
    """
    out = toy_run(tmp_path)
    before = read(out / "localization.json")["cost"]["budget"]
    result = run(*MINIMAL, "--budget", "4096", "--output", str(out))
    assert result.returncode != 0, "a resume with a changed --budget must be rejected"
    assert "Resume rejected" in (result.stderr + result.stdout)
    # The rejected run must not have rewritten the evidence.
    assert read(out / "localization.json")["cost"]["budget"] == before


def test_resume_rejects_a_changed_circuitkit_flag(tmp_path):
    """--no-circuitkit changes circuitkit.json and localization.json, so it is part of the identity too."""
    out = toy_run(tmp_path)
    status = read(out / "circuitkit.json")["status"]
    result = run(*MINIMAL, "--no-circuitkit", "--output", str(out))
    assert result.returncode != 0, "a resume with --no-circuitkit flipped must be rejected"
    assert "Resume rejected" in (result.stderr + result.stdout)
    assert read(out / "circuitkit.json")["status"] == status


def test_validation_and_final_datasets_are_refused(tmp_path):
    for dataset, fragment in (("datasets/wsp-validation.json", "Refused"),
                              ("datasets/wsp-final.json", "Refused")):
        result = run("--toy", "--dataset", dataset, "--output", str(tmp_path / Path(dataset).stem))
        assert result.returncode != 0, dataset
        assert fragment in (result.stderr + result.stdout)
        assert not (tmp_path / Path(dataset).stem / "manifest.json").exists()


def test_sealed_and_uncommitted_datasets_are_refused_by_the_loader(tmp_path):
    protocol = read(ROOT / "configs" / "protocol-v3.json")
    with pytest.raises(SystemExit):
        localize.load_committed_cases(ROOT / "datasets" / "wsp-validation.json", protocol, None)
    with pytest.raises(SystemExit):
        localize.load_committed_cases(ROOT / "datasets" / "wsp-final.json", protocol, None)
    with pytest.raises(SystemExit):
        localize.load_committed_cases(ROOT / "configs" / "wsp-data.json", protocol, None)
    # A copied dataset has the right bytes but the wrong path commitment.
    copy = tmp_path / "wsp-discovery.json"
    copy.write_bytes((ROOT / "datasets" / "wsp-discovery.json").read_bytes())
    with pytest.raises(SystemExit):
        localize.load_committed_cases(copy, protocol, None)


def test_transfer_wordings_are_refused(tmp_path):
    protocol = read(ROOT / "configs" / "protocol-v3.json")
    for wording in protocol["transfer_wordings_final_only"]:
        result = run("--toy", "--wording", wording, "--output", str(tmp_path / wording))
        assert result.returncode != 0, wording
        assert "reserved for final evaluation" in (result.stderr + result.stdout)
        assert not (tmp_path / wording / "manifest.json").exists()


def test_non_primary_condition_is_refused(tmp_path):
    result = run("--toy", "--condition", "neutral", "--output", str(tmp_path / "n"))
    assert result.returncode != 0
    assert "primary condition" in (result.stderr + result.stdout)


def test_budget_plan_refuses_instead_of_truncating(tmp_path):
    result = run("--toy", "--budget", "3", "--output", str(tmp_path / "b"))
    assert result.returncode != 0
    assert "above --budget" in (result.stderr + result.stdout)


# ---------------------------------------------------------------------------------------------
# unit tests on a tiny transformer
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_setup():
    """A tiny random Cohere model and a word-level tokenizer that covers the real discovery prompts."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from security_wsp.data import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM
    cases = load_dataset(ROOT / "datasets" / "wsp-discovery.json")[:2]
    specs = localize.instrument_specs()
    tokenizer = _toy_tokenizer(cases, specs)
    torch.manual_seed(0)
    config = AutoConfig.for_model("cohere", vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                  max_position_embeddings=4096, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.unk_token_id)
    model = AutoModelForCausalLM.from_config(config).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.5 * torch.randn(param.shape, generator=generator))
            elif name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight")):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
    return model, tokenizer, cases, specs


def test_paired_statistic_flags_a_consistent_shift_and_not_noise():
    rng = __import__("random").Random(11)
    consistent = [0.4 + rng.gauss(0, 0.05) for _ in range(20)]
    noise = [rng.gauss(0, 1.0) for _ in range(20)]
    good = localize.sign_flip_p(consistent, 2000, 5)
    bad = localize.sign_flip_p(noise, 2000, 5)
    assert good["statistic"] > 0.35 and good["p_value"] < 0.01
    assert bad["p_value"] > 0.01
    assert 0 < good["p_value"] <= 1 and 0 < bad["p_value"] <= 1
    assert good["permutations"] == 2000 and good["seed"] == 5
    # Same inputs, same seed, same answer.
    assert localize.sign_flip_p(consistent, 2000, 5) == good


def test_family_bootstrap_needs_families_and_reports_a_single_one_honestly():
    clustered = {"f1": [1.0, 1.0], "f2": [0.0, 0.0], "f3": [0.5, 0.5]}
    report = localize.family_bootstrap_mean(clustered, 500, 3)
    assert report["families"] == 3 and report["cases"] == 6
    assert report["difference"] == pytest.approx(0.5)
    assert report["ci95"][0] <= report["difference"] <= report["ci95"][1]
    single = localize.family_bootstrap_mean({"only": [0.2, 0.4]}, 100, 3)
    assert "not informative" in single["interval"]
    assert localize.family_bootstrap_mean({}, 100, 3)["difference"] is None


def test_balanced_accuracy_interval_runs_on_forced_choice_rows():
    pytest.importorskip("torch")
    statistics = pytest.importorskip("security_wsp.statistics")
    rows = []
    # The no-op arm is right on every vulnerable case and wrong on every clean case (BA 0.5); the edited
    # arm is right everywhere (BA 1.0), so the paired difference is 0.5 and must be positive.
    for index, family in enumerate(("f1", "f2", "f3", "f4")):
        for condition in ("vulnerable", "not_vulnerable_for_target"):
            case = f"c{index}{condition[:3]}"
            rows.append({"method": "noop", "condition": "incorrect_suggestion:json_prefill", "case_id": case,
                         "family_id": family, "oracle_verdict": condition, "predicted_verdict": "vulnerable",
                         "correct": condition == "vulnerable", "parse_error": None, "generation_error": None})
            rows.append({"method": "mean_direction", "condition": "incorrect_suggestion:json_prefill",
                         "case_id": case, "family_id": family, "oracle_verdict": condition,
                         "predicted_verdict": condition, "correct": True, "parse_error": None,
                         "generation_error": None})
    interval = statistics.paired_cluster_interval(rows, left="mean_direction", right="noop",
                                                  condition="incorrect_suggestion:json_prefill", replicates=200)
    assert interval["difference"] == pytest.approx(0.5)
    assert interval["ci95"][0] <= interval["difference"] <= interval["ci95"][1]


def test_margin_table_scores_the_prefill_identically_for_both_label_mappings(tiny_setup):
    model, tokenizer, cases, specs = tiny_setup
    table = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion", specs, 4096)
    assert set(table) == set(specs)
    for name, block in table.items():
        assert len(block["cases"]) == len(cases)
        for entry in block["cases"].values():
            assert entry["degenerate_continuations"] is False
            assert entry["correct_margin"] is not None
            assert entry["prefix_tokens"] > 0
    # The two letter instruments state A/B the other way round in the prompt text, so the margins move in
    # opposite directions; the prompt text differs, so the magnitudes only approximately cancel.
    a, b = table["letters_a_vulnerable"]["cases"], table["letters_b_vulnerable"]["cases"]
    for case_id in a:
        assert a[case_id]["correct_margin"] * b[case_id]["correct_margin"] < 0
        assert a[case_id]["correct_margin"] == pytest.approx(-b[case_id]["correct_margin"], rel=0.1, abs=1e-4)

    # A true label-mapping swap on one instrument negates the margin exactly: the prefill is unchanged and
    # only the verdict attached to each continuation is exchanged.
    swapped = localize.instrument_specs()
    mapping = swapped["json_prefill"]["verdict_by_continuation"]
    swapped["json_prefill"]["verdict_by_continuation"] = {c: (CLEAN if v == VULNERABLE else VULNERABLE)
                                                          for c, v in mapping.items()}
    flipped = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion",
                                    {"json_prefill": swapped["json_prefill"]}, 4096)
    for case_id, entry in table["json_prefill"]["cases"].items():
        assert flipped["json_prefill"]["cases"][case_id]["correct_margin"] == pytest.approx(
            -entry["correct_margin"], abs=1e-9)
        assert flipped["json_prefill"]["cases"][case_id]["prefix_sha256"] == entry["prefix_sha256"]
    rows = localize.forced_choice_rows(table, cases, "noop", "incorrect_suggestion")
    assert rows and all(row["family_id"] and row["method"] == "noop" for row in rows)


def test_mapping_swap_control_compares_raw_shifts_not_verdict_margins(tiny_setup):
    model, tokenizer, cases, specs = tiny_setup
    base = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion", specs, 4096)
    clean = localize.mapping_swap_consistency(base, base, cases)
    assert clean["status"] == "consistent"
    assert clean["max_absolute_difference"] == pytest.approx(0.0)
    # Swapping which verdict each letter states swaps the margins but not the raw letter shifts, so the
    # verdict-referenced margins differ in sign while the swap control still passes.
    swapped = localize.instrument_specs()
    swapped["letters_b_vulnerable"]["verdict_by_continuation"] = {" A": VULNERABLE, " B": CLEAN}
    table = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion", swapped, 4096)
    a, b = base["letters_a_vulnerable"]["cases"], table["letters_b_vulnerable"]["cases"]
    # A genuine label-mapping swap is the same prompt text with only the verdict attached to each letter
    # exchanged, which negates the verdict-referenced margin exactly. The two shipped letter instruments
    # differ in their sentence wording, so they only approximately cancel.
    flipped_specs = localize.instrument_specs()
    flipped_specs["letters_a_vulnerable"]["verdict_by_continuation"] = {" A": CLEAN, " B": VULNERABLE}
    flipped = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion",
                                    {"letters_a_vulnerable": flipped_specs["letters_a_vulnerable"]}, 4096)
    for case_id, entry in a.items():
        assert flipped["letters_a_vulnerable"]["cases"][case_id]["correct_margin"] == pytest.approx(
            -entry["correct_margin"], abs=1e-9)
        assert flipped["letters_a_vulnerable"]["cases"][case_id]["prefix_sha256"] == entry["prefix_sha256"]
    for case_id in a:
        # The shipped letter margins are tiny on this random model, so only their near-equality is stable.
        assert a[case_id]["correct_margin"] == pytest.approx(b[case_id]["correct_margin"], rel=0.1, abs=1e-4)
    # The prefill instrument has large margins, so its exact negation is a stable property.
    swapped_prefill = localize.instrument_specs()
    swapped_prefill["json_prefill"]["verdict_by_continuation"] = {VULNERABLE: CLEAN, CLEAN: VULNERABLE}
    other = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion",
                                  {"json_prefill": swapped_prefill["json_prefill"]}, 4096)
    for case_id, entry in base["json_prefill"]["cases"].items():
        assert other["json_prefill"]["cases"][case_id]["correct_margin"] == pytest.approx(
            -entry["correct_margin"], abs=1e-9)
    control = localize.mapping_swap_consistency(base, table, cases)
    assert control["status"] == "consistent"
    assert control["max_absolute_difference"] <= localize.MAPPING_TOLERANCE
    # One continuation moved by a large amount must be reported as a disagreement.
    tampered = {name: {"cases": {cid: {"scores": dict(entry["scores"])} for cid, entry in block["cases"].items()}}
                for name, block in base.items()}
    for cid in tampered["letters_b_vulnerable"]["cases"]:
        tampered["letters_b_vulnerable"]["cases"][cid]["scores"][" A"] += 1.0
    broken = localize.mapping_swap_consistency(base, tampered, cases)
    assert broken["status"] == "disagree" and broken["max_absolute_difference"] > localize.MAPPING_TOLERANCE
    assert localize.mapping_swap_consistency(base, {"json_prefill": base["json_prefill"]}, cases)["status"] == "unavailable"

def test_per_verdict_split_exposes_an_edit_that_trades_one_class_for_the_other():
    from security_wsp.data import load_dataset
    cases = [case for case in load_dataset(ROOT / "datasets" / "wsp-discovery.json")
             if case.oracle_verdict == VULNERABLE][:2] + \
            [case for case in load_dataset(ROOT / "datasets" / "wsp-discovery.json")
             if case.oracle_verdict == CLEAN][:2]
    deltas = {case.case_id: (-0.5 if case.oracle_verdict == VULNERABLE else 0.6) for case in cases}
    split = localize.per_verdict_breakdown(deltas, cases)
    assert split[VULNERABLE] == {"cases": 2, "mean_margin_delta": -0.5}
    assert split[CLEAN] == {"cases": 2, "mean_margin_delta": 0.6}
    # A mean of +0.05 hides opposite signs, which is exactly why the causal label needs both to be positive.
    assert sum(deltas.values()) / len(deltas) == pytest.approx(0.05)


def test_margin_of_the_noop_arm_equals_the_baseline_exactly(tiny_setup):
    model, tokenizer, cases, specs = tiny_setup
    base = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion", specs, 4096)
    direction = torch_direction(model.config.hidden_size)
    with localize.projected(model, ["model.layers.0.mlp.down_proj.weight"], direction, 0.0) as info:
        after = localize.margin_table(model, tokenizer, cases, "incorrect_suggestion", specs, 4096)
    assert info["restored"] is True
    for instrument in specs:
        for case_id in base[instrument]["cases"]:
            assert after[instrument]["cases"][case_id]["correct_margin"] == pytest.approx(
                base[instrument]["cases"][case_id]["correct_margin"], abs=1e-9)


def _toy_tokenizer(cases, specs):
    from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from security_wsp.prompts import render
    corpus = ["system", "user", "assistant", "<", ">", "A", "B", "Answer:"]
    for case in cases:
        for condition in ("neutral", "incorrect_suggestion"):
            corpus.extend(message["content"] for message in render(case, condition).messages)
    for spec in specs.values():
        corpus.extend([spec["suffix"]] + [spec["suffix"] + c for c in spec["continuations"]])
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    for token in sorted({w for text in corpus for w in localize.WORD.findall(text)}):
        vocabulary.setdefault(token, len(vocabulary))
    backend = Tokenizer(tokenizer_models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        chat_template="{% for message in messages %}<{{ message['role'] }}> {{ message['content'] }}\n"
                      "{% endfor %}{% if add_generation_prompt %}<assistant> {% endif %}")


def torch_direction(size):
    import torch
    generator = torch.Generator().manual_seed(29)
    vector = torch.randn(size, generator=generator, dtype=torch.float64)
    return vector / vector.norm()


def test_attention_plan_never_exceeds_the_total_call_cap():
    """The documented cap is TOTAL calls = layers_used * cases_used; per-case caps break it.

    With 8 cases and a cap of 4 the subsample must fit 4 calls in total, which forces 1 layer over 4 cases.
    A per-case reading of the cap leaves all 3 layers in place and plans 8 * 3 = 24 calls.
    """
    layers = [0, 1, 2]
    plan = localize.plan_attention(layers, list(range(8)), 4)
    assert len(plan["layers_used"]) * len(plan["cases_used"]) <= 4, plan
    assert plan["calls_planned"] <= 4
    assert plan["layers_dropped"], "the layers the cap could not afford must be reported"
    assert set(plan["layers_used"]) | set(plan["layers_dropped"]) == set(layers)
    assert not set(plan["layers_used"]) & set(plan["layers_dropped"])
    # The invariant holds over the whole grid, not just that one point.
    for case_count in (1, 2, 3, 5, 8, 12):
        for layer_count in (1, 2, 3, 6):
            for cap in (1, 2, 3, 4, 7, 16, 40):
                p = localize.plan_attention(list(range(layer_count)), list(range(case_count)), cap)
                assert len(p["layers_used"]) * len(p["cases_used"]) <= cap, (case_count, layer_count, cap, p)
                assert p["calls_planned"] == len(p["layers_used"]) * len(p["cases_used"])
                assert set(p["layers_used"]) | set(p["layers_dropped"]) == set(range(layer_count))
                assert len(p["cases_used"]) + len(p["cases_dropped"]) == case_count


def test_attention_layer_choice_reports_what_the_cap_dropped():
    layers = list(range(8))
    used, dropped = localize.choose_attention_layers(layers, 16, 4)
    assert len(used) <= 4 and set(used) | set(dropped) == set(layers) and not set(used) & set(dropped)
    assert localize.choose_attention_layers(layers, 64, 8) == (layers, [])
    with pytest.raises(SystemExit):
        localize.choose_attention_layers(layers, 4, 0)
    # The cap binds TOTAL calls, so a plan may not exceed it even when layers fit per case. With 8 cases and
    # a cap of 4 the layers must be subsampled to 1 (4 cases x 1 layer = 4 calls), not left at 4 per case.
    used, dropped = localize.choose_attention_layers(layers, 4, 8)
    assert len(used) * min(8, 4) <= 4, (used, dropped)
    assert dropped, "the cap must report the layers it dropped"


class _FakeAttentionTools:
    """Stands in for WspTools: records each call, returns per-head masses, meters its own forwards."""

    def __init__(self, calls, budget_used=1):
        self._calls = calls
        self._units = budget_used
        self.budget_used = 0

    def attention_to_span(self, layer):
        self._calls.append(layer)
        self.budget_used += self._units
        return [{"head": head, "status": "ok", "cases_used": 1,
                 "neutral": {"source_mass": 0.4, "other_mass": 0.5},
                 "misleading": {"opinion_mass": 0.1 * (head + 1), "source_mass": 0.3, "other_mass": 0.5},
                 "source_mass_change_misleading_minus_neutral": -0.1,
                 "match_failures": {"count": 0, "reasons": {}, "case_ids": []}}
                for head in range(2)]


def test_attention_stage_caps_total_calls_not_calls_per_case():
    """8 cases and a cap of 4 must issue at most 4 calls in total, and say which layers it dropped.

    Before the fix the cap was applied per case, so this stage issued layers_used * cases_used calls (8 here)
    against a documented total cap of 4.
    """
    from security_wsp.data import load_dataset
    pool = load_dataset(ROOT / "datasets" / "wsp-discovery.json")[:8]
    assert len(pool) == 8
    calls: list[int] = []
    stage = localize.attention_stage(lambda cases, budget: _FakeAttentionTools(calls), pool,
                                     [0, 1, 2], call_cap=4, units_per_call=1)
    assert len(calls) <= 4, f"issued {len(calls)} calls against a total cap of 4"
    assert stage["calls_used"] == len(calls) <= stage["call_cap"] == 4
    assert stage["call_budget"] <= stage["call_cap"]
    assert stage["calls_used"] <= stage["call_budget"]
    # The layers the cap could not afford are reported, and the layer actually invoked is accounted for.
    assert stage["layers_dropped_by_cap"], "dropped layers must be reported"
    assert set(stage["layers_invoked"]) | set(stage["layers_dropped_by_cap"]) == {0, 1, 2}
    assert sum(stage["calls_per_layer"].values()) == stage["calls_used"]
    assert stage["tool_units_used"] == stage["calls_used"], "one metered unit per issued call"
    # A cap large enough for every pair uses them all and reports no drop.
    wide_calls: list[int] = []
    wide = localize.attention_stage(lambda cases, budget: _FakeAttentionTools(wide_calls), pool,
                                    [0, 1], call_cap=64, units_per_call=1)
    assert wide["calls_used"] == len(wide_calls) <= 64
    assert wide["layers_dropped_by_cap"] == []
    assert wide["calls_used"] == len(wide["cases_used"]) * len(wide["layers_used"])


def test_forced_choice_rows_read_the_prediction_off_the_margin_sign():
    """A model that prefers the oracle continuation everywhere must score balanced accuracy 1.0.

    The bug this pins: predicted_verdict was hard-coded to VULNERABLE, so a table where the model is right on
    every clean case still predicted VULNERABLE there and true_negative stayed 0.
    """
    pytest.importorskip("torch")
    from security_wsp.statistics import metrics
    verdicts = [VULNERABLE, CLEAN, VULNERABLE, CLEAN, VULNERABLE, CLEAN]
    cases = [Case(case_id=f"c{index}", family_id=f"f{index}", source_id="s", split="wsp_discovery",
                  language="python", target_category="cat", code="", threat_model="", oracle_verdict=verdict,
                  source_sha256="0" * 64, oracle_test_ids=()) for index, verdict in enumerate(verdicts)]

    def table_with(margin_for):
        """one json_prefill block whose per-case margin is what `margin_for(case)` returns"""
        return {"json_prefill": {"condition": "incorrect_suggestion", "cases": {
            case.case_id: {"correct_margin": margin_for(case), "oracle_verdict": case.oracle_verdict,
                           "scores": {}, "continuation_tokens": {}, "prefix_sha256": "x", "prefix_tokens": 1,
                           "degenerate_continuations": False}
            for case in cases}}}

    # Prefers the oracle continuation on every case: positive margin for both verdict classes.
    perfect = localize.forced_choice_rows(table_with(lambda case: 0.7), cases, "mean_direction", "cond")
    assert len(perfect) == len(cases)
    for row in perfect:
        assert row["predicted_verdict"] == row["oracle_verdict"], row
        assert row["correct"] is True
    summary = metrics(perfect)
    assert summary["balanced_accuracy"] == pytest.approx(1.0)
    clean_count = sum(case.oracle_verdict == CLEAN for case in cases)
    assert clean_count == 3
    assert summary["true_negative"] == clean_count, "the clean cases must be predicted clean"

    # Prefers the anti-oracle continuation on every case: margin negative for both classes.
    anti = localize.forced_choice_rows(table_with(lambda case: -0.7), cases, "mean_direction", "cond")
    for row in anti:
        assert row["predicted_verdict"] == localize._opposite(row["oracle_verdict"])
        assert row["correct"] is False
    anti_summary = metrics(anti)
    assert anti_summary["balanced_accuracy"] == pytest.approx(0.0)
    assert anti_summary["true_negative"] == 0

    # A mixed table scores 0.5 on both classes, which is the only way the two above can be distinguished.
    mixed = localize.forced_choice_rows(
        table_with(lambda case: 0.7 if case.oracle_verdict == VULNERABLE else -0.7), cases, "m", "cond")
    assert metrics(mixed)["balanced_accuracy"] == pytest.approx(0.5)
    # Degenerate cases are skipped rather than guessed.
    degenerate = localize.forced_choice_rows(table_with(lambda case: None), cases, "m", "cond")
    assert degenerate == []



    components = [{"component": "mlp_out-L00"}]
    absent = localize.circuitkit_crosscheck(True, [], components)
    if importlib.util.find_spec("circuitkit") is None:
        assert absent["status"] == "absent"
        assert absent["available"] is False
        assert absent["agreement"] is None
        assert "ModuleNotFoundError" in absent["reason"]
    else:
        assert absent["status"] in ("unsupported_api", "available_but_not_run")
        assert absent["agreement"] is None
    disabled = localize.circuitkit_crosscheck(False, [], components)
    assert disabled["status"] == "disabled_by_flag" and disabled["agreement"] is None


def test_ranking_puts_degenerate_components_last():
    report = {"site": "mlp_out", "layer_index": 0, "degenerate": True, "direction": None,
              "separation": {"statistic": 0.0, "p_value": 1.0}, "random_baseline": {"mean_separation": 0.5}}
    healthy = {"site": "resid_post", "layer_index": 1, "degenerate": False, "direction": [1.0, 0.0],
               "separation": {"statistic": 0.9, "p_value": 0.01},
               "random_baseline": {"mean_separation": 0.8}, "cosine": {"cos_to_mean_direction_mean": 0.9}}
    ranked = localize.rank_components([report, healthy])
    assert [row["component"] for row in ranked] == ["resid_post-L01", "mlp_out-L00"]
    assert ranked[-1]["degenerate"] is True and ranked[-1]["direction_sha256"] is None


def test_layer_parsing_refuses_out_of_range():
    assert localize.parse_layers("all", 3) == [0, 1, 2]
    assert localize.parse_layers("2,0,2", 3) == [0, 2]
    for text in ("3", "-1", "x"):
        with pytest.raises(SystemExit):
            localize.parse_layers(text, 3)
