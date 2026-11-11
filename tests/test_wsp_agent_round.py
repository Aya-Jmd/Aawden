"""Offline tests for scripts/wsp_agent_round.py on a tiny random Cohere model. CPU only, no network.

The driver is exercised in-process through its `run()` entry point so the toy model is built once
per test rather than once per subprocess. Every test uses a 4-case balanced discovery prefix, where
one scored round is 12 prompts = 1 batch = 1 forward unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wsp_agent_round.py"
DOCS = ROOT / "docs" / "WSP_AGENT.md"


def load_driver():
    """Import the driver as a module. It must be in sys.modules first: dataclass() looks itself up."""
    import importlib.util
    import sys
    name = "wsp_agent_round"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load_driver()
LIMIT = ["--limit-cases", "4"]


def dump(tmp_path, *extra):
    return R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"), *extra])


def workbench(tmp_path) -> Path:
    return tmp_path / "run" / "workbench"


def sites_payload(tmp_path) -> dict:
    return json.loads((workbench(tmp_path) / "sites.json").read_text())


def make_spec(tmp_path, indices=(0,), tensor=None, axis=0, width=None, values=None, rationale="test edit"):
    """A valid spec built from the workbench's own site list."""
    sites = sites_payload(tmp_path)["sites"]
    entry = next(s for s in sites if (tensor is None or s["tensor_name"] == tensor))
    if width is None:
        width = entry["shape"][1 - axis]
    if values is None:
        values = [[0.0] * width for _ in indices]
    return {"schema_version": 1, "rationale": rationale,
            "edits": [{"tensor_name": entry["tensor_name"], "axis": axis,
                       "indices": list(indices), "values": values}]}


def write_spec(path: Path, spec) -> Path:
    path.write_text(json.dumps(spec, indent=2))
    return path


def apply_spec(tmp_path, spec, *extra):
    path = write_spec(tmp_path / "spec.json", spec)
    return R.run(["apply", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--spec", str(path), *extra])


def state(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / "state.json").read_text())


def save_state(tmp_path, value) -> None:
    """Tamper with state.json exactly as an outside editor would: write it, chain and all untouched."""
    (tmp_path / "run" / "state.json").write_text(json.dumps(value, indent=2))


def rewrite_state(tmp_path, mutate) -> dict:
    """Change state through the driver's own writer: reseals the chain and appends to the ledger.

    Use this to set up a cap condition. Changing the numbers without the driver is what the chain
    exists to refuse, so the tampering tests call save_state instead.
    """
    value = state(tmp_path)
    mutate(value)
    R.save_state(tmp_path / "run", value)
    return value


# ---------------------------------------------------------------------------------------------
# dump and the workbench
# ---------------------------------------------------------------------------------------------
def test_dump_writes_workbench_for_the_programmer(tmp_path):
    result = dump(tmp_path)
    assert result["mode"] == "dump"
    directory = workbench(tmp_path)
    for name in ("README.md", "protocol.json", "sites.json", "contrasts.json", "causal.json",
                 "attention.json", "baseline.json", "schema.json", "manifest.json"):
        assert (directory / name).is_file(), name
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == R.WORKBENCH_FORMAT
    assert manifest["total_bytes"] <= manifest["cap_bytes"]
    assert manifest["truncated"] == []
    assert manifest["total_bytes"] == sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def test_workbench_describes_the_edit_contract(tmp_path):
    dump(tmp_path)
    schema = json.loads((workbench(tmp_path) / "schema.json").read_text())
    spec = schema["json_schema"]
    assert spec["required"] == ["rationale", "edits"]
    edit = spec["properties"]["edits"]["items"]
    assert sorted(edit["required"]) == ["axis", "indices", "tensor_name", "values"]
    assert edit["properties"]["axis"]["enum"] == [0, 1]
    assert schema["format_name"] == R.EDIT_SPEC_FORMAT
    protocol = json.loads((workbench(tmp_path) / "protocol.json").read_text())
    assert protocol["edit_schema_version"] == R.EDIT_SCHEMA_VERSION
    assert protocol["limits"]["forward_pass_budget"] == 50
    assert protocol["limits"]["max_submissions"] == 10
    assert protocol["limits"]["max_changed_fraction"] == 0.0005
    assert protocol["output_format"]["json_prefill"] == '{"verdict":"'
    # The arithmetic the round actually uses, recorded rather than assumed.
    arithmetic = protocol["budget_arithmetic"]
    assert arithmetic["prompts_per_round"] == arithmetic["cases"] * len(protocol["conditions"])
    assert arithmetic["batches_per_round"] == R.round_units(arithmetic["prompts_per_round"], 16)


def test_workbench_sites_are_editable_only(tmp_path):
    dump(tmp_path)
    sites = sites_payload(tmp_path)
    patterns = json.loads((workbench(tmp_path) / "protocol.json").read_text())["limits"]
    assert sites["count"] == len(sites["sites"])
    for entry in sites["sites"]:
        assert len(entry["shape"]) == 2
        assert entry["total_scalars"] == entry["shape"][0] * entry["shape"][1]
        assert entry["top_columns"], "the programmer needs column norms"
        assert entry["column_norm_summary"]["count"] == entry["shape"][1]
    names = {s["tensor_name"] for s in sites["sites"]}
    assert any("down_proj" in name for name in names)
    assert all(("mlp." in name or "self_attn." in name) for name in names)


def test_dump_reports_budget_and_baseline(tmp_path):
    result = dump(tmp_path)
    assert result["baseline_available"] is True
    baseline = json.loads((workbench(tmp_path) / "baseline.json").read_text())
    assert baseline["available"] is True
    for condition in ("neutral", "correct_suggestion", "incorrect_suggestion"):
        assert condition in baseline["metrics"]
    # A cold dump spends one contrast plus the baseline round.
    assert result["budget"]["forward_units_used"] == 2
    assert result["budget"]["forward_units_remaining"] == 48


def test_workbench_byte_cap_is_respected_and_recorded(tmp_path):
    # A cap below the untruncated size, so the reduction path is exercised.
    untruncated = dump(tmp_path)["workbench_bytes"]
    cap = untruncated - 4000
    result = dump(tmp_path, "--workbench-byte-cap", str(cap))
    directory = workbench(tmp_path)
    actual = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    assert actual <= cap
    assert result["workbench_bytes"] == actual
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["truncated"], "reductions must be recorded"
    assert result["truncated"]
    for record in manifest["truncated"]:
        assert set(record) == {"file", "action", "kept", "dropped"}
    readme = (directory / "README.md").read_text()
    assert "cannot be" not in readme
    assert "Reduced to fit" in readme


def test_workbench_cap_too_small_is_refused(tmp_path):
    with pytest.raises(R.Refused):
        dump(tmp_path, "--workbench-byte-cap", "4096")


def test_dump_is_idempotent_and_reuses_the_baseline(tmp_path):
    first = dump(tmp_path)
    second = dump(tmp_path)
    assert second["budget"]["forward_units_used"] == first["budget"]["forward_units_used"]
    assert second["baseline_available"] is True
    assert json.loads((workbench(tmp_path) / "baseline.json").read_text())["source"]["kind"] != "budget"


# ---------------------------------------------------------------------------------------------
# apply: one round
# ---------------------------------------------------------------------------------------------
def test_apply_writes_round_artifacts(tmp_path):
    dump(tmp_path)
    before = state(tmp_path)["budget"]["forward_units_used"]
    result = apply_spec(tmp_path, make_spec(tmp_path))
    assert result["mode"] == "apply" and result["round"] == 1
    directory = tmp_path / "run" / "rounds" / "round-01"
    for name in ("spec.json", "results.json", "summary.json"):
        assert (directory / name).is_file(), name

    summary = json.loads((directory / "summary.json").read_text())
    assert summary["format"] == R.ROUND_FORMAT
    assert summary["kind"] == "proposal"
    for condition in ("neutral", "correct_suggestion", "incorrect_suggestion"):
        metrics = summary["metrics"][condition]
        assert "balanced_accuracy" in metrics
        # Per-class counts the protocol scores with, all present.
        for key in ("n_vulnerable", "n_clean", "true_positive", "true_negative", "false_positive",
                    "false_negative", "invalid_count"):
            assert key in metrics, key
    assert summary["restoration"]["verified"] is True
    before_hashes = summary["restoration"]["tensor_sha256_before"]
    edited_hashes = summary["restoration"]["tensor_sha256_after_apply"]
    assert before_hashes and set(before_hashes) == set(edited_hashes)
    # The recorded "before" hash is the base model's tensor, so a later reader can prove the edit
    # started from the base, and the edited hash differs wherever a value actually changed.
    assert all(isinstance(value, str) and len(value) == 64 for value in before_hashes.values())
    assert any(before_hashes[name] != edited_hashes[name] for name in before_hashes), \
        "an edit that changes values must change the tensor hash"
    # The spec on disk is the spec that was applied.
    assert json.loads((directory / "spec.json").read_text()) == make_spec(tmp_path)
    assert summary["spec"]["sha256"] == R.file_hash(directory / "spec.json")
    # Artifacts are hashed into state, so every reported number is traceable.
    record = state(tmp_path)["rounds"][-1]
    for name, expected in record["artifacts"].items():
        assert R.file_hash(directory / name) == expected
    # A round is one batch at this size.
    assert state(tmp_path)["budget"]["forward_units_used"] == before + 1
    assert result["budget"]["forward_units_remaining"] == 50 - before - 1


def test_apply_records_per_class_counts_from_results(tmp_path):
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path))
    directory = tmp_path / "run" / "rounds" / "round-01"
    results = json.loads((directory / "results.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    (result_set,) = results["edit_sets"]
    assert summary["misleading_balanced_accuracy"] == result_set["misleading_balanced_accuracy"]
    assert summary["per_class_counts"] == result_set["per_class_counts"]
    assert summary["changed_scalars"] == result_set["changed_scalars"]
    assert summary["restoration"] == result_set["restoration"]
    # Every row is retained, so a number can be recomputed from answers.
    assert len(result_set["rows"]) == results["prompts_per_set"]
    assert {row["condition"] for row in result_set["rows"]} == {
        "neutral", "correct_suggestion", "incorrect_suggestion"}


def test_second_round_diffs_against_the_previous(tmp_path):
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path, indices=(0,)))
    result = apply_spec(tmp_path, make_spec(tmp_path, indices=(0, 3)))
    assert result["round"] == 2
    diff = result["diff"]
    assert diff["vs_previous"]["round"] == 1
    slices = diff["vs_previous"]["slices"]
    assert slices["shared"] == 1 and slices["added"] == 1 and slices["removed"] == 0
    assert diff["vs_baseline"]["misleading_balanced_accuracy"]["delta"] is not None
    assert diff["vs_previous"]["update_norm"]["current"] is not None


def test_column_edit_is_supported(tmp_path):
    dump(tmp_path)
    spec = make_spec(tmp_path, indices=(0, 1), axis=1)
    result = apply_spec(tmp_path, spec)
    assert result["round"] == 1
    summary = json.loads((tmp_path / "run" / "rounds" / "round-01" / "summary.json").read_text())
    # Two columns of a tensor whose axis-1 size is the row count: the touched count is the tensor's
    # shape[0] times the number of indices.
    entry = next(s for s in sites_payload(tmp_path)["sites"]
                 if s["tensor_name"] == spec["edits"][0]["tensor_name"])
    assert summary["edits"]["axis_counts"] == {"rows": 0, "columns": 1}
    assert summary["touched_scalars"] == entry["shape"][0] * 2
    assert summary["changed_scalars"] > 0
    assert summary["restoration"]["verified"] is True


def test_submission_patch_is_written_and_reloadable(tmp_path):
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path))
    patches = sorted((tmp_path / "run" / "submissions").glob("submission-*.json"))
    assert len(patches) == 1
    manifest = json.loads(patches[0].read_text())
    assert manifest["extra"]["discovery_only"] is True
    assert manifest["weights_sha256"]
    assert (tmp_path / "run" / "submissions" / manifest["weights_file"]).is_file()


# ---------------------------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------------------------
def _validate(tmp_path, spec):
    payload = sites_payload(tmp_path)
    sites = R.site_index(payload["sites"])
    return R.validate_spec(spec, sites, payload["total_parameters"], 0.0005)


@pytest.mark.parametrize("case", ["unsorted", "duplicate", "out_of_range", "unknown_tensor",
                                  "wrong_width", "non_finite", "extra_key", "twice_same_tensor",
                                  "no_edits", "bad_rationale", "unknown_top_key"])
def test_schema_refusals(tmp_path, case):
    dump(tmp_path)
    spec = make_spec(tmp_path, indices=(0,))
    entry = spec["edits"][0]
    width = len(entry["values"][0])
    if case == "unsorted":
        entry.update(indices=[3, 0], values=[[0.0] * width] * 2)
    elif case == "duplicate":
        entry.update(indices=[1, 1], values=[[0.0] * width] * 2)
    elif case == "out_of_range":
        entry.update(indices=[10 ** 6], values=[[0.0] * width])
    elif case == "unknown_tensor":
        entry["tensor_name"] = "model.layers.0.self_attn.q_proj.weight"
    elif case == "wrong_width":
        entry.update(indices=[0], values=[[0.0] * (width - 1)])
    elif case == "non_finite":
        entry.update(indices=[0], values=[[float("nan")] * width])
    elif case == "extra_key":
        entry["note"] = "not in the schema"
    elif case == "twice_same_tensor":
        spec["edits"].append(dict(entry, indices=[1], values=[[0.0] * width]))
    elif case == "no_edits":
        spec["edits"] = []
    elif case == "bad_rationale":
        spec["rationale"] = ""
    elif case == "unknown_top_key":
        spec["notes"] = "x"
    with pytest.raises(R.SpecError):
        _validate(tmp_path, spec)


def test_schema_reports_every_problem_at_once(tmp_path):
    dump(tmp_path)
    spec = make_spec(tmp_path, indices=(0,))
    entry = spec["edits"][0]
    width = len(entry["values"][0])
    spec["edits"] = [dict(entry, indices=[5, 1], values=[[0.0] * width] * 2),
                     {"tensor_name": "model.layers.0.mlp.gate_proj.weight", "axis": 7,
                      "indices": [0], "values": [[0.0] * width]}]
    with pytest.raises(R.SpecError) as error:
        _validate(tmp_path, spec)
    joined = " ".join(error.value.problems)
    assert "sorted" in joined and "axis" in joined
    assert len(error.value.problems) >= 2


def test_oversized_fraction_is_refused_with_arithmetic(tmp_path):
    dump(tmp_path)
    sites = sites_payload(tmp_path)
    entry = next(s for s in sites["sites"]
                 if f"{s['shape'][1] * len(s['shape'])}" and s["shape"][1] == max(x["shape"][1] for x in sites["sites"]))
    width = entry["shape"][1]
    spec = {"schema_version": 1, "rationale": "too large on purpose",
            "edits": [{"tensor_name": entry["tensor_name"], "axis": 0, "indices": [0, 1, 2],
                       "values": [[0.0] * width] * 3}]}
    with pytest.raises(R.SpecError) as error:
        _validate(tmp_path, spec)
    message = " ".join(error.value.problems)
    assert "changed fraction" in message
    assert "limit is" in message
    assert str(sites["total_parameters"]) in message


def test_valid_spec_returns_edit_summary(tmp_path):
    dump(tmp_path)
    sites = sites_payload(tmp_path)
    spec = make_spec(tmp_path)
    edits, summary = _validate(tmp_path, spec)
    assert len(edits) == 1
    assert summary["touched_scalars"] > 0
    assert summary["touched_fraction"] <= 0.0005
    assert summary["tensors"] == [spec["edits"][0]["tensor_name"]]


def test_apply_refuses_bad_spec_without_spending_budget(tmp_path):
    dump(tmp_path)
    before = state(tmp_path)["budget"]["forward_units_used"]
    bad = make_spec(tmp_path, indices=(0,))
    bad["edits"][0]["indices"] = [9, 2]
    width = len(bad["edits"][0]["values"][0])
    bad["edits"][0]["values"] = [[0.0] * width] * 2
    with pytest.raises(R.SpecError):
        apply_spec(tmp_path, bad)
    assert state(tmp_path)["budget"]["forward_units_used"] == before
    assert not (tmp_path / "run" / "rounds" / "round-01" / "results.json").exists()


def test_check_validates_without_a_model_or_budget(tmp_path):
    dump(tmp_path)
    before = state(tmp_path)["budget"]["forward_units_used"]
    path = write_spec(tmp_path / "s.json", make_spec(tmp_path))
    result = R.run(["check", "--toy", "--output", str(tmp_path / "run"), "--spec", str(path)])
    assert result["valid"] is True
    assert result["forward_units_needed"] == 1
    assert result["affordable"] is True
    assert result["submissions_remaining"] == 10
    assert state(tmp_path)["budget"]["forward_units_used"] == before


def test_check_refuses_bad_spec(tmp_path):
    dump(tmp_path)
    spec = make_spec(tmp_path)
    spec["edits"][0]["tensor_name"] = "model.layers.0.mlp.nonexistent.weight"
    path = write_spec(tmp_path / "s.json", spec)
    with pytest.raises(R.SpecError):
        R.run(["check", "--toy", "--output", str(tmp_path / "run"), "--spec", str(path)])


# ---------------------------------------------------------------------------------------------
# budget and caps across invocations
# ---------------------------------------------------------------------------------------------
def test_budget_carries_across_invocations(tmp_path):
    dump(tmp_path)
    after_dump = state(tmp_path)["budget"]
    apply_spec(tmp_path, make_spec(tmp_path, indices=(0,)))
    after_first = state(tmp_path)["budget"]
    apply_spec(tmp_path, make_spec(tmp_path, indices=(1,)))
    after_second = state(tmp_path)["budget"]
    assert after_dump["forward_units_used"] == 2
    assert after_first["forward_units_used"] == 3
    assert after_second["forward_units_used"] == 4
    assert after_second["forward_units_remaining"] == 46
    # Counts persist too, so a restarted process cannot exceed the protocol caps.
    counts = state(tmp_path)["counts"]
    assert counts["rounds"] == 2 and counts["proposals"] == 2 and counts["submissions"] == 2
    assert state(tmp_path)["next_round"] == 3


def test_exhausted_budget_refuses_before_scoring(tmp_path):
    dump(tmp_path)
    rewrite_state(tmp_path, lambda payload: payload.update(budget=R.budget_block(50, 50)))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "forward-pass budget" in str(error.value)
    # No side effect: the round left no artifacts and spent nothing.
    assert not (tmp_path / "run" / "rounds" / "round-01" / "results.json").exists()
    assert not (tmp_path / "run" / "rounds" / "round-01" / "summary.json").exists()
    assert not list((tmp_path / "run" / "submissions").glob("submission-*.json"))
    assert state(tmp_path)["budget"]["forward_units_used"] == 50
    assert state(tmp_path)["counts"]["rounds"] == 0


def test_insufficient_budget_reports_remaining(tmp_path):
    """One unit short: the round is refused with the arithmetic, not attempted."""
    dump(tmp_path)
    rewrite_state(tmp_path, lambda payload: payload.update(budget=R.budget_block(50, 50)))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "exhausted" in str(error.value)
    assert not (tmp_path / "run" / "rounds").exists() or not list(
        (tmp_path / "run" / "rounds").glob("round-*/results.json"))

    # One unit of headroom is not enough for a one-batch round only if it needs more; at this size
    # one round is one unit, so prove the shortfall path directly.
    with pytest.raises(R.Refused) as short:
        R.require_units({"budget": R.budget_block(49, 50)},
                        {"forward_pass_budget": 50}, 2, "this round")
    assert "remain" in str(short.value) and "49" in str(short.value)


def test_submission_limit_is_enforced(tmp_path):
    dump(tmp_path)
    rewrite_state(tmp_path, lambda payload: payload["counts"].update(submissions=10))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "submission limit" in str(error.value)
    # Refused before any work: nothing was scored, nothing was submitted, no unit was spent.
    assert not (tmp_path / "run" / "rounds" / "round-01" / "results.json").exists()
    assert not list((tmp_path / "run" / "submissions").glob("submission-*.json"))
    assert state(tmp_path)["budget"]["forward_units_used"] == 2
    assert state(tmp_path)["counts"]["submissions"] == 10


def test_edited_recipe_limit_is_enforced(tmp_path):
    dump(tmp_path)
    rewrite_state(tmp_path, lambda payload: payload["counts"].update(proposals=10))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "edited-recipe limit" in str(error.value)
    assert not (tmp_path / "run" / "rounds" / "round-01" / "results.json").exists()
    assert state(tmp_path)["budget"]["forward_units_used"] == 2
    assert state(tmp_path)["counts"]["proposals"] == 10


def test_resume_rejects_a_changed_identity(tmp_path):
    dump(tmp_path)
    with pytest.raises(R.Refused) as error:
        R.run(["apply", "--toy", *LIMIT, "--strong", "--output", str(tmp_path / "run"),
               "--spec", str(write_spec(tmp_path / "s.json", make_spec(tmp_path)))])
    assert "resume rejected" in str(error.value).lower() or "Rejected" in str(error.value)


def test_apply_without_dump_is_refused(tmp_path):
    with pytest.raises(R.Refused):
        apply_spec(tmp_path, {"schema_version": 1, "rationale": "x", "edits": []})


# ---------------------------------------------------------------------------------------------
# state integrity: the hash chain and the artifact hashes
# ---------------------------------------------------------------------------------------------
def test_state_is_hash_chained(tmp_path):
    dump(tmp_path)
    payload = state(tmp_path)
    chain = payload["chain"]
    assert chain["length"] >= 1 and chain["previous"] is not None
    assert len(chain["head"]) == 64 and len(chain["body_sha256"]) == 64
    ledger = [json.loads(line) for line in
              (tmp_path / "run" / "state-chain.jsonl").read_text().splitlines() if line.strip()]
    assert ledger[-1]["head"] == chain["head"]
    assert ledger[-1]["length"] == chain["length"]
    # Every write appends one link, so a later load can confirm the head was written by the driver.
    assert len(ledger) == chain["length"]
    apply_spec(tmp_path, make_spec(tmp_path))
    advanced = state(tmp_path)["chain"]
    assert advanced["length"] > chain["length"]
    assert advanced["previous"] is not None


def test_edited_budget_in_state_is_refused(tmp_path):
    """Editing the budget in state.json between restarts cannot exceed the protocol caps."""
    dump(tmp_path)
    payload = state(tmp_path)
    payload["budget"] = R.budget_block(0, 50)
    save_state(tmp_path, payload)
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "hash-chain" in str(error.value)
    assert not (tmp_path / "run" / "rounds" / "round-01" / "results.json").exists()


def test_edited_counts_in_state_are_refused(tmp_path):
    """A doctored count can never be loaded: the chain covers every key of the state body."""
    dump(tmp_path)
    payload = state(tmp_path)
    # A round was never applied, so the honest counts are zero; claim one submission was already made
    # to buy headroom against the submission cap.
    payload["counts"]["submissions"] = 9
    payload["counts"]["proposals"] = 9
    save_state(tmp_path, payload)
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "hash-chain" in str(error.value)
    assert not (tmp_path / "run" / "rounds" / "round-01" / "summary.json").exists()


def test_state_resealed_offline_does_not_match_the_ledger(tmp_path):
    """A state file rewritten with a freshly computed chain is refused against the recorded ledger."""
    dump(tmp_path)
    payload = state(tmp_path)
    payload["budget"] = R.budget_block(0, 50)
    R.seal_state(payload)
    (tmp_path / "run" / "state.json").write_text(json.dumps(payload, indent=2))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "ledger" in str(error.value)


def test_state_without_a_chain_is_refused(tmp_path):
    dump(tmp_path)
    payload = state(tmp_path)
    payload.pop("chain")
    (tmp_path / "run" / "state.json").write_text(json.dumps(payload, indent=2))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "hash chain" in str(error.value)


def test_edited_prior_round_summary_is_refused(tmp_path):
    """A round summary edited after it was written is not consumed silently by the next round."""
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path, indices=(0,)))
    directory = tmp_path / "run" / "rounds" / "round-01"
    summary = json.loads((directory / "summary.json").read_text())
    summary["misleading_balanced_accuracy"] = 1.0
    summary["changed_scalars"] = 999999
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path, indices=(0, 3)))
    assert "summary.json" in str(error.value)
    assert "edited afterwards" in str(error.value)
    # No second round was written from the tampered record.
    assert not (tmp_path / "run" / "rounds" / "round-02" / "summary.json").exists()


def test_missing_prior_round_artifact_is_refused(tmp_path):
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path))
    (tmp_path / "run" / "rounds" / "round-01" / "results.json").unlink()
    with pytest.raises(R.Refused) as error:
        apply_spec(tmp_path, make_spec(tmp_path))
    assert "missing" in str(error.value)


# ---------------------------------------------------------------------------------------------
# what a round costs and what it scores
# ---------------------------------------------------------------------------------------------
def test_check_charges_the_cases_actually_scored(tmp_path):
    """The charge follows the case count; the editable-tensor count is not a case count.

    At four cases the two numbers coincide by accident (both cost one batch), so the load-bearing
    check is `test_check_charges_a_full_discovery_round_correctly`, where the old fallback would have
    charged one unit for a twelve-unit round. This test pins the case count the charge was built from.
    """
    dump(tmp_path)
    sites = sites_payload(tmp_path)
    path = write_spec(tmp_path / "s.json", make_spec(tmp_path))
    result = R.run(["check", "--toy", "--output", str(tmp_path / "run"), "--spec", str(path)])
    assert result["cases"] == 4
    assert result["prompts_per_round"] == result["cases"] * 3
    assert result["forward_units_needed"] == R.round_units(result["prompts_per_round"], 16)
    # The case count is what the charge was built from: the tensor count is a different quantity.
    assert result["cases"] != sites["count"], "the charge must not be the editable-tensor count"


def test_check_charges_a_full_discovery_round_correctly(tmp_path):
    """With no --limit-cases the charge is the committed 60 cases, not 16 tensors and not 1 unit."""
    R.run(["dump", "--toy", "--output", str(tmp_path / "run")])
    path = write_spec(tmp_path / "s.json", make_spec(tmp_path))
    result = R.run(["check", "--toy", "--output", str(tmp_path / "run"), "--spec", str(path)])
    assert result["cases"] == 60
    assert result["prompts_per_round"] == 180
    assert result["forward_units_needed"] == 12


def test_limit_prompts_reaches_the_scorer(tmp_path):
    """A small --limit-prompts reduces the rows scored and the units charged together."""
    flags = ("--limit-prompts", "3", "--batch-size", "4")
    dump(tmp_path, *flags)
    before = state(tmp_path)["budget"]["forward_units_used"]
    spec = make_spec(tmp_path, indices=(0,))
    # batch 4 with 3 prompts: one batch of work, one unit, and exactly 3 rows scored.
    result = apply_spec(tmp_path, spec, *flags)
    directory = tmp_path / "run" / "rounds" / "round-01"
    results = json.loads((directory / "results.json").read_text())
    (result_set,) = results["edit_sets"]
    assert results["prompts_per_set"] == 3, "the scorer must not ignore --limit-prompts"
    assert len(result_set["rows"]) == 3
    assert results["forward_units_per_set"] == 1
    assert state(tmp_path)["budget"]["forward_units_used"] == before + 1
    assert result["budget"]["forward_units_remaining"] == 50 - before - 1


def test_limit_prompts_with_full_width_reduces_the_charge(tmp_path):
    """Without --limit-prompts the same round scores 12 rows and costs 3 units at batch 4."""
    flags = ("--batch-size", "4")
    dump(tmp_path, *flags)
    before = state(tmp_path)["budget"]["forward_units_used"]
    apply_spec(tmp_path, make_spec(tmp_path, indices=(0,)), *flags)
    results = json.loads((tmp_path / "run" / "rounds" / "round-01" / "results.json").read_text())
    assert results["prompts_per_set"] == 12
    assert results["forward_units_per_set"] == 3
    assert state(tmp_path)["budget"]["forward_units_used"] == before + 3


# ---------------------------------------------------------------------------------------------
# dataset and wording refusals
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,split", [("datasets/wsp-final.json", "sealed"),
                                        ("datasets/wsp-validation.json", "validation")])
def test_refuses_sealed_and_validation_datasets(tmp_path, name, split):
    with pytest.raises(R.Refused) as error:
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--dataset", name])
    assert split in str(error.value)
    assert not (tmp_path / "run" / "workbench").exists()


def test_refuses_a_sealed_dataset_by_absolute_path(tmp_path):
    with pytest.raises(R.Refused):
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"),
               "--dataset", str(ROOT / "datasets" / "wsp-final.json")])


def test_sealed_dataset_is_never_opened(tmp_path, monkeypatch):
    """The seal is enforced by name, before any file handle: nothing at all is read.

    The spy intercepts every read path the driver could use — Path.read_text, Path.open and io.open
    — because an earlier version of this test patched only builtins.open while the code read through
    pathlib, which reaches io.open, so the assertion could never fail.
    """
    import io
    opened = []
    real_read_text, real_path_open, real_io_open = Path.read_text, Path.open, io.open

    def spy_read_text(self, *args, **kwargs):
        opened.append(str(self))
        return real_read_text(self, *args, **kwargs)

    def spy_path_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    def spy_io_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)
    monkeypatch.setattr(Path, "open", spy_path_open)
    monkeypatch.setattr(io, "open", spy_io_open)

    # Positive control: the spy does see a legitimate read, so a clean result means something.
    R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "control")])
    assert any("wsp-discovery" in name for name in opened), opened

    opened.clear()
    with pytest.raises(R.Refused) as error:
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"),
               "--dataset", str(ROOT / "datasets" / "wsp-final.json")])
    assert "sealed final" in str(error.value)
    assert not any("wsp-final" in name for name in opened), opened


def test_sealed_split_handed_in_as_localization_is_refused(tmp_path):
    """A sealed split supplied as --localization never reaches the round builder.

    The stand-in is a file *named* wsp-final.json holding harmless JSON: the refusal fires on the
    name, so the real sealed split is never opened by this test either.
    """
    stand_in = tmp_path / "wsp-final.json"
    stand_in.write_text(json.dumps({"entries": []}))
    with pytest.raises(R.Refused) as error:
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"),
               "--localization", str(stand_in)])
    assert "sealed final" in str(error.value)
    assert "--localization" in str(error.value)
    assert not (tmp_path / "run" / "workbench").exists()
    assert not (tmp_path / "run" / "state.json").exists()


def test_validation_split_handed_in_as_baseline_scores_is_refused(tmp_path):
    with pytest.raises(R.Refused) as error:
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"),
               "--baseline-scores", str(ROOT / "datasets" / "wsp-validation.json")])
    assert "validation split" in str(error.value)
    assert not (tmp_path / "run" / "workbench").exists()


def test_sealed_split_is_recognised_by_content_under_another_name(tmp_path):
    """A copy of the sealed split under an innocent name is caught by its committed sha256.

    The copy is harmless JSON whose sha256 is written into a stand-in commitments dict, so the
    content check is exercised without opening the real sealed file.
    """
    copy = tmp_path / "contrasts-localization.json"
    copy.write_text(json.dumps({"entries": [], "note": "harmless stand-in"}))
    protocol = json.loads((ROOT / "configs" / "protocol-v3.json").read_text())
    commitments = json.loads((ROOT / "configs" / "wsp-data.json").read_text())
    commitments["files"]["wsp_final"]["sha256"] = R.file_hash(copy)
    assert R.split_kind(copy, protocol, commitments) == "final"
    with pytest.raises(R.Refused) as error:
        R.refuse_split(copy, protocol, commitments, "--localization")
    assert "sealed final" in str(error.value)


def test_tampered_localization_is_refused(tmp_path):
    """A localization.json pinned by the first dump cannot be swapped for a different file."""
    localization = tmp_path / "localization.json"
    localization.write_text(json.dumps({"entries": []}))
    dump(tmp_path, "--localization", str(localization), "--no-contrasts")
    pinned = state(tmp_path)["inputs"]["localization"]
    assert pinned["sha256"] == R.file_hash(localization)
    # The pin is in the workbench too, so a reader sees which file a round used.
    protocol = json.loads((workbench(tmp_path) / "protocol.json").read_text())
    assert protocol["inputs"]["localization"]["sha256"] == R.file_hash(localization)

    budget_before = state(tmp_path)["budget"]["forward_units_used"]
    localization.write_text(json.dumps({"entries": [], "note": "swapped after the pin"}))
    with pytest.raises(R.Refused) as error:
        dump(tmp_path, "--localization", str(localization), "--no-contrasts")
    assert "localization" in str(error.value) and "changed since" in str(error.value)
    # Refused before any work: the budget did not move and the pin was not overwritten.
    assert state(tmp_path)["budget"]["forward_units_used"] == budget_before
    assert state(tmp_path)["inputs"]["localization"]["sha256"] == pinned["sha256"]


def test_sealed_path_guard_is_not_dead_code():
    """The sealed path set is built from the commitment, without a doubled prefix."""
    commitments = json.loads((ROOT / "configs" / "wsp-data.json").read_text())
    paths = R.sealed_paths(commitments)
    expected = str((ROOT / "datasets" / "wsp-final.json").resolve())
    assert paths == {expected}, paths
    assert not any("wsp-wsp_" in path for path in paths), paths
    assert R.sealed_names(commitments) == {"wsp-final.json"}


def test_refuses_an_uncommitted_dataset(tmp_path):
    copy = tmp_path / "wsp-discovery.json"
    data = json.loads((ROOT / "datasets" / "wsp-discovery.json").read_text())
    data["cases"][0]["oracle_verdict"] = ("not_vulnerable_for_target"
                                          if data["cases"][0]["oracle_verdict"] == "vulnerable" else "vulnerable")
    copy.write_text(json.dumps(data))
    with pytest.raises(R.Refused):
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--dataset", str(copy)])


@pytest.mark.parametrize("wording", ["authority", "ciso", "scanner", "consensus", "code_comment"])
def test_refuses_transfer_wordings(tmp_path, wording):
    with pytest.raises(R.Refused) as error:
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--wording", wording])
    assert "transfer wording" in str(error.value)


def test_accepts_the_development_wording(tmp_path):
    result = R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--wording", "reviewer"])
    assert result["mode"] == "dump"
    captured = state(tmp_path)["identity"]["scoring"]
    assert captured["wording"] == "reviewer"


def test_refuses_an_unknown_wording(tmp_path):
    with pytest.raises(R.Refused):
        R.run(["dump", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--wording", "hint"])


# ---------------------------------------------------------------------------------------------
# random control
# ---------------------------------------------------------------------------------------------
def test_random_control_matches_the_proposal(tmp_path):
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path, indices=(0,)))
    result = R.run(["random-control", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--from", "1"])
    assert result["mode"] == "random-control"
    assert result["seeds"] == [17, 29, 43]
    assert result["rounds"] == [2, 3, 4]
    for number in result["rounds"]:
        summary = json.loads((tmp_path / "run" / "rounds" / f"round-{number:02d}" / "summary.json").read_text())
        assert summary["kind"] == "random_control"
        assert summary["source_round"] == 1
        control = summary["control"]
        assert control["within_1e-3"] is True
        assert control["relative_norm_mismatch"] <= 1e-3
        # Size matched: the same number of changed slices as the proposal.
        assert sum(control["changed_slices"].values()) == 1
        assert summary["restoration"]["verified"] is True
    counts = state(tmp_path)["counts"]
    assert counts["proposals"] == 1 and counts["random_controls"] == 3


def test_random_control_needs_a_source_round(tmp_path):
    dump(tmp_path)
    with pytest.raises(R.Refused):
        R.run(["random-control", "--toy", *LIMIT, "--output", str(tmp_path / "run"), "--from", "7"])


def test_random_control_accepts_explicit_seeds(tmp_path):
    dump(tmp_path)
    apply_spec(tmp_path, make_spec(tmp_path, indices=(0,)))
    result = R.run(["random-control", "--toy", *LIMIT, "--output", str(tmp_path / "run"),
                    "--from", "1", "--seeds", "5"])
    assert result["seeds"] == [5]
    assert len(result["rounds"]) == 1


# ---------------------------------------------------------------------------------------------
# documentation
# ---------------------------------------------------------------------------------------------
def test_docs_and_readme_name_every_refusal(tmp_path):
    dump(tmp_path)
    readme = (workbench(tmp_path) / "README.md").read_text()
    docs = DOCS.read_text()
    assert DOCS.is_file()
    missing = {where: [term for term in R.REFUSAL_TERMS if term.lower() not in text.lower()]
               for where, text in (("docs", docs), ("README", readme))}
    assert not any(missing.values()), missing


def test_docs_states_the_budget_arithmetic():
    docs = DOCS.read_text()
    assert "RTX 5090" in docs
    # The committed discovery size, not the 100-case figure sometimes quoted for this phase.
    assert "60 cases" in docs
    assert "12" in docs and "180" in docs
    assert "UNKNOWN - not measured" in docs


def test_docs_toy_parameter_count_matches_the_measured_model(tmp_path):
    """The documented toy size is the measured one: 469,056 parameters, 234 admitted scalars.

    The figure in the docs used to read 247,680 parameters admitting 123 scalars, which matched
    neither the driver's path nor any measured build.
    """
    R.run(["dump", "--toy", "--output", str(tmp_path / "run")])
    total = sites_payload(tmp_path)["total_parameters"]
    assert total == 469_056, total
    allowed = int(0.0005 * total)
    assert allowed == 234, allowed
    docs = DOCS.read_text()
    assert "469,056" in docs
    assert "234" in docs
    assert "247,680" not in docs


def test_readme_is_written_for_the_programmer(tmp_path):
    dump(tmp_path)
    readme = (workbench(tmp_path) / "README.md").read_text()
    assert "You are the programmer" in readme
    assert R.EDIT_SPEC_FORMAT in readme
    assert '"rationale"' in readme
    assert "sealed" in readme.lower()


def test_driver_holds_the_protocol_values_it_claims():
    protocol = json.loads((ROOT / "configs" / "protocol-v3.json").read_text())
    limits = R.limits_from(protocol, ROOT / "configs" / "protocol-v3.json")
    assert limits["forward_pass_budget"] == protocol["wsp_agent"]["forward_pass_budget"] == 50
    assert limits["max_submissions"] == protocol["wsp_agent"]["max_submissions"] == 10
    assert limits["max_changed_fraction"] == protocol["wsp_agent"]["max_changed_fraction_of_weights"]
    assert limits["max_edited_recipes"] == protocol["selection"]["max_edit_candidates_per_method"] == 10
    assert limits["transfer_wordings_final_only"] == protocol["transfer_wordings_final_only"]
    assert limits["json_prefill"] == protocol["output_format"]["json_prefill"]
    assert limits["wording"] == protocol["primary_wording"] == "reviewer"


def test_round_unit_arithmetic():
    assert R.round_units(12, 16) == 1
    assert R.round_units(180, 16) == 12      # 60 committed discovery cases x 3 conditions
    assert R.round_units(300, 16) == 19      # 100 cases, the figure this phase is sometimes quoted with
    assert R.prompts_for(60, ("neutral", "correct_suggestion", "incorrect_suggestion")) == 180
    assert R.prompts_for(60, ("a", "b", "c"), limit_prompts=10) == 10


# ---------------------------------------------------------------------------------------------
# localization carried into the workbench
# ---------------------------------------------------------------------------------------------
def _localization_dir(tmp_path, components, heads):
    """A minimal hand-made localization directory: localization.json plus a sibling attention.json."""
    directory = tmp_path / "localization"
    directory.mkdir(parents=True, exist_ok=True)
    document = {"claim_policy": {"exploratory": "exploratory localization, not a causal claim"},
                "causal": {"components": components, "random_seeds": [17, 29, 43], "strength": 1.0}}
    (directory / "localization.json").write_text(json.dumps(document))
    (directory / "attention.json").write_text(json.dumps({
        "heads_ranked": heads,
        "layers_used": [18],
        "statistic_definition": "relative opinion mass, misleading minus neutral"}))
    return directory / "localization.json"


def _component(name="attn_out-L18", supported=False):
    return {"component": name, "site": "attn_out", "layer": 18, "rank": 1, "score": 0.25,
            "strength": 1.0, "projected_tensors": ["model.layers.18.self_attn.o_proj.weight"],
            "primary_instrument": "json_prefill",
            "effect": {"cases": 12, "mean_margin_delta": 0.604, "noop_max_absolute_delta": 0.0,
                       "sign_flip": {"p_value": 0.03}, "family_interval": {"ci95": [0.13, 0.62]}},
            "no_op_control": {"max_absolute_margin_delta": 0.0, "passed": True},
            "largest_random_absolute_effect": 0.0367,
            "label_mapping_swap": {"status": "disagree"},
            "per_verdict_effect": {"vulnerable": {"cases": 6, "mean_margin_delta": 0.85}},
            "supported_by_controls": supported,
            "conclusion": "exploratory localization, not a causal claim"}


def test_causal_summary_keeps_the_control_verdict(tmp_path):
    summary = R._causal_summary(_component())
    assert summary["component"] == "attn_out-L18"
    assert summary["sign_flip_p_value"] == 0.03
    assert summary["family_ci95"] == [0.13, 0.62]
    # A component whose controls failed must arrive labelled exploratory, never silently promoted.
    assert summary["supported_by_controls"] is False
    assert summary["conclusion"].startswith("exploratory")


def test_attention_from_localization_reads_ranked_heads(tmp_path):
    head = {"layer": 18, "head": 8, "cases_used": 8, "mean_opinion_mass": 0.5,
            "mean_relative_opinion_mass": 0.42, "relative_test": {"p_value": 0.008},
            "relative_interval": {"ci95": [0.36, 0.47]}, "mean_source_mass_shift": 0.007,
            "source_shift_test": {"p_value": 0.068}}
    path = _localization_dir(tmp_path, [_component()], [head])
    entries, source = R._attention_from_localization(path.parent / "attention.json")
    assert source["kind"] == "localization" and source["forward_units"] == 0
    assert entries[0]["head"] == 8 and entries[0]["relative_p_value"] == 0.008
    assert entries[0]["relative_ci95"] == [0.36, 0.47]


def test_attention_from_localization_handles_no_heads(tmp_path):
    path = _localization_dir(tmp_path, [], [])
    entries, source = R._attention_from_localization(path.parent / "attention.json")
    assert entries == [] and source["kind"] == "unavailable"


def test_dump_carries_localization_causal_and_attention(tmp_path):
    path = _localization_dir(tmp_path, [_component()], [
        {"layer": 18, "head": 8, "cases_used": 8, "mean_relative_opinion_mass": 0.42,
         "relative_test": {"p_value": 0.008}, "relative_interval": {"ci95": [0.36, 0.47]}}])
    result = dump(tmp_path, "--localization", str(path))
    directory = workbench(tmp_path)
    causal = json.loads((directory / "causal.json").read_text())
    attention = json.loads((directory / "attention.json").read_text())
    assert causal["source"]["kind"] == "localization" and causal["entries"][0]["layer"] == 18
    assert causal["source"]["supported_count"] == 0
    # The regression this guards: without --attention-layers the table used to be empty even
    # though a finished localization was handed in.
    assert result["attention_source"]["kind"] == "localization"
    assert attention["entries"] and attention["entries"][0]["head"] == 8
