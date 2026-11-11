"""Tests for the fix arms: prompt identity, leakage guards, and the direction-edit maths."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_antidef_rows_match_evaluator_prompt(tmp_path):
    """Training prompt == the bytes wsp_cross_model.render_for scores, before the chat template."""
    from security_wsp.prompts import PREFILL, render
    make = _load("_antidef", "scripts/antidef_make_data.py")
    out = tmp_path / "rows.jsonl"
    make.main(["--output", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    cases = {c.case_id: c for c in make.load_cases(ROOT / "datasets" / "distill-review-train.json")}
    for row in rows[:9]:
        cid, condition = row["key"].split(":")
        req = render(cases[cid], condition, wording="reviewer")
        assert row["system"] == req.messages[0]["content"]
        assert row["prompt"] == req.messages[1]["content"]
        assert row["prefill"] == PREFILL
        assert row["completion"] == cases[cid].oracle_verdict
    assert len(rows) == 3 * len(cases)


def test_antidef_refuses_eval_and_sealed_files(tmp_path):
    make = _load("_antidef2", "scripts/antidef_make_data.py")
    for name in ("plain-pooled.json", "wsp-final.json"):
        with pytest.raises(SystemExit):
            make.main(["--dataset", str(ROOT / "datasets" / name), "--output", str(tmp_path / "x")])


def test_training_families_disjoint_from_evaluation():
    make = _load("_antidef3", "scripts/antidef_make_data.py")
    train = {c.family_id for c in make.load_cases(ROOT / "datasets" / "distill-review-train.json")}
    for name in ("plain-pooled.json", "wsp-discovery.json"):
        fams = {c["family_id"] for c in json.loads((ROOT / "datasets" / name).read_text())["cases"]}
        assert not train & fams


torch = pytest.importorskip("torch")


def test_guard_and_ablate_maths():
    d = _load("_dir", "scripts/wsp_direction_edit.py")
    g = torch.Generator().manual_seed(0)
    opinion, code = torch.randn(64, generator=g), torch.randn(64, generator=g)
    r = d.guard(opinion, code)
    assert abs(float(r.norm()) - 1) < 1e-5
    assert abs(float(r @ d.unit(code))) < 1e-5          # guard removes the code-truth component
    h = torch.randn(5, 64, generator=g)
    out = d.ablate(h, r)
    assert torch.allclose(out @ r, torch.zeros(5), atol=1e-5)   # direction removed
    # the code-truth component of h is untouched by ablating a direction orthogonal to it
    assert torch.allclose(out @ d.unit(code), h @ d.unit(code), atol=1e-4)


def test_output_hook_equals_weight_orthogonalisation():
    """Removing r from a linear map's output == orthogonalising its weight; the arm relies on this."""
    d = _load("_dir2", "scripts/wsp_direction_edit.py")
    g = torch.Generator().manual_seed(1)
    w, x = torch.randn(16, 8, generator=g), torch.randn(3, 8, generator=g)
    r = d.unit(torch.randn(16, generator=g))
    hooked = d.ablate(x @ w.T, r)
    w_edit = w - r[:, None] * (r @ w)[None, :]
    assert torch.allclose(hooked, x @ w_edit.T, atol=1e-5)


def _script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scale_adapter_scales_only_lora_b():
    torch = pytest.importorskip("torch")
    mod = _script("scale_adapter")
    state = {"x.lora_A.weight": torch.ones(2, 3), "x.lora_B.weight": torch.full((3, 2), 2.0)}
    scaled, touched = mod.scale_state(state, 0.25)
    assert touched == 1
    assert torch.equal(scaled["x.lora_A.weight"], state["x.lora_A.weight"])
    assert torch.allclose(scaled["x.lora_B.weight"], torch.full((3, 2), 0.5))
    # B @ A scales by exactly the factor
    assert torch.allclose(scaled["x.lora_B.weight"] @ scaled["x.lora_A.weight"],
                          0.25 * (state["x.lora_B.weight"] @ state["x.lora_A.weight"]))


def test_offdomain_items_are_balanced_and_conditions_differ_only_by_the_opinion_line():
    mod = _script("offdomain_deference")
    rows = [{"id": f"q{i}", "question": f"Q{i}?", "answerKey": "C",
             "choices": {"label": ["A", "B", "C", "D"], "text": [f"w{i}a", f"w{i}b", f"right{i}", f"w{i}d"]}}
            for i in range(40)]
    items = mod.build_items(rows, 20, 17)
    assert items == mod.build_items(rows, 20, 17)
    assert sum(i["key"] == "A" for i in items) == 10
    for item in items:
        assert item[item["key"]].startswith("right")
        neutral = mod.messages(item, "neutral")[1]["content"]
        right = mod.messages(item, "correct_suggestion")[1]["content"]
        wrong = mod.messages(item, "incorrect_suggestion")[1]["content"]
        other = "B" if item["key"] == "A" else "A"
        assert right.replace(f"I think the answer is {item['key']}.\n", "") == neutral
        assert wrong.replace(f"I think the answer is {other}.\n", "") == neutral


def test_round2_influence_and_policy_on_a_toy_table():
    mod = _script("fix_report_r2")
    def block(margins):
        return {"json_prefill": {"verdict_by_continuation": {"v": "vulnerable", "c": "not_vulnerable_for_target"},
                                 "cases": {k: {"oracle_verdict": "vulnerable", "correct_margin": m,
                                               "scores": {"v": m, "c": 0.0}} for k, m in margins.items()}}}
    table = {"conditions": {"neutral": block({"a": 2.0, "b": -1.0}),
                            "correct_suggestion": block({"a": 5.0, "b": 3.0}),
                            "incorrect_suggestion": block({"a": -4.0, "b": -2.0})}}
    inf = mod.influence(table)
    assert inf["correct_reviewer"]["mean"] == 3.5      # (3 + 4) / 2
    assert inf["wrong_reviewer"]["mean"] == 3.5        # (6 + 1) / 2
    pol = mod.policy(table)
    assert pol["follows_reviewer"] == 2 and pol["follows_code"] == 0
    assert pol["accuracy"] == {"no_reviewer": 1, "correct_reviewer": 2, "wrong_reviewer": 0}


def test_multilang_builder_filters():
    mod = _script("build_realcode_multilang")
    assert mod.strip_comments('a = "http://x"; // note\n/* gone */b = 1;\n', "php") == 'a = "http://x";\nb = 1;\n'
    assert mod.strip_comments('q = "#{name}" # note\n', "ruby") == 'q = "#{name}"\n'
    assert mod.has_stray_text('echo "Login failed.";ellow\n')
    assert mod.has_stray_text('println("x")Bs\n')
    assert not mod.has_stray_text('for (int i = 0; i < n; i++) {\n  x(); }\nclass User < Base; end\n')
    bad = '$sql = "SELECT * FROM users WHERE id=" . $_GET["id"];\n'
    good = '$stmt = $conn->prepare("SELECT * FROM users WHERE id=?");\n$stmt->bind_param("i", $id);\n'
    assert mod.sink_ok("sql_injection", "php", bad, good)
    assert not mod.sink_ok("sql_injection", "php", good, bad)
    assert not mod.sink_ok("sql_injection", "php", bad, bad)


def test_prior_answer_rows_hold_both_answers_and_no_reviewer():
    import json as _json
    make = _script("antidef_make_data")
    cases = make.load_cases(ROOT / "datasets" / "distill-review-train.json")[:4]
    for case in cases:
        rows = make.prior_answer_rows(case)
        assert [r["condition"] for r in rows] == ["prior_right", "prior_wrong"]
        priors = [_json.loads(r["messages"][2]["content"])["verdict"] for r in rows]
        assert priors[0] == case.oracle_verdict and priors[1] != case.oracle_verdict
        for r in rows:
            assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant", "user"]
            assert r["completion"] == case.oracle_verdict
            assert "Reviewer" not in "".join(m["content"] for m in r["messages"][1:])


def test_multi_wording_rows(tmp_path):
    import json as _json
    make = _script("antidef_make_data")
    out = tmp_path / "m.jsonl"
    assert make.main(["--wordings", "reviewer,authority,scanner,code_comment", "--prior-answer", "--output", str(out)]) == 0
    rows = [_json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 268 * 11 and len({r["key"] for r in rows}) == len(rows)
    text = "".join(r["prompt"] for r in rows)
    for held_out in ("CISO sign-off", "three reviewers", "checked this file three times", "I traced the untrusted"):
        assert held_out not in text
    assert all("I disagree with your answer" not in _json.dumps(r) for r in rows)
