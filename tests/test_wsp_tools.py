"""Offline tests for security_wsp.wsp_tools on a tiny random Cohere model. No GPU, no real checkpoint."""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from security_wsp.data import CLEAN, VULNERABLE, load_dataset  # noqa: E402
from security_wsp.interventions import tensor_hash  # noqa: E402
from security_wsp.patches import apply_patch, build_patch, revert_patch  # noqa: E402
from security_wsp.prompts import CONDITIONS, render  # noqa: E402
from security_wsp.wsp_tools import BudgetExhausted, Edit, WspTools  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITES = [
    "model.layers.*.mlp.down_proj.weight",
    "model.layers.*.self_attn.o_proj.weight",
    "model.layers.*.mlp.up_proj.weight",
    "model.layers.*.mlp.gate_proj.weight",
]
WORD = re.compile(r"\w+|[^\w\s]+")
TEMPLATE = (
    "{% for message in messages %}<{{ message['role'] }}> {{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<assistant> {% endif %}"
)


def discovery_cases():
    """Two vulnerable and two clean discovery cases from the committed discovery dataset."""
    cases = load_dataset(ROOT / "datasets" / "wsp-discovery.json")
    vulnerable = [c for c in cases if c.oracle_verdict == VULNERABLE][:2]
    clean = [c for c in cases if c.oracle_verdict == CLEAN][:2]
    assert len(vulnerable) == 2 and len(clean) == 2
    return vulnerable + clean


def build_tokenizer(cases):
    corpus = []
    for case in cases:
        for condition in CONDITIONS:
            corpus.extend(m["content"] for m in render(case, condition, prefill=True).messages)
    corpus.extend(["system", "user", "assistant", "<", ">"])
    words = sorted({w for text in corpus for w in WORD.findall(text)})
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    for word in words:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()  # same regex as WORD: context-free, so spans match
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                   eos_token="[EOS]", chat_template=TEMPLATE)


def randomize_norms(model, seed=1):
    """Make a stock random Cohere model input-sensitive.

    Cohere zero-initializes the residual output projections (o_proj, down_proj), so those outputs are
    exactly zero, and it zero-initializes the LayerNorm shifts while leaving the weights uninitialized
    (uninitialized memory can be all zeros). Either makes activation contrasts degenerate. Re-drawing
    those weights and LayerNorm weights keeps the model random but responsive to the prompt.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.5 * torch.randn(param.shape, generator=generator))
            elif name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight")):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)


@pytest.fixture()
def tiny():
    cases = discovery_cases()
    tokenizer = build_tokenizer(cases)
    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "cohere", vocab_size=len(tokenizer), hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.unk_token_id)
    model = AutoModelForCausalLM.from_config(config).eval()
    randomize_norms(model)
    return model, tokenizer, cases


def make_tools(model, tokenizer, cases, **overrides):
    params = dict(device="cpu", site_patterns=SITES, max_changed_fraction=0.01, forward_pass_budget=100,
                  batch_size=16, max_new_tokens=6, context_limit=4096)
    params.update(overrides)
    return WspTools(model, tokenizer, cases, **params)


def snapshot(model):
    return {name: tensor_hash(p) for name, p in model.named_parameters()}


def sample_edit(model, seed=3, name="model.layers.0.mlp.down_proj.weight", rows=(0, 5)):
    param = dict(model.named_parameters())[name]
    generator = torch.Generator().manual_seed(seed)
    values = (torch.randn((len(rows), param.shape[1]), generator=generator) * 0.5).tolist()
    return Edit(tensor_name=name, axis=0, indices=list(rows), values=values)


def test_refuses_validation_and_final_cases(tiny):
    model, tokenizer, cases = tiny
    for split in ("wsp_validation", "wsp_final"):
        with pytest.raises(ValueError, match=split):
            make_tools(model, tokenizer, [replace(cases[0], split=split)])
    with pytest.raises(ValueError):
        make_tools(model, tokenizer, cases, allowed_splits=("wsp_discovery", "wsp_final"))
    with pytest.raises(ValueError):
        make_tools(model, tokenizer, cases, allowed_splits=("wsp_validation",))
    with pytest.raises(ValueError, match="not in"):
        make_tools(model, tokenizer, [replace(cases[0], split="wsp_other")])


def test_list_sites_reports_only_editable_sites(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    sites = tools.list_sites()
    names = {s["tensor_name"] for s in sites}
    expected = {f"model.layers.{i}.{k}" for i in range(2)
                for k in ("mlp.down_proj.weight", "self_attn.o_proj.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight")}
    assert names == expected
    assert not any("embed" in n or "norm" in n or "q_proj" in n for n in names)
    down = next(s for s in sites if s["tensor_name"] == "model.layers.0.mlp.down_proj.weight")
    assert down["shape"] == [64, 160] and down["dtype"] == "torch.float32"
    assert set(down["stats"]) == {"mean", "std", "min", "max"}
    assert len(down["top_columns"]) == 5
    norms = [c["norm"] for c in down["top_columns"]]
    assert norms == sorted(norms, reverse=True)
    assert all(0 <= c["column"] < 160 for c in down["top_columns"])
    assert [s["tensor_name"] for s in tools.list_sites("layers.1.mlp.down")] == ["model.layers.1.mlp.down_proj.weight"]
    assert tools.list_sites() == tools.list_sites()  # deterministic


def test_read_weights_returns_exact_values_and_caps(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    name = "model.layers.0.mlp.down_proj.weight"
    param = dict(model.named_parameters())[name]

    rows = tools.read_weights(name, axis=0, indices=[0, 3])
    assert rows["values_returned"] is True and rows["slice_meaning"] == "row"
    assert rows["slices"] == param.detach()[[0, 3]].tolist()

    cols = tools.read_weights(name, axis=1, indices=[7])
    assert cols["slice_length"] == 64 and cols["slices"][0] == param.detach()[:, 7].tolist()

    whole = tools.read_weights(name, axis=0)  # 64 * 160 = 10240 scalars, above the 4096 cap
    assert whole["values_returned"] is False and "slices" not in whole
    assert set(whole["stats"]) == {"mean", "std", "min", "max"}

    with pytest.raises(PermissionError):
        tools.read_weights("model.embed_tokens.weight", axis=0, indices=[0])
    with pytest.raises(ValueError):
        tools.read_weights(name, axis=0, indices=[64])


def test_activation_contrast_is_finite_and_deterministic(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    for site in ("resid_post", "mlp_out", "attn_out"):
        first = tools.activation_contrast(site, 0)
        assert first["prompt_count"] == 2 * len(cases) and first["forward_units"] == 1
        assert 0.0 <= first["separation"]["p_value"] <= 1.0
        if not first["degenerate"]:
            assert first["separation"]["permutations"] == tools.permutations
        assert math.isfinite(first["mean_difference_norm"])
        if not first["degenerate"]:
            assert len(first["direction"]) == 64
            assert math.isclose(sum(x * x for x in first["direction"]), 1.0, rel_tol=1e-6)
            assert all(math.isfinite(v) for v in (first["cosine"]["pairwise_mean"], first["cosine"]["pairwise_min"]))
            assert first["random_baseline"]["directions"] == 32
            assert math.isfinite(first["random_baseline"]["mean_separation"])
        assert tools.activation_contrast(site, 0) == first  # seeded permutation test is reproducible
    assert tools.activation_contrast("hook_resid_post", 1)["site"] == "resid_post"
    with pytest.raises(ValueError):
        tools.activation_contrast("resid_post", 2)


def test_activation_contrast_direction_is_not_degenerate_on_the_tiny_model(tiny):
    model, tokenizer, cases = tiny
    report = make_tools(model, tokenizer, cases).activation_contrast("resid_post", 0)
    assert report["degenerate"] is False and report["direction"] is not None
    assert report["mean_difference_norm"] > 0
    assert math.isclose(sum(x * x for x in report["direction"]), 1.0, rel_tol=1e-6)
    assert -1.0 <= report["cosine"]["cos_to_mean_direction_mean"] <= 1.0
    assert report["random_baseline"]["max_separation"] >= report["random_baseline"]["mean_separation"]


def test_activation_contrast_reports_a_degenerate_pairing(tiny, monkeypatch):
    """Zero MLP and attention contributions leave the residual input-independent: the report must say so
    instead of inventing a direction."""
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    layer = model.model.layers[0]
    zeros = lambda *a, **k: torch.zeros(1, 1, model.config.hidden_size)
    zeros_attn = lambda *a, **k: (torch.zeros(1, 1, model.config.hidden_size), None)
    # Patch the classes: an instance attribute would not be bound as a method.
    monkeypatch.setattr(type(layer.mlp), "forward", zeros)          # mlp returns a tensor
    monkeypatch.setattr(type(layer.self_attn), "forward", zeros_attn)  # attn returns (tensor, weights)
    report = tools.activation_contrast("resid_post", 0)
    assert report["degenerate"] is True and report["direction"] is None
    assert report["separation"] == {"statistic": 0.0, "p_value": 1.0, "permutations": 0}


def test_attention_to_span_reports_or_matches(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    results = tools.attention_to_span(0)
    assert len(results) == 4
    for entry in results:
        assert entry["status"] in ("ok", "no_usable_cases")
        assert "match_failures" in entry
        if entry["status"] == "ok":
            n = entry["neutral"]
            assert math.isclose(n["source_mass"] + n["other_mass"], 1.0, abs_tol=1e-4)
            m = entry["misleading"]
            assert math.isclose(m["opinion_mass"] + m["source_mass"] + m["other_mass"], 1.0, abs_tol=1e-4)
    assert [r["head"] for r in tools.attention_to_span(1, head=2)] == [2]
    with pytest.raises(ValueError):
        tools.attention_to_span(0, head=9)


def test_span_locator_refuses_ambiguous_matches(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    ids = tokenizer("return conn", add_special_tokens=False)["input_ids"]
    assert tools._locate(ids + ids, "return conn", "opinion") == (None, "opinion: 2 matches in prefix")
    assert tools._locate(ids, "return conn", "opinion")[0] == list(range(len(ids)))
    assert tools._locate(ids, "unrelated tokens here", "opinion") == (None, "opinion: no match in prefix")


def test_evaluate_edits_applies_and_restores_exactly(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    before = snapshot(model)
    edit = sample_edit(model)
    result = tools.evaluate_edits([edit])
    assert snapshot(model) == before

    summary = result["edit_sets"][0]
    name = edit.tensor_name
    assert summary["restoration"]["verified"] is True
    assert summary["restoration"]["tensor_sha256_before"][name] == before[name]
    assert summary["restoration"]["tensor_sha256_after_apply"][name] != before[name]
    assert summary["changed_scalars"] == len(edit.indices) * 160
    assert summary["touched_scalars"] == summary["changed_scalars"]
    assert len(summary["rows"]) == len(cases) * len(CONDITIONS)
    for condition in CONDITIONS:
        score = summary["balanced_accuracy"][condition]
        assert score is None or 0.0 <= score <= 1.0
        counts = summary["per_class_counts"][condition]
        assert set(counts) >= {"true_positive", "false_negative", "invalid_count"}
    assert all(row["parse_error"] is not None or row["predicted_verdict"] in (VULNERABLE, CLEAN, "insufficient_information")
               for row in summary["rows"])
    assert tools.budget_used == 1 and result["forward_units_per_set"] == 1


def test_batched_evaluation_matches_sequential(tiny):
    model, tokenizer, cases = tiny
    edit = sample_edit(model)
    batched = make_tools(model, tokenizer, cases, batch_size=16).evaluate_edits([edit])["edit_sets"][0]
    sequential = make_tools(model, tokenizer, cases, batch_size=1, forward_pass_budget=1000).evaluate_edits([edit])
    seq_rows = sequential["edit_sets"][0]["rows"]
    assert [r["response_text"] for r in batched["rows"]] == [r["response_text"] for r in seq_rows]
    assert sequential["forward_units_per_set"] == len(cases) * len(CONDITIONS)


def test_budget_exhaustion_raises_before_any_write(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases, forward_pass_budget=2)
    tools.activation_contrast("resid_post", 0)  # one unit
    tools.evaluate_edits([sample_edit(model)])  # one unit
    assert tools.budget_used == 2 and tools.budget_remaining == 0
    before = snapshot(model)
    with pytest.raises(BudgetExhausted):
        tools.evaluate_edits([sample_edit(model)])
    assert snapshot(model) == before
    assert tools.budget_used == 2
    assert tools.transcript()[-1]["status"] == "error"


def test_oversized_edit_is_refused_without_consuming_budget(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases, max_changed_fraction=1e-4)
    before = snapshot(model)
    with pytest.raises(ValueError, match="limit"):
        tools.evaluate_edits([sample_edit(model)])
    assert snapshot(model) == before and tools.budget_used == 0


def test_non_editable_and_out_of_range_edits_refused(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    with pytest.raises(PermissionError):
        tools.evaluate_edits([Edit("model.embed_tokens.weight", 0, [0], [[0.0] * 64])])
    with pytest.raises(ValueError, match="within"):
        tools.evaluate_edits([sample_edit(model, rows=(0, 64))])


def test_context_limit_refuses_instead_of_truncating(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases, context_limit=20)
    with pytest.raises(ValueError, match="truncation"):
        tools.evaluate_edits([sample_edit(model)])
    assert tools.budget_used == 0


def test_random_control_matches_update_norm(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    edit = sample_edit(model)
    control = tools.propose_random_control([edit], seed=17)
    assert control["within_1e-3"] is True and control["relative_norm_mismatch"] <= 1e-3
    assert control["tensor_names"] == [edit.tensor_name]
    assert control["changed_slices"] == {edit.tensor_name: len(edit.indices)}
    assert math.isclose(control["achieved_update_norm"], control["target_update_norm"], rel_tol=1e-3)
    assert [e.indices for e in control["edits"]] == [sorted(e.indices) for e in control["edits"]]
    assert tools.propose_random_control([edit], seed=17)["edits"][0].values == control["edits"][0].values
    assert tools.propose_random_control([edit], seed=29)["edits"][0].values != control["edits"][0].values


def test_random_control_handles_column_patches(tiny):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    name = "model.layers.1.mlp.up_proj.weight"  # shape (160, 64): column slices have length 160
    original = dict(model.named_parameters())[name].detach().clone()
    edited = original.clone()
    edited[:, [2, 9]] += 0.25
    patch = build_patch({name: original}, {name: edited})
    assert patch.tensors[0].axis == 1
    expected = float((edited - original).norm())
    control = tools.propose_random_control(patch, seed=43)
    assert math.isclose(control["target_update_norm"], expected, rel_tol=1e-5)
    assert control["changed_slices"] == {name: 2} and control["within_1e-3"] is True


def test_submit_enforces_limits_and_persists(tiny, tmp_path):
    model, tokenizer, cases = tiny
    out = tmp_path / "out"
    tools = make_tools(model, tokenizer, cases, output_dir=out, max_submissions=2)
    base = snapshot(model)
    first = tools.submit([sample_edit(model, seed=3)])
    second = tools.submit([sample_edit(model, seed=4)])
    assert first["submission_index"] == 1 and second["submission_index"] == 2
    assert snapshot(model) == base  # submit never leaves the edit applied
    with pytest.raises(ValueError, match="submission limit"):
        tools.submit([sample_edit(model, seed=5)])

    manifest = json.loads((out / "submission-01.json").read_text())
    assert manifest["format"] == "aawden-patch-v1" and manifest["changed_scalars"] == 2 * 160
    entry = manifest["tensors"][0]
    apply_patch(model, out / "submission-01")
    assert tensor_hash(dict(model.named_parameters())[entry["name"]]) == entry["patched_sha256"]
    revert_patch(model, out / "submission-01")
    assert snapshot(model) == base


def test_submit_refuses_oversized_edit_and_writes_nothing(tiny, tmp_path):
    model, tokenizer, cases = tiny
    out = tmp_path / "out"
    tools = make_tools(model, tokenizer, cases, output_dir=out, max_changed_fraction=1e-4)
    with pytest.raises(ValueError, match="limit"):
        tools.submit([sample_edit(model)])
    assert not list(out.glob("submission-*")) if out.exists() else True


def test_transcript_order_costs_and_errors(tiny, tmp_path):
    model, tokenizer, cases = tiny
    tools = make_tools(model, tokenizer, cases)
    tools.list_sites()
    tools.read_weights("model.layers.0.mlp.down_proj.weight", axis=0, indices=[1])
    with pytest.raises(PermissionError):
        tools.read_weights("model.embed_tokens.weight", axis=0, indices=[0])
    tools.activation_contrast("mlp_out", 1)

    log = tools.transcript()
    assert [e["seq"] for e in log] == [1, 2, 3, 4]
    assert [e["tool"] for e in log] == ["list_sites", "read_weights", "read_weights", "activation_contrast"]
    assert [e["status"] for e in log] == ["ok", "ok", "error", "ok"]
    assert "PermissionError" in log[2]["error"]
    assert log[3]["forward_units"] == 1 and log[3]["model_prompt_tokens"] > 0
    assert all(e["estimated_tokens"]["input"] >= 1 and e["estimated_tokens"]["output"] >= 1 for e in log)
    assert tools.calls == 4

    path = tools.save_transcript(tmp_path / "transcript.json")
    saved = json.loads(path.read_text())
    assert saved["budget"]["used"] == 1 and len(saved["entries"]) == 4
    assert saved["entries"] == log


def test_validation_and_final_cannot_be_reached_through_tools(tiny):
    model, tokenizer, cases = tiny
    with pytest.raises(ValueError):
        make_tools(model, tokenizer, cases + [replace(cases[0], case_id="x", split="wsp_final")])
    with pytest.raises(ValueError):
        make_tools(model, tokenizer, cases + [replace(cases[0], case_id="y", split="wsp_validation")])
