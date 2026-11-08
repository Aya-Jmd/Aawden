"""Small genuine PyTorch/safetensors checks, not claims of Aya parity."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch", reason="Optional tensor runtime absent; no real-weight claim")
from security_wsp.interventions import project_output_tensor, export_hf_projection, tensor_hash
from security_wsp.models import HuggingFaceAdapter, cached_weight_directory, weight_file_hashes
from security_wsp.artifacts import atomic_json, file_hash
from security_wsp.workflow import temporary_projection


def test_projection_removes_only_the_specified_output_direction_and_preserves_zero_edit():
    weight = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    d = torch.tensor([1., 0.])
    edited = project_output_tensor(weight, d, 1)
    assert torch.equal(edited[0], torch.zeros(3))
    assert torch.equal(edited[1], weight[1])
    assert torch.equal(weight, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    assert torch.equal(project_output_tensor(weight.bfloat16(), d, 0), weight.bfloat16())
    with pytest.raises(ValueError, match="matching"):
        project_output_tensor(weight, torch.ones(3), 1)


def test_sharded_weight_hashes_do_not_require_a_complete_hub_snapshot(tmp_path, monkeypatch):
    constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    revision = "a" * 40
    checkpoint = tmp_path / "models--example--checkpoint" / "snapshots" / revision
    index = checkpoint / "model.safetensors.index.json"
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    atomic_json(index, {"weight_map": {"first": shards[0], "second": shards[1]}})
    for name in shards:
        (checkpoint / name).write_bytes(name.encode())

    loaded = cached_weight_directory("example/checkpoint", revision)
    assert weight_file_hashes(loaded) == {name: file_hash(checkpoint / name) for name in [index.name, *shards]}


def small_adapter(fail_save=False):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].mlp = torch.nn.Module()
            self.model.layers[0].mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
        def save_pretrained(self, directory, **kwargs):
            if fail_save:
                raise RuntimeError("simulated interrupted export")
            from safetensors.torch import save_file
            save_file(self.state_dict(), str(directory / "model.safetensors"))
    class Tokenizer:
        def save_pretrained(self, directory):
            (directory / "tokenizer.json").write_text("{}")
    return SimpleNamespace(model=Model(), tokenizer=Tokenizer(), metadata={"model_id": "unit-test", "revision": "immutable"})


def test_serialized_tensor_hashes_and_source_restore_even_when_export_fails(tmp_path):
    pytest.importorskip("safetensors")
    adapter = small_adapter()
    name = "model.layers.0.mlp.down_proj.weight"
    parameter = dict(adapter.model.named_parameters())[name]
    original = tensor_hash(parameter)
    report = export_hf_projection(adapter, [name], [1, 0], .5, tmp_path / "checkpoint", "discovery-hash")
    assert tensor_hash(parameter) == original
    assert report["tensors"][0]["changed_entries"] > 0
    assert report["runtime_hooks_required"] is False
    failing = small_adapter(fail_save=True)
    original = tensor_hash(dict(failing.model.named_parameters())[name])
    with pytest.raises(RuntimeError, match="interrupted"):
        export_hf_projection(failing, [name], [1, 0], 1, tmp_path / "failed", "discovery-hash")
    assert tensor_hash(dict(failing.model.named_parameters())[name]) == original


def test_in_memory_trial_restores_on_failure():
    adapter = small_adapter()
    name = "model.layers.0.mlp.down_proj.weight"
    original = tensor_hash(dict(adapter.model.named_parameters())[name])
    with pytest.raises(RuntimeError):
        with temporary_projection(adapter, name, [1, 0], 1):
            raise RuntimeError("trial failed")
    assert tensor_hash(dict(adapter.model.named_parameters())[name]) == original


def test_continuation_scoring_uses_the_second_token_of_shared_prefix_labels():
    adapter = HuggingFaceAdapter.__new__(HuggingFaceAdapter)
    adapter.torch, adapter.device, adapter.config = torch, "cpu", {"context_limit": 30}
    alphabet = {"P": 0, "a": 1, "b": 2, "c": 3}
    adapter.tokenizer = lambda text, **kwargs: {"input_ids": torch.tensor([[alphabet[x] for x in text]])}
    class Model:
        config = SimpleNamespace(max_position_embeddings=30)
        def __call__(self, input_ids):
            logits = torch.zeros((1, input_ids.shape[1], 4))
            logits[:, 0, 1] = 2  # Both candidate labels start with 'a'.
            logits[:, 1:, 3] = 4  # Only the second continuation token differs.
            return SimpleNamespace(logits=logits)
    adapter.model = Model()
    b, c = adapter.score_continuations("P", ["ab", "ac"])
    assert c > b + 3.9
    assert adapter.score_continuations("P", ["a"])[0] > c
    adapter.config["context_limit"] = 2
    with pytest.raises(ValueError, match="truncation"):
        adapter.score_continuations("P", ["ac"])

def test_lora_starts_as_noop_and_keeps_base_weights_frozen():
    import torch
    from security_wsp.adaptation import LowRankLinear
    base = torch.nn.Linear(8, 8, bias=False)
    base.requires_grad_(False)
    layer = LowRankLinear(base)
    inputs = torch.randn(2, 8)
    assert torch.equal(layer(inputs), base(inputs))
    layer(inputs).square().mean().backward()
    assert base.weight.grad is None
    assert layer.b.grad.abs().sum().item() > 0


def test_lora_loader_accepts_checkpoint_paths_from_json(tmp_path):
    import copy
    import torch
    from safetensors.torch import save_file
    from security_wsp.adaptation import attach, load_adapter
    from security_wsp.artifacts import file_hash
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.q_proj = torch.nn.Linear(8, 8, bias=False)
    model.layer.v_proj = torch.nn.Linear(8, 8, bias=False)
    restored = copy.deepcopy(model)
    parameters = attach(model)
    checkpoint = tmp_path / "adapter.safetensors"
    save_file({name: value.detach() for name, value in parameters.items()}, checkpoint)
    specification = {"checkpoint": str(checkpoint), "rank": 8, "sha256": file_hash(checkpoint),
                     "base_checkpoint_hashes": {"base": "fixed"}}
    load_adapter(restored, specification, {"base": "fixed"})
    actual = dict(restored.named_parameters())
    assert all(torch.equal(actual[name], value) for name, value in parameters.items())
    assert all(not parameter.requires_grad for parameter in actual.values())


def test_disabling_adapter_restores_flags_and_weights_after_failure():
    import torch
    import pytest
    from security_wsp.adaptation import LowRankLinear, without_adapter
    model = torch.nn.Sequential(LowRankLinear(torch.nn.Linear(8, 8)))
    saved = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(RuntimeError, match="fixture"):
        with without_adapter(model):
            assert model[0].enabled is False
            raise RuntimeError("fixture")
    assert model[0].enabled is True
    assert all(torch.equal(value, saved[name]) for name, value in model.named_parameters())


def test_return_estimate_cancels_option_letter_bias():
    from security_wsp.mental_review import score_simulation
    from security_wsp.adaptation import LowRankLinear
    from security_wsp.data import build_cases
    model = torch.nn.Sequential(LowRankLinear(torch.nn.Linear(8, 8)))
    class Scorer:
        def __init__(self):
            self.model = model
        def chat_prefix(self, messages):
            return messages[-1]["content"]
        def _tokens(self, text):
            return torch.zeros((1, 10), dtype=torch.long)
        def score_continuations(self, prefix, answers):
            assert not model[0].enabled
            assert answers == (" A", " B")
            # Strong A preference in both orders, plus a smaller semantic preference for None.
            return (-1., -6.) if "A = None" in prefix else (-3., -4.)
    case = next(case for case in build_cases() if case.target_category == "access_control")
    result = score_simulation(case, Scorer())
    assert result["none_minus_record_margin"] == 2
    assert "estimate: None" in result["working_note"]
    assert result["forward_calls"] == 4
    assert result["output_tokens"] == 0
    assert model[0].enabled

