"""merge_dense folds LoRA updates into base Linear weights without changing outputs."""
import pytest

torch = pytest.importorskip("torch", reason="Optional tensor runtime absent; no real-weight claim")
from security_wsp.adaptation import LowRankLinear, attach, merge_dense, without_adapter


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 32, bias=True)
        self.v_proj = torch.nn.Linear(16, 32, bias=False)
        self.out_proj = torch.nn.Linear(32, 16)  # not a LoRA target


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.head = torch.nn.Linear(16, 4)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer.out_proj(torch.relu(layer.q_proj(x) + layer.v_proj(x)))
        return self.head(x)


def build_adapted(seed, bfloat16=False):
    torch.manual_seed(seed)
    model = Tiny()
    attach(model, rank=4, targets=("q_proj", "v_proj"), alpha=12.0)  # scale = 12 / 4 = 3.0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LowRankLinear):
                module.b.normal_(0.0, 0.1)  # non-zero B so the update is real
    if bfloat16:
        # Cast the frozen Linear layers (base weights and head); LoRA A/B stay float32 as attach() creates them.
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)
    return model


def test_merge_matches_adapted_outputs_in_float32():
    model = build_adapted(0)
    x = torch.randn(5, 16)
    lora_layer = model.layers[0].q_proj
    assert isinstance(lora_layer, LowRankLinear) and lora_layer.scale == 3.0
    expected_weight = (lora_layer.base.weight + lora_layer.scale * (lora_layer.b @ lora_layer.a)).detach().clone()
    with torch.no_grad():
        adapted = model(x)
        with without_adapter(model):
            base_only = model(x)
    assert (adapted - base_only).abs().max() > 1e-3  # the adapter must change the output for this test to mean anything

    summary = merge_dense(model)

    assert summary["modules_merged"] == 4
    assert sorted(summary["names"]) == ["layers.0.q_proj", "layers.0.v_proj", "layers.1.q_proj", "layers.1.v_proj"]
    assert summary["disabled_without_update"] == []
    assert not any(isinstance(module, LowRankLinear) for module in model.modules())
    assert type(model.layers[0].q_proj) is torch.nn.Linear
    assert torch.allclose(model.layers[0].q_proj.weight, expected_weight, atol=1e-6)
    with torch.no_grad():
        merged = model(x)
    assert torch.allclose(adapted, merged, atol=1e-5)
    assert (adapted - merged).abs().max().item() < 1e-5


def test_disabled_adapter_is_removed_without_adding_its_update():
    model = build_adapted(2)
    model.layers[1].v_proj.enabled = False
    base_weight = model.layers[1].v_proj.base.weight.detach().clone()
    x = torch.randn(3, 16)
    with torch.no_grad():
        reference = model(x)  # only layers.1.v_proj is disabled; the other adapters stay active
    summary = merge_dense(model)
    assert summary["disabled_without_update"] == ["layers.1.v_proj"]
    assert torch.equal(model.layers[1].v_proj.weight, base_weight)
    with torch.no_grad():
        assert torch.allclose(model(x), reference, atol=1e-5)


def test_merge_bfloat16_within_looser_tolerance():
    model = build_adapted(1, bfloat16=True)
    x = torch.randn(5, 16).to(torch.bfloat16)
    with torch.no_grad():
        adapted = model(x)

    summary = merge_dense(model)

    assert summary["modules_merged"] == 4
    assert not any(isinstance(module, LowRankLinear) for module in model.modules())
    assert model.layers[0].q_proj.weight.dtype == torch.bfloat16
    with torch.no_grad():
        merged = model(x)
    assert merged.dtype == adapted.dtype == torch.bfloat16
    # bf16 keeps about three significant digits, so the merged weight rounds differently.
    assert torch.allclose(adapted.float(), merged.float(), atol=5e-2, rtol=5e-2)
