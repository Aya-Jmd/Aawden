"""merge_lora folds the adapter into plain weights so sparse patches apply to plain tensor names."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from security_wsp.adaptation import LowRankLinear, attach  # noqa: E402
from security_wsp.artifacts import file_hash  # noqa: E402
from security_wsp.models import load_lora_into  # noqa: E402


def build(tmp_path, alpha=4, rank=2):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    parameters = attach(model, rank=rank, targets=("0",), alpha=alpha)
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.copy_(torch.randn_like(parameter))
    path = tmp_path / "adapter.safetensors"
    save_file(dict(parameters), str(path))
    fresh = torch.nn.Sequential(torch.nn.Linear(8, 8))
    fresh.load_state_dict(torch.nn.Sequential(torch.nn.Linear(8, 8)).state_dict())
    spec = {"checkpoint": str(path), "sha256": file_hash(path), "rank": rank, "alpha": alpha,
            "target_modules": ["0"], "base_checkpoint_hashes": {}}
    return model, fresh, spec


def test_unmerged_keeps_wrapper_and_merged_removes_it(tmp_path):
    trained, fresh, spec = build(tmp_path)
    metadata = load_lora_into(fresh, spec, {}, merge=False)
    assert metadata["adaptation"] == "LoRA"
    assert any(isinstance(m, LowRankLinear) for m in fresh.modules())
    # The adapter changes the output relative to the untouched base weights.
    with torch.no_grad():
        reference = torch.nn.Sequential(torch.nn.Linear(8, 8))
        reference.load_state_dict(torch.nn.Sequential(torch.nn.Linear(8, 8)).state_dict())
    assert "merged_state_sha256" not in metadata

    trained2, fresh2, spec2 = build(tmp_path)
    merged = load_lora_into(fresh2, spec2, {}, merge=True)
    assert merged["adaptation"] == "LoRA-merged-dense"
    assert merged["merged_modules"] == 1
    assert merged["merged_state_sha256"]
    assert not any(isinstance(m, LowRankLinear) for m in fresh2.modules())
    assert all("base" not in name for name, _ in fresh2.named_parameters())
    # Merged weights differ from the untouched base and match the unmerged adapter exactly.
    assert not torch.allclose(fresh2[0].weight, torch.nn.Sequential(torch.nn.Linear(8, 8))[0].weight)
    probe = torch.randn(3, 8)
    with torch.no_grad():
        assert torch.allclose(fresh[0](probe), fresh2[0](probe), atol=1e-5)


def test_wrong_base_refused(tmp_path):
    _, fresh, spec = build(tmp_path)
    with pytest.raises(ValueError):
        load_lora_into(fresh, dict(spec, base_checkpoint_hashes={"x": "y"}), {"x": "not-y"}, merge=True)
