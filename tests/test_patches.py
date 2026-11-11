"""Sparse weight patches: exact round trip, tamper and wrong-base rejection, manifest accounting."""
import copy

import pytest

torch = pytest.importorskip("torch", reason="Optional tensor runtime absent; no real-weight claim")
from safetensors.torch import load_file, save_file
import security_wsp.patches as patches
from security_wsp.artifacts import atomic_json, file_hash, read_json
from security_wsp.interventions import tensor_hash
from security_wsp.patches import apply_patch, build_patch, revert_patch, save_patch

EXTRA = {"base_model_id": "unit/tiny", "base_revision": "fixture"}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.a = torch.nn.Linear(128, 128, bias=False)
        self.b = torch.nn.Linear(128, 128, bias=False)
        with torch.no_grad():
            self.a.weight.copy_(torch.randn(128, 128, generator=generator))
            self.b.weight.copy_(torch.randn(128, 128, generator=generator))


def state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def sparse_edit(base):
    edited = copy.deepcopy(base)
    with torch.no_grad():
        edited.a.weight[:, [3, 40, 99]] += 0.5  # three full columns of a.weight
        edited.b.weight[[5, 77], :] -= 1.25  # two full rows of b.weight
    return edited


def assert_same(model, expected):
    actual, wanted = state(model), state(expected)
    assert actual.keys() == wanted.keys()
    for name in wanted:
        assert torch.equal(actual[name], wanted[name]), name
        assert tensor_hash(actual[name]) == tensor_hash(wanted[name]), name


def test_round_trip_apply_then_revert_is_exact(tmp_path):
    base = Tiny()
    edited = sparse_edit(base)
    patch = build_patch(state(base), state(edited))
    save_patch(patch, tmp_path / "edit", EXTRA)

    fresh = copy.deepcopy(base)
    apply_patch(fresh, tmp_path / "edit")
    assert_same(fresh, edited)
    assert tensor_hash(fresh.a.weight) == tensor_hash(edited.a.weight)
    assert tensor_hash(fresh.b.weight) == tensor_hash(edited.b.weight)

    revert_patch(fresh, tmp_path / "edit")
    assert_same(fresh, base)


def test_tampered_patch_bytes_are_rejected_before_any_write(tmp_path):
    base = Tiny()
    edited = sparse_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)
    weights = tmp_path / "edit.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0x01
    weights.write_bytes(bytes(data))

    target = copy.deepcopy(base)
    with pytest.raises(ValueError, match="sha256"):
        apply_patch(target, tmp_path / "edit")
    assert_same(target, base)
    with pytest.raises(ValueError, match="sha256"):
        revert_patch(copy.deepcopy(edited), tmp_path / "edit")


def test_wrong_base_is_rejected_and_model_left_unchanged(tmp_path):
    base = Tiny()
    edited = sparse_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)

    wrong = copy.deepcopy(base)
    with torch.no_grad():
        wrong.b.weight[0, 0] += 1e-3  # one unpatched scalar differs: a different base checkpoint
    before = copy.deepcopy(wrong)
    with pytest.raises(ValueError, match="wrong base"):
        apply_patch(wrong, tmp_path / "edit")
    assert_same(wrong, before)

    # Applying twice is also a wrong base: the model no longer matches the recorded original hashes.
    patched = copy.deepcopy(base)
    apply_patch(patched, tmp_path / "edit")
    with pytest.raises(ValueError, match="wrong base"):
        apply_patch(patched, tmp_path / "edit")
    # Reverting an unpatched model is equally refused.
    with pytest.raises(ValueError, match="wrong base"):
        revert_patch(copy.deepcopy(base), tmp_path / "edit")


def test_manifest_counts_axes_and_hashes_are_correct(tmp_path):
    base = Tiny()
    edited = sparse_edit(base)
    manifest = save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)

    assert manifest["format"] == "aawden-patch-v1"
    assert manifest["extra"] == EXTRA
    assert manifest["weights_sha256"] == file_hash(tmp_path / "edit.safetensors")
    assert read_json(tmp_path / "edit.json") == manifest
    by_name = {entry["name"]: entry for entry in manifest["tensors"]}
    assert set(by_name) == {"a.weight", "b.weight"}
    assert by_name["a.weight"]["axis"] == 1 and by_name["a.weight"]["indices_count"] == 3
    assert by_name["a.weight"]["changed_scalars"] == 3 * 128
    assert by_name["b.weight"]["axis"] == 0 and by_name["b.weight"]["indices_count"] == 2
    assert by_name["b.weight"]["changed_scalars"] == 2 * 128
    assert by_name["a.weight"]["shape"] == [128, 128] and by_name["a.weight"]["dtype"] == "torch.float32"
    assert by_name["a.weight"]["original_sha256"] == tensor_hash(base.a.weight)
    assert by_name["a.weight"]["patched_sha256"] == tensor_hash(edited.a.weight)
    assert by_name["b.weight"]["original_sha256"] == tensor_hash(base.b.weight)
    assert by_name["b.weight"]["patched_sha256"] == tensor_hash(edited.b.weight)
    assert manifest["changed_scalars"] == 5 * 128
    assert manifest["total_parameters"] == 2 * 128 * 128
    assert manifest["percent_changed"] == pytest.approx(100 * 640 / 32768)


def test_sparse_patch_is_much_smaller_than_full_state_dict(tmp_path):
    base = Tiny()
    edited = sparse_edit(base)
    save_file(state(edited), str(tmp_path / "full.safetensors"))
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)
    full = (tmp_path / "full.safetensors").stat().st_size
    patch = (tmp_path / "edit.safetensors").stat().st_size
    assert patch < 0.05 * full, (patch, full)


class Square(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(3, 3) / 7)


def square_edit(base):
    edited = copy.deepcopy(base)
    with torch.no_grad():
        edited.w[0] += 1.0  # row 0 changed
        edited.w[2] -= 0.5  # row 2 changed; row 1 untouched
    return edited


def forge(tmp_path, mutate):
    """Edit the stored tensors of a saved patch and re-hash the file, so only semantic checks can catch it."""
    weights = tmp_path / "edit.safetensors"
    stored = load_file(str(weights))
    mutate(stored)
    save_file(stored, str(weights), metadata={"format": "aawden-patch-v1"})
    manifest = read_json(tmp_path / "edit.json")
    manifest["weights_sha256"] = file_hash(weights)
    atomic_json(tmp_path / "edit.json", manifest)


@pytest.mark.parametrize("indices, message", [
    ([0, 999], "out of range"),  # the review case: row 0 is valid and was written before the failure
    ([-1, 2], "out of range"),
    ([2, 0], "strictly increasing"),
    ([0, 0], "strictly increasing"),
])
def test_bad_indices_leave_model_byte_identical(tmp_path, indices, message):
    base = Square()
    edited = square_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)
    forge(tmp_path, lambda stored: stored.__setitem__("w::indices", torch.tensor(indices, dtype=torch.int64)))

    target = copy.deepcopy(base)
    with pytest.raises(ValueError, match=message):
        apply_patch(target, tmp_path / "edit")
    assert_same(target, base)


def test_wrong_value_shape_is_rejected_before_mutation(tmp_path):
    base = Square()
    edited = square_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)
    forge(tmp_path, lambda stored: stored.__setitem__("w::values", stored["w::values"][:, :2].contiguous()))

    target = copy.deepcopy(base)
    with pytest.raises(ValueError, match="shape"):
        apply_patch(target, tmp_path / "edit")
    assert_same(target, base)


def test_values_that_do_not_reproduce_patched_hash_are_rejected_before_mutation(tmp_path):
    base = Square()
    edited = square_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)
    forge(tmp_path, lambda stored: stored.__setitem__("w::values", stored["w::values"] + 1e-3))

    target = copy.deepcopy(base)
    with pytest.raises(ValueError, match="do not reproduce"):
        apply_patch(target, tmp_path / "edit")
    assert_same(target, base)


def test_corrupted_stored_originals_cannot_be_used_to_restore(tmp_path):
    base = Square()
    edited = square_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)
    forge(tmp_path, lambda stored: stored.__setitem__("w::original", stored["w::original"] + 0.25))

    patched = copy.deepcopy(base)
    apply_patch(patched, tmp_path / "edit")
    before = copy.deepcopy(patched)
    with pytest.raises(ValueError, match="do not reproduce"):
        revert_patch(patched, tmp_path / "edit")
    assert_same(patched, before)


def test_post_write_hash_failure_rolls_back_from_snapshots(tmp_path, monkeypatch):
    base = Square()
    edited = square_edit(base)
    save_patch(build_patch(state(base), state(edited)), tmp_path / "edit", EXTRA)

    target = copy.deepcopy(base)
    real_hash = patches.tensor_hash
    param_calls = {"n": 0}

    def flaky_hash(tensor):
        if any(tensor is p for p in target.parameters()):
            param_calls["n"] += 1
            if param_calls["n"] == 2:  # 1st call: pre-write check; 2nd: post-write check
                return "0" * 64
        return real_hash(tensor)

    monkeypatch.setattr(patches, "tensor_hash", flaky_hash)
    with pytest.raises(ValueError, match="after write"):
        apply_patch(target, tmp_path / "edit")
    assert_same(target, base)
