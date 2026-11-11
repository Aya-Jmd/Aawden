"""Sparse row/column weight patches: build, save, verified apply, and verified revert.

A patch stores only the rows (axis 0) or columns (axis 1) of each tensor that changed, together
with the original slices needed for revert. Every apply and revert checks the sha256 of the patch
file and of each target tensor, so a patch cannot be applied to the wrong base or silently altered.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .artifacts import atomic_json, file_hash, read_json
from .interventions import tensor_hash

PATCH_FORMAT = "aawden-patch-v1"


@dataclass
class TensorPatch:
    name: str
    shape: tuple
    dtype: torch.dtype
    axis: int
    indices: torch.Tensor  # int64 positions of the changed slices along `axis`
    values: torch.Tensor  # edited slices
    original: torch.Tensor  # original slices, used by revert
    original_sha256: str
    patched_sha256: str
    changed_scalars: int


@dataclass
class Patch:
    tensors: list[TensorPatch]
    total_parameters: int


def _differs(original: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
    """Element mask of scalars whose stored bytes differ (NaN and signed zeros compare bitwise)."""
    width = original.element_size()
    a = original.detach().cpu().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    b = edited.detach().cpu().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    return (a != b).any(dim=1).reshape(original.shape)


def _slice_mask(differs: torch.Tensor, axis: int) -> torch.Tensor:
    return differs.movedim(axis, 0).reshape(differs.shape[axis], -1).any(dim=1)


def build_patch(original_state: dict[str, torch.Tensor], edited_state: dict[str, torch.Tensor]) -> Patch:
    if set(original_state) != set(edited_state):
        raise ValueError("Original and edited state dicts have different tensor names")
    tensors, total = [], 0
    for name in sorted(original_state):
        original, edited = original_state[name], edited_state[name]
        if original.shape != edited.shape or original.dtype != edited.dtype:
            raise ValueError(f"{name}: shape or dtype differs between original and edited tensors")
        total += original.numel()
        if original.numel() == 0:
            continue
        if original.ndim == 0:
            raise ValueError(f"{name}: scalar tensors cannot be patched by row or column")
        differs = _differs(original, edited)
        changed = int(differs.sum())
        if changed == 0:
            continue
        candidates = [0, 1] if original.ndim >= 2 else [0]
        masks = {axis: _slice_mask(differs, axis) for axis in candidates}
        axis = min(candidates, key=lambda a: int(masks[a].sum()))  # ties keep axis 0
        indices = masks[axis].nonzero().flatten().to(torch.int64)
        original_cpu = original.detach().cpu().contiguous()
        edited_cpu = edited.detach().cpu().contiguous()
        values = edited_cpu.index_select(axis, indices).clone()
        saved_original = original_cpu.index_select(axis, indices).clone()
        rebuilt = original_cpu.clone()
        rebuilt.index_copy_(axis, indices, values)
        if tensor_hash(rebuilt) != tensor_hash(edited_cpu):
            raise RuntimeError(f"{name}: sparse slices do not reconstruct the edited tensor")
        tensors.append(TensorPatch(
            name=name, shape=tuple(original.shape), dtype=original.dtype, axis=axis, indices=indices,
            values=values, original=saved_original, original_sha256=tensor_hash(original_cpu),
            patched_sha256=tensor_hash(edited_cpu), changed_scalars=changed))
    return Patch(tensors=tensors, total_parameters=total)


def _paths(path) -> tuple[Path, Path]:
    stem = Path(path)
    return stem.with_name(stem.name + ".safetensors"), stem.with_name(stem.name + ".json")


def save_patch(patch: Patch, path, manifest_extra: dict) -> dict:
    """Write <path>.safetensors and <path>.json; the JSON manifest is written last."""
    weights_path, manifest_path = _paths(path)
    stored, entries, changed = {}, [], 0
    for item in patch.tensors:
        stored[f"{item.name}::values"] = item.values.contiguous()
        stored[f"{item.name}::original"] = item.original.contiguous()
        stored[f"{item.name}::indices"] = item.indices.contiguous()
        changed += item.changed_scalars
        entries.append({
            "name": item.name, "shape": list(item.shape), "dtype": str(item.dtype), "axis": item.axis,
            "indices_count": int(item.indices.numel()), "changed_scalars": item.changed_scalars,
            "original_sha256": item.original_sha256, "patched_sha256": item.patched_sha256,
        })
    save_file(stored, str(weights_path), metadata={"format": PATCH_FORMAT})
    total = patch.total_parameters
    manifest = {
        "format": PATCH_FORMAT,
        "weights_file": weights_path.name,
        "weights_sha256": file_hash(weights_path),
        "tensors": entries,
        "changed_scalars": changed,
        "total_parameters": total,
        "percent_changed": 100.0 * changed / total if total else 0.0,
        "extra": manifest_extra,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _load_verified(path):
    weights_path, manifest_path = _paths(path)
    manifest = read_json(manifest_path)
    if manifest.get("format") != PATCH_FORMAT:
        raise ValueError("Unsupported or missing patch format")
    if not weights_path.exists() or file_hash(weights_path) != manifest["weights_sha256"]:
        raise ValueError("Patch weights failed sha256 verification (tampered or corrupt)")
    return manifest, load_file(str(weights_path))


@dataclass
class _Step:
    name: str
    target: torch.Tensor
    axis: int
    indices: torch.Tensor
    slices: torch.Tensor  # slices written by this direction (values for apply, original for revert)
    before: str  # sha256 of the target bytes before the transfer
    after: str  # sha256 the target must have after the transfer
    snapshot: torch.Tensor  # clone of the live target bytes, used for rollback


def _check_indices(name: str, axis: int, indices: torch.Tensor, count: int, length: int) -> None:
    if indices.dtype != torch.int64 or indices.ndim != 1 or indices.numel() != count:
        raise ValueError(f"{name}: stored indices do not match the manifest (dtype, rank or count)")
    if count > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError(f"{name}: stored indices must be unique and strictly increasing")
    if count and (int(indices[0]) < 0 or int(indices[-1]) >= length):
        raise ValueError(f"{name}: stored index out of range for axis {axis} of length {length}")


def _restore(plan: list[_Step]) -> None:
    """Rebuild every target from the snapshots taken before the first write, then verify the bytes."""
    with torch.no_grad():
        for step in plan:
            step.target.copy_(step.snapshot)
    for step in plan:
        if tensor_hash(step.target) != step.before:
            raise RuntimeError(f"{step.name}: rollback did not restore the pre-transfer hash")


def _transfer(model, path, forward: bool) -> dict:
    manifest, stored = _load_verified(path)
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    direction = "apply" if forward else "revert"
    seen, plan = set(), []
    # Phase 1: validate everything on copies. Nothing in the model is written here.
    for entry in manifest["tensors"]:
        name = entry["name"]
        if name in seen:
            raise ValueError(f"Patch lists {name} more than once")
        seen.add(name)
        if name not in targets:
            raise ValueError(f"Patch target {name} is missing from the model")
        target = targets[name]
        if list(target.shape) != entry["shape"] or str(target.dtype) != entry["dtype"]:
            raise ValueError(f"{name}: shape or dtype does not match the patch")
        axis = entry["axis"]
        if not isinstance(axis, int) or not 0 <= axis < target.ndim:
            raise ValueError(f"{name}: patch axis {axis} is invalid for shape {list(target.shape)}")
        missing = [part for part in ("values", "original", "indices") if f"{name}::{part}" not in stored]
        if missing:
            raise ValueError(f"{name}: patch file is missing stored tensors {missing}")
        count = entry["indices_count"]
        indices = stored[f"{name}::indices"]
        _check_indices(name, axis, indices, count, target.shape[axis])
        slice_shape = list(target.shape)
        slice_shape[axis] = count
        values, original = stored[f"{name}::values"], stored[f"{name}::original"]
        for part, tensor in (("values", values), ("original", original)):
            if list(tensor.shape) != slice_shape or tensor.dtype != target.dtype:
                raise ValueError(f"{name}: stored {part} slices have shape {list(tensor.shape)} and dtype "
                                 f"{tensor.dtype}; expected {slice_shape} and {target.dtype}")
        slices = values if forward else original
        before = entry["original_sha256"] if forward else entry["patched_sha256"]
        after = entry["patched_sha256"] if forward else entry["original_sha256"]
        if tensor_hash(target) != before:
            raise ValueError(f"{name}: wrong base, current tensor does not match the "
                             f"{'original' if forward else 'patched'} hash recorded in the patch")
        probe = target.detach().clone()
        probe.index_copy_(axis, indices.to(probe.device), slices.to(device=probe.device, dtype=probe.dtype))
        if tensor_hash(probe) != after:
            raise ValueError(f"{name}: stored {'values' if forward else 'original'} slices do not reproduce "
                             f"the recorded {'patched' if forward else 'original'} hash; {direction} refused, "
                             "nothing written")
        plan.append(_Step(name=name, target=target, axis=axis, indices=indices, slices=slices,
                          before=before, after=after, snapshot=target.detach().clone()))
    # Phase 2: write. Any failure restores every target from its pre-write snapshot.
    try:
        with torch.no_grad():
            for step in plan:
                step.target.index_copy_(step.axis, step.indices.to(step.target.device),
                                        step.slices.to(device=step.target.device, dtype=step.target.dtype))
            for step in plan:
                if tensor_hash(step.target) != step.after:
                    raise ValueError(f"{step.name}: tensor hash after write does not match the recorded hash")
    except BaseException:
        _restore(plan)
        raise
    return manifest


def apply_patch(model, path) -> dict:
    """Apply a verified patch in place; raises ValueError (before any write) on a wrong base."""
    return _transfer(model, path, forward=True)


def revert_patch(model, path) -> dict:
    """Restore the original slices in place; raises ValueError if the model is not the patched one."""
    return _transfer(model, path, forward=False)
