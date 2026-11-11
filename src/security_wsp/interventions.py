"""Reversible projection edits, exact tensor accounting, and bounded trials."""
from __future__ import annotations
import hashlib
import math
import os
import random
import struct
from tempfile import TemporaryDirectory
from pathlib import Path
from .artifacts import atomic_json, digest, file_hash, read_json, utc_now


def fp32(values):
    return [struct.unpack("<f", struct.pack("<f", float(x)))[0] for x in values]


def vector_hash(values):
    return hashlib.sha256(b"".join(struct.pack("<f", v) for v in fp32(values))).hexdigest()


def project_vector(weights, direction, strength):
    if not 0 <= strength <= 1 or len(weights) != len(direction):
        raise ValueError("Invalid projection strength or shape")
    norm = math.sqrt(sum(x*x for x in direction))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("Direction must be finite and nonzero")
    d = [x/norm for x in direction]
    dot = sum(w*x for w, x in zip(weights, d))
    return fp32([w-strength*dot*x for w, x in zip(weights, d)])


def write_toy_checkpoint(path: Path, original, edited, recipe):
    original, edited = fp32(original), fp32(edited)
    record = {"format": "aawden-constructed-toy-v1", "architecture": "2-input-linear-choice",
              "weights": edited, "recipe": recipe, "base_sha256": vector_hash(original),
              "edited_sha256": vector_hash(edited), "dtype": "float32", "shape": [2],
              "touched_entries": 2, "changed_entries": sum(a != b for a, b in zip(original, edited)),
              "update_l2": math.sqrt(sum((a-b)**2 for a, b in zip(original, edited))),
              "non_parameter_buffers": 0, "architecture_changed": False, "training_used": False,
              "evidence_kind": "constructed_toy", "warning": "Not a Cohere checkpoint or cybersecurity capability result."}
    atomic_json(path, record)
    return record


def load_toy_checkpoint(path):
    record = read_json(path)
    if record.get("format") != "aawden-constructed-toy-v1" or vector_hash(record["weights"]) != record["edited_sha256"]:
        raise ValueError("Invalid toy checkpoint or tensor integrity failure")
    return record["weights"]


def random_matched_update(weights, norm, seed):
    rng = random.Random(seed)
    d = [rng.gauss(0, 1) for _ in weights]
    length = math.sqrt(sum(v*v for v in d))
    return fp32([w + norm*v/length for w, v in zip(weights, d)])


class TrialRegistry:
    def __init__(self, path: Path, identity: dict, budget=10):
        if not 1 <= budget <= 10:
            raise ValueError("Candidate budget must be between one and ten")
        self.path, self.identity, self.budget = path, digest(identity), budget
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.state = read_json(path)
            if self.state["identity_sha256"] != self.identity or self.state["budget"] != budget:
                raise ValueError("Trial registry belongs to a different experiment")
        else:
            self.state = {"identity_sha256": self.identity, "budget": budget, "trials": []}
            atomic_json(path, self.state)

    def reserve(self, recipe):
        lock = self.path.with_suffix(".lock")
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ValueError("Another candidate writer owns the registry")
        try:
            self.state = read_json(self.path)
            candidate_id = digest(recipe)[:16]
            for trial in self.state["trials"]:
                if trial["candidate_id"] == candidate_id:
                    return trial
            if len(self.state["trials"]) >= self.budget:
                raise ValueError("Declared edit-candidate budget exhausted")
            trial = {"candidate_id": candidate_id, "recipe": recipe, "status": "running", "reserved_at": utc_now()}
            self.state["trials"].append(trial)
            atomic_json(self.path, self.state)
            return trial
        finally:
            os.close(handle)
            lock.unlink()

    def complete(self, candidate_id, **results):
        return self.record(candidate_id, status="completed", completed_at=utc_now(), **results)

    def record(self, candidate_id, **results):
        self.state = read_json(self.path)
        for trial in self.state["trials"]:
            if trial["candidate_id"] == candidate_id:
                trial.update(**results)
                atomic_json(self.path, self.state)
                return trial
        raise ValueError("Unknown trial")


def tensor_hash(tensor):
    """Hash actual stored bytes, including BF16 rounding."""
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def project_output_tensor(weight, direction, strength):
    import torch
    if weight.ndim != 2 or direction.ndim != 1 or weight.shape[0] != direction.numel():
        raise ValueError("Expected a Hugging Face linear weight [residual_output, input] and a matching residual direction")
    if not 0 <= strength <= 1 or not torch.isfinite(direction).all():
        raise ValueError("Invalid projection direction/strength")
    direction = direction.to(device=weight.device, dtype=torch.float32)
    norm = direction.norm()
    if norm <= 1e-12:
        raise ValueError("Zero direction")
    d = direction / norm
    result = weight.float() - strength * d[:, None] * (d @ weight.float())[None, :]
    return result.to(weight.dtype)


def export_hf_projection(adapter, tensor_names, direction, strength, output: Path, discovery_hash: str) -> dict:
    """Write a new checkpoint; restore all in-memory source parameters afterward."""
    if output.exists():
        raise ValueError("Checkpoint output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".checkpoint-") as temporary:
        staging = Path(temporary) / "checkpoint"
        manifest = _serialize_hf_projection(adapter, tensor_names, direction, strength, staging, discovery_hash)
        staging.rename(output)
        return manifest


def _serialize_hf_projection(adapter, tensor_names, direction, strength, output: Path, discovery_hash: str) -> dict:
    import torch
    params = dict(adapter.model.named_parameters())
    if not tensor_names or len(set(tensor_names)) != len(tensor_names):
        raise ValueError("Select unique output-projection tensors")
    originals, edits = {}, []
    try:
        for name in tensor_names:
            if not name.endswith((".self_attn.o_proj.weight", ".mlp.down_proj.weight")) or name not in params:
                raise ValueError("Only named attention/MLP residual output projections are allowed")
            param = params[name]
            originals[name] = param.detach().clone()
            d = torch.tensor(direction, device=param.device)
            updated = project_output_tensor(param.detach(), d, strength)
            delta = updated.float() - originals[name].float()
            edits.append({"name": name, "shape": list(param.shape), "dtype": str(param.dtype),
                          "original_sha256": tensor_hash(param), "edited_sha256": tensor_hash(updated),
                          "touched_entries": param.numel(), "changed_entries": int((updated != param).sum().item()),
                          "update_l2": delta.norm().item(), "update_max": delta.abs().max().item()})
            with torch.no_grad():
                param.copy_(updated)
        output.mkdir(parents=True)
        adapter.model.save_pretrained(output, safe_serialization=True)
        adapter.tokenizer.save_pretrained(output)
        # Verify serialized tensors, not only the temporary model in memory.
        from safetensors import safe_open
        stored = {}
        for shard in output.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for edit in edits:
                    if edit["name"] in tensors.keys():
                        stored[edit["name"]] = tensors.get_tensor(edit["name"])
        for edit in edits:
            if edit["name"] not in stored or tensor_hash(stored[edit["name"]]) != edit["edited_sha256"]:
                raise ValueError("Serialized tensor verification failed")
        manifest = {"version": 1, "base": adapter.metadata, "discovery_sha256": discovery_hash, "strength": strength,
                    "direction_sha256": digest(direction),
                    "tensors": edits, "training_used": False, "runtime_hooks_required": False, "architecture_changed": False,
                    "formula": "W' = W - alpha*d*(d^T*W), d is unit length, output dimension first",
                    "bias_policy": "Output projection biases are unchanged; the edit does not claim equivalence to full activation removal.",
                    "status": "serialized_verified; fresh-process behavioral reload still required",
                    "checkpoint_hashes": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
        atomic_json(output / "aawden_edit.json", manifest)
        return manifest
    finally:
        for name, original in originals.items():
            with torch.no_grad():
                params[name].copy_(original)
            if tensor_hash(params[name]) != tensor_hash(original):
                raise RuntimeError("Source-parameter restoration failed")

