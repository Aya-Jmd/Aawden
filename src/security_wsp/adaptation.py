"""Small LoRA layers and strict adapter loading for the pinned base checkpoint."""
from __future__ import annotations
import torch
from torch import nn
from pathlib import Path
from contextlib import contextmanager

from .artifacts import file_hash


DEFAULT_SCALE = 2.0  # Historical fixed scale; adapters saved without an alpha keep it.


def lora_scale(rank, alpha=None):
    """Return alpha / rank when alpha is given, otherwise the historical scale of 2.0."""
    if alpha is None:
        return DEFAULT_SCALE
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    return float(alpha) / rank


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8, scale=DEFAULT_SCALE):
        super().__init__()
        self.base = base
        self.enabled = True
        self.scale = float(scale)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)

    def forward(self, value):
        if not self.enabled:
            return self.base(value)
        update = torch.nn.functional.linear(torch.nn.functional.linear(value.float(), self.a), self.b)
        return self.base(value) + update.to(value.dtype) * self.scale


def attach(model, rank=8, targets=("q_proj", "v_proj"), alpha=None):
    model.requires_grad_(False)
    scale = lora_scale(rank, alpha)
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] in targets or (name and "." not in name and name in targets):
            parent, _, child = name.rpartition(".")
            holder = model.get_submodule(parent) if parent else model
            setattr(holder, child, LowRankLinear(module, rank, scale))
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


@torch.no_grad()
def merge_dense(model):
    """Fold every LowRankLinear update into its base nn.Linear, in place.

    For each LowRankLinear, W' = W + scale * (B @ A) is computed in float32 and cast back to the
    base weight dtype; the LowRankLinear is replaced by its base module. A disabled adapter is
    removed without adding its update, so the merged model matches the disabled-adapter behaviour.
    """
    found = [(name, module) for name, module in model.named_modules() if isinstance(module, LowRankLinear)]
    if any(not name for name, _ in found):
        raise ValueError("Cannot merge a root LowRankLinear; wrap it in a parent module first")
    names, disabled = [], []
    for name, layer in found:
        base = layer.base
        merged = base.weight.detach().float()
        if layer.enabled:
            merged = merged + layer.scale * (layer.b.float() @ layer.a.float())
        else:
            disabled.append(name)
        base.weight.copy_(merged.to(base.weight.dtype))
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, base)
        names.append(name)
    return {"modules_merged": len(names), "names": names, "disabled_without_update": disabled}


def load_adapter(model, specification, base_hashes):
    from safetensors.torch import load_file
    path = Path(specification["checkpoint"])
    if file_hash(path) != specification["sha256"] or specification["base_checkpoint_hashes"] != base_hashes:
        raise ValueError("The adapter or base checkpoint differs from its recorded hashes")
    tensors = load_file(path, device=str(next(model.parameters()).device))
    parameters = attach(model, specification["rank"], specification.get("target_modules", ("q_proj", "v_proj")),
                        alpha=specification.get("alpha"))
    if set(tensors) != set(parameters) or any(tensors[name].shape != value.shape for name, value in parameters.items()):
        raise ValueError("Adapter tensors do not match the configured architecture")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name])
            parameter.requires_grad_(False)
    model.eval()


@contextmanager
def without_adapter(model):
    layers = [module for module in model.modules() if isinstance(module, LowRankLinear)]
    states = [layer.enabled for layer in layers]
    try:
        for layer in layers:
            layer.enabled = False
        yield
    finally:
        for layer, enabled in zip(layers, states, strict=True):
            layer.enabled = enabled
