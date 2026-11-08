"""Small LoRA layers and strict adapter loading for the pinned base checkpoint."""
from __future__ import annotations
import torch
from torch import nn
from pathlib import Path
from contextlib import contextmanager

from .artifacts import file_hash


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.enabled = True
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)

    def forward(self, value):
        if not self.enabled:
            return self.base(value)
        update = torch.nn.functional.linear(torch.nn.functional.linear(value.float(), self.a), self.b)
        return self.base(value) + update.to(value.dtype) * 2


def attach(model, rank=8, targets=("q_proj", "v_proj")):
    model.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] in targets:
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child, LowRankLinear(module, rank))
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_adapter(model, specification, base_hashes):
    from safetensors.torch import load_file
    path = Path(specification["checkpoint"])
    if file_hash(path) != specification["sha256"] or specification["base_checkpoint_hashes"] != base_hashes:
        raise ValueError("The adapter or base checkpoint differs from its recorded hashes")
    tensors = load_file(path, device=str(next(model.parameters()).device))
    parameters = attach(model, specification["rank"], specification.get("target_modules", ("q_proj", "v_proj")))
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
