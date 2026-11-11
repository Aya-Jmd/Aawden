"""Bounded tool layer for the weight-space programming (WSP) agent.

WspTools is the only route by which the phase-4 programmer agent touches the model. It offers:
read-only inspection of editable weight sites, residual/MLP/attention contrasts and attention
measurements on discovery cases, batched evaluation of sparse row/column edits with verified
restoration, a size-matched random control, bounded submission, and a full ordered transcript.

Invariants (see docs/WSP_TOOLS.md):
- Discovery data only. Cases with split wsp_validation or wsp_final are refused unconditionally;
  any split outside `allowed_splits` is refused too.
- Editable sites only: names must match `site_patterns` and name a 2-D parameter. No gradients.
- No truncation. A prompt that does not fit the context limit raises before any model work.
- Edits are written in place under torch.no_grad, evaluated, and restored. Touched tensors are hashed
  before and after; a mismatch raises RuntimeError.
- Budget: one unit per physical batch sent through the model (generate or forward). Units are
  reserved after planning and before any weight is written, so an exhausted budget raises without
  side effects. list_sites, read_weights, propose_random_control and submit consume no units.
- Every public tool call increments a sequence counter and appends one transcript entry, including
  calls that raise.

Batching: prompts are sorted by token length and grouped into batches of up to batch_size. Each
batch is left-padded with the pad token and carries an attention mask. Forward passes get explicit
position_ids (cumsum(mask) - 1), so padded and unpadded rows see identical positions; the answer
position is then the last column for every row. Greedy generation uses HF generate, which derives
positions from the mask itself. Batched bf16 arithmetic can differ from batch size 1 by rounding.
attention_to_span uses a separate, smaller physical batch (attention_batch_size) because
output_attentions returns every layer.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import functools
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from .artifacts import atomic_json, digest
from .data import Case
from .interventions import tensor_hash
from .parsing import parse_response
from .patches import Patch, build_patch, save_patch
from .prompts import CONDITIONS, PREFILL, render
from .statistics import metrics

FORBIDDEN_SPLITS = frozenset({"wsp_validation", "wsp_final"})
DEFAULT_ALLOWED_SPLITS = ("wsp_discovery",)
SITE_KINDS = {"resid_post": "", "mlp_out": ".mlp.down_proj", "attn_out": ".self_attn.o_proj"}
PROBE_CONDITIONS = ("neutral", "incorrect_suggestion")
SCALAR_CAP = 4096
RANDOM_DIRECTIONS = 32
TRANSCRIPT_FORMAT = "aawden-wsp-transcript-v1"


class BudgetExhausted(RuntimeError):
    """Raised before any model work when a call would exceed the forward-pass budget."""


@dataclass
class Edit:
    """Overwrite whole rows (axis=0) or whole columns (axis=1) of one 2-D editable tensor.

    values[k] is the replacement for indices[k]: a row of length in_features when axis=0, or a
    column of length out_features when axis=1.
    """

    tensor_name: str
    axis: int
    indices: list
    values: list

    @classmethod
    def from_any(cls, item):
        if isinstance(item, Edit):
            return item
        if isinstance(item, dict):
            return cls(item["tensor_name"], item["axis"], list(item["indices"]), [list(v) for v in item["values"]])
        raise TypeError("An Edit or a dict with tensor_name/axis/indices/values is required")


def _jsonable(value):
    if isinstance(value, Patch):
        return {"patch_tensors": [t.name for t in value.tensors], "total_parameters": value.total_parameters}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _estimate_tokens(value) -> int:
    """Rough cost estimate: one token per four characters of canonical JSON."""
    return math.ceil(len(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)) / 4)


def _logged(tool):
    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            self._seq += 1
            entry = {"seq": self._seq, "tool": tool,
                     "arguments": _jsonable({"args": list(args), "kwargs": kwargs}),
                     "status": "running", "forward_units": 0, "model_prompt_tokens": 0}
            self._log.append(entry)
            self._current = entry
            try:
                output = method(self, *args, **kwargs)
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
            else:
                entry["status"] = "ok"
                entry["output"] = _jsonable(output)
                return output
            finally:
                self._current = None
                entry["estimated_tokens"] = {"input": _estimate_tokens(entry["arguments"]),
                                             "output": _estimate_tokens(entry.get("output"))}
        return wrapper
    return decorate


def _stats(x: torch.Tensor) -> dict:
    x = x.detach().double()
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)), "min": float(x.min()), "max": float(x.max())}


def _changed_scalars(before: torch.Tensor, after: torch.Tensor) -> int:
    """Count elements whose stored bytes differ (bitwise, so signed zeros and NaN are not conflated)."""
    before, after = before.detach().cpu().contiguous(), after.detach().cpu().contiguous()
    width = before.element_size()
    a = before.reshape(-1).view(torch.uint8).reshape(-1, width)
    b = after.reshape(-1).view(torch.uint8).reshape(-1, width)
    return int((a != b).any(dim=1).sum())


def _orient(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """Convert between index_select/index_copy_ orientation and (slice_count, slice_length) orientation."""
    return tensor if axis == 0 else tensor.T


def _find_positions(haystack: list, needle: list) -> list:
    width = len(needle)
    if width == 0:
        return []
    return [i for i in range(len(haystack) - width + 1) if haystack[i:i + width] == needle]


def _seed_for(*parts) -> int:
    return int(digest(list(parts))[:12], 16)


def _edit_sets(edits) -> list:
    """Normalize a Patch, one edit set, or a list of edit sets into a list of edit sets (lists of Edit)."""
    if isinstance(edits, Patch):
        # TensorPatch.values are stored in index_select orientation; Edit.values are slice-major.
        return [[Edit(t.name, t.axis, t.indices.tolist(), _orient(t.values.float(), t.axis).tolist())
                 for t in edits.tensors]]
    items = list(edits)
    if not items:
        raise ValueError("No edits supplied")
    nested = [isinstance(item, (list, tuple)) for item in items]
    if all(nested):
        return [[Edit.from_any(e) for e in edit_set] for edit_set in items]
    if any(nested):
        raise ValueError("Pass either one edit set or a list of edit sets, not a mix of both")
    return [[Edit.from_any(e) for e in items]]


class WspTools:
    """Discovery-only tool layer. See the module docstring and docs/WSP_TOOLS.md for the rules.

    `model` must be a dense Cohere checkpoint loaded in eval mode (attention_to_span switches it to
    eager attention temporarily). `cases` must all have a split in `allowed_splits`.
    """

    def __init__(self, model, tokenizer, cases, *, device, site_patterns, max_changed_fraction,
                 forward_pass_budget, batch_size=16, seed=42, allowed_splits=DEFAULT_ALLOWED_SPLITS,
                 max_new_tokens=256, max_submissions=10, output_dir=None, context_limit=None,
                 attention_batch_size=4, permutations=2000, top_k=5):
        if getattr(model.config, "model_type", None) not in ("cohere", "cohere2"):
            raise ValueError("WSP tools support dense Cohere checkpoints only")
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("The tokenizer has no chat template; implicit prompt substitutions are prohibited")
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("The tokenizer needs a pad or eos token for left padding")
        allowed = tuple(allowed_splits)
        if not allowed or FORBIDDEN_SPLITS.intersection(allowed):
            raise ValueError("allowed_splits must be non-empty and must not contain validation or final splits")
        cases = tuple(cases)
        if not cases:
            raise ValueError("At least one discovery case is required")
        for case in cases:
            if case.split in FORBIDDEN_SPLITS:
                raise ValueError(f"Refusing case {case.case_id}: split {case.split!r} is never visible to WSP tools")
            if case.split not in allowed:
                raise ValueError(f"Refusing case {case.case_id}: split {case.split!r} is not in {allowed}")
        if not site_patterns:
            raise ValueError("site_patterns must list the editable tensors")
        if not 0 < max_changed_fraction <= 1:
            raise ValueError("max_changed_fraction must be in (0, 1]")
        if forward_pass_budget < 0 or batch_size < 1 or attention_batch_size < 1 or max_new_tokens < 1:
            raise ValueError("Budget must be non-negative; batch sizes and max_new_tokens must be positive")
        candidates = [v for v in (context_limit, getattr(model.config, "max_position_embeddings", None)) if v]
        if not candidates:
            raise ValueError("A context limit is required; prompts are never truncated")

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.cases = cases
        self.site_patterns = tuple(site_patterns)
        self.max_changed_fraction = float(max_changed_fraction)
        self.forward_pass_budget = int(forward_pass_budget)
        self.batch_size = int(batch_size)
        self.attention_batch_size = int(attention_batch_size)
        self.seed = int(seed)
        self.allowed_splits = allowed
        self.max_new_tokens = int(max_new_tokens)
        self.max_submissions = int(max_submissions)
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.permutations = int(permutations)
        self.top_k = int(top_k)
        self.context_limit = int(min(candidates))
        self._pad_id = int(pad)
        self._params = dict(self.model.named_parameters())
        self.total_parameters = sum(p.numel() for p in self._params.values())
        self._seq = 0
        self._log: list[dict] = []
        self._current: dict | None = None
        self._units_used = 0

    # ----- budget and bookkeeping -------------------------------------------------------------
    @property
    def budget_used(self) -> int:
        return self._units_used

    @property
    def budget_remaining(self) -> int:
        return self.forward_pass_budget - self._units_used

    @property
    def calls(self) -> int:
        return self._seq

    def _reserve(self, units: int, prompt_tokens: int = 0) -> None:
        if units < 0:
            raise ValueError("Negative budget request")
        if self._units_used + units > self.forward_pass_budget:
            raise BudgetExhausted(f"forward-pass budget exhausted: {self._units_used} used, {units} requested, "
                                  f"budget {self.forward_pass_budget}")
        self._units_used += units
        if self._current is not None:
            self._current["forward_units"] += units
            self._current["model_prompt_tokens"] += prompt_tokens

    # ----- editable sites ---------------------------------------------------------------------
    def _is_editable(self, name: str) -> bool:
        param = self._params.get(name)
        return (param is not None and param.ndim == 2
                and any(fnmatch.fnmatchcase(name, pattern) for pattern in self.site_patterns))

    def _check_editable(self, name: str) -> None:
        if not self._is_editable(name):
            raise PermissionError(f"{name!r} is not an editable site under the protocol site patterns")

    def _plan_edits(self, edit_set: list) -> list[dict]:
        """Validate one edit set and return write-ready entries. Raises before any weight is touched."""
        if not edit_set:
            raise ValueError("Empty edit set")
        seen, plan, touched_total = set(), [], 0
        for raw in edit_set:
            edit = Edit.from_any(raw)
            name = edit.tensor_name
            self._check_editable(name)
            if name in seen:
                raise ValueError(f"{name}: each tensor may appear once per edit set")
            seen.add(name)
            if isinstance(edit.axis, bool) or edit.axis not in (0, 1):
                raise ValueError(f"{name}: axis must be 0 (rows) or 1 (columns)")
            param = self._params[name]
            size = param.shape[edit.axis]
            width = param.shape[1 - edit.axis]
            indices = edit.indices
            if not isinstance(indices, (list, tuple)) or not indices:
                raise ValueError(f"{name}: indices must be a non-empty list")
            if any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
                raise ValueError(f"{name}: indices must be integers")
            if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= size:
                raise ValueError(f"{name}: indices must be unique and within [0, {size})")
            if len(edit.values) != len(indices) or any(len(v) != width for v in edit.values):
                raise ValueError(f"{name}: values must hold {len(indices)} slices of length {width}")
            try:
                values = torch.tensor(edit.values, dtype=torch.float64)
            except (TypeError, ValueError):
                raise ValueError(f"{name}: values must be numeric") from None
            if not torch.isfinite(values).all():
                raise ValueError(f"{name}: values must be finite")
            src = values.to(dtype=param.dtype).contiguous()
            if not torch.isfinite(src.float()).all():
                raise ValueError(f"{name}: values overflow the tensor dtype {param.dtype}")
            src = _orient(src, edit.axis).contiguous()  # index_copy_ orientation
            touched = len(indices) * width
            touched_total += touched
            plan.append({"tensor_name": name, "param": param, "axis": edit.axis,
                         "index": torch.tensor(indices, dtype=torch.long), "src": src, "touched": touched})
        fraction = touched_total / self.total_parameters
        if fraction > self.max_changed_fraction:
            raise ValueError(f"Edit set touches {touched_total} of {self.total_parameters} parameters "
                             f"({fraction:.6g}); the limit is {self.max_changed_fraction:.6g}")
        return plan

    @contextmanager
    def _applied(self, plan: list[dict]):
        """Write the edit set in place, yield a summary, then restore and verify by tensor hash."""
        before, applied = {}, []
        info = {"before_sha256": {}, "edited_sha256": {}, "changed_scalars": 0, "restored": False}
        try:
            with torch.no_grad():
                for item in plan:
                    param, axis, name = item["param"], item["axis"], item["tensor_name"]
                    index = item["index"].to(param.device)
                    before[name] = tensor_hash(param)
                    original = param.index_select(axis, index).clone()
                    src = item["src"].to(device=param.device, dtype=param.dtype)
                    info["changed_scalars"] += _changed_scalars(original, src)
                    param.index_copy_(axis, index, src)
                    applied.append((param, axis, index, original))
                    info["edited_sha256"][name] = tensor_hash(param)
                    info["before_sha256"][name] = before[name]
            yield info
        finally:
            with torch.no_grad():
                for param, axis, index, original in reversed(applied):
                    param.index_copy_(axis, index, original)
            for name, digest_before in before.items():
                if tensor_hash(self._params[name]) != digest_before:
                    raise RuntimeError(f"Restoration failed for {name}: the model is no longer the base model")
            info["restored"] = True

    # ----- prompts and batching ---------------------------------------------------------------
    def _request_parts(self, case: Case, condition: str, **kwargs):
        request = render(case, condition, prefill=True, max_new_tokens=self.max_new_tokens, seed=self.seed, **kwargs)
        context, final = request.messages[:-1], request.messages[-1]
        if final["role"] != "assistant" or final["content"] != PREFILL:
            raise RuntimeError("Expected the JSON prefill as the final assistant message")
        text = self.tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True) + PREFILL
        return text, context[-1]["content"]

    def _prepare(self, requests, *, reserve: int, **kwargs) -> list[dict]:
        prompts = []
        for case, condition in requests:
            text, user = self._request_parts(case, condition, **kwargs)
            tokens = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(tokens) + reserve > self.context_limit:
                raise ValueError(f"Prompt {case.case_id}/{condition} needs {len(tokens)} + {reserve} tokens, above "
                                 f"the context limit {self.context_limit}; truncation is prohibited")
            prompts.append({"case": case, "condition": condition, "text": text, "user": user, "tokens": tokens})
        return prompts

    @staticmethod
    def _batches(token_lists: list, size: int) -> list[list[int]]:
        order = sorted(range(len(token_lists)), key=lambda i: (len(token_lists[i]), i))
        return [order[k:k + size] for k in range(0, len(order), size)]

    def _inputs(self, token_lists: list, rows: list[int]):
        width = max(len(token_lists[i]) for i in rows)
        ids = torch.full((len(rows), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        pads = []
        for r, i in enumerate(rows):
            seq = token_lists[i]
            start = width - len(seq)
            ids[r, start:] = torch.tensor(seq, dtype=torch.long)
            mask[r, start:] = 1
            pads.append(start)
        positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
        return ids.to(self.device), mask.to(self.device), positions.to(self.device), pads

    def _locate(self, tokens: list, text: str, label: str):
        """Locate tokenize(text) inside the prefix tokens. Unique match or a reported failure, never a guess."""
        needle = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if not needle:
            return None, f"{label}: empty tokenization"
        hits = _find_positions(tokens, needle)
        if not hits:
            return None, f"{label}: no match in prefix"
        if len(hits) > 1:
            return None, f"{label}: {len(hits)} matches in prefix"
        return list(range(hits[0], hits[0] + len(needle))), None

    def _spans(self, prompt: dict, opinion_required: bool) -> dict:
        code = prompt["case"].visible()["code"].rstrip("\n")
        source, reason = self._locate(prompt["tokens"], code, "source")
        if reason:
            return {"reason": reason}
        opinion = None
        if opinion_required:
            lines = [line for line in prompt["user"].split("\n") if line.startswith("Reviewer assessment")]
            if len(lines) != 1:
                return {"reason": f"opinion: {len(lines)} candidate lines"}
            opinion, reason = self._locate(prompt["tokens"], lines[0], "opinion")
            if reason:
                return {"reason": reason}
        return {"source": source, "opinion": opinion}

    # ----- tools ------------------------------------------------------------------------------
    @_logged("list_sites")
    def list_sites(self, tensor_name_filter=None, top_k=None) -> list[dict]:
        """Editable 2-D tensors matching the protocol patterns, with stats and top columns by 2-norm."""
        k = self.top_k if top_k is None else int(top_k)
        sites = []
        for name, param in self._params.items():
            if not self._is_editable(name):
                continue
            if tensor_name_filter is not None and tensor_name_filter not in name:
                continue
            with torch.no_grad():
                values = param.detach().float()
                norms = values.norm(dim=0).tolist()
            order = sorted(range(len(norms)), key=lambda j: (-norms[j], j))[:k]
            sites.append({"tensor_name": name, "shape": list(param.shape), "dtype": str(param.dtype),
                          "total_scalars": param.numel(), "stats": _stats(values),
                          "top_columns": [{"column": j, "norm": norms[j]} for j in order]})
        return sites

    @_logged("read_weights")
    def read_weights(self, tensor_name, axis, indices=None, *, max_scalars=SCALAR_CAP) -> dict:
        """Exact stored values for selected slices. Over the scalar cap, only shape and stats are returned."""
        self._check_editable(tensor_name)
        if isinstance(axis, bool) or axis not in (0, 1):
            raise ValueError("axis must be 0 (rows) or 1 (columns)")
        param = self._params[tensor_name]
        size, width = param.shape[axis], param.shape[1 - axis]
        if indices is None:
            chosen = list(range(size))
        else:
            chosen = list(indices)
            if not chosen or any(isinstance(i, bool) or not isinstance(i, int) for i in chosen):
                raise ValueError("indices must be a non-empty list of integers")
            if len(set(chosen)) != len(chosen) or min(chosen) < 0 or max(chosen) >= size:
                raise ValueError(f"indices must be unique and within [0, {size})")
        scalars = len(chosen) * width
        info = {"tensor_name": tensor_name, "shape": list(param.shape), "dtype": str(param.dtype), "axis": axis,
                "slice_count": len(chosen), "slice_length": width, "scalar_count": scalars,
                "scalar_cap": int(max_scalars)}
        with torch.no_grad():
            selected = param.detach().index_select(axis, torch.tensor(chosen, dtype=torch.long, device=param.device))
            if scalars > max_scalars:
                return {**info, "values_returned": False, "stats": _stats(selected),
                        "reason": "requested scalars exceed the cap; shape and statistics returned instead"}
            slices = _orient(selected, axis).float().cpu().tolist()
        return {**info, "values_returned": True, "slice_meaning": "row" if axis == 0 else "column",
                "slices": slices}

    def _site_module(self, site: str, layer_index: int):
        key = site[5:] if site.startswith("hook_") else site
        if key not in SITE_KINDS:
            raise ValueError(f"site must be one of {sorted(SITE_KINDS)}")
        n_layers = self.model.config.num_hidden_layers
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or not 0 <= layer_index < n_layers:
            raise ValueError(f"layer_index must be an integer in [0, {n_layers})")
        target = f"model.layers.{layer_index}{SITE_KINDS[key]}"
        return self.model.get_submodule(target), key, target

    @_logged("activation_contrast")
    def activation_contrast(self, site: str, layer_index: int) -> dict:
        """Mean incorrect-minus-neutral activation direction at the answer position, with a sign-flip test.

        `site` is resid_post (decoder layer output), mlp_out (mlp.down_proj output) or attn_out
        (self_attn.o_proj output). The `hook_` prefix is accepted.
        """
        module, key, target = self._site_module(site, layer_index)
        requests = [(case, cond) for case in self.cases for cond in PROBE_CONDITIONS]
        prompts = self._prepare(requests, reserve=0)
        token_lists = [p["tokens"] for p in prompts]
        batches = self._batches(token_lists, self.batch_size)
        self._reserve(len(batches), sum(len(t) for t in token_lists))
        acts: list = [None] * len(prompts)
        for batch in batches:
            ids, mask, positions, _ = self._inputs(token_lists, batch)
            captured = {}

            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Keep one answer-position vector per row, not the whole sequence.
                captured["last"] = hidden[:, -1, :].detach().to(torch.float64).cpu()

            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    self.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
            finally:
                handle.remove()
            for row, index in enumerate(batch):
                acts[index] = captured["last"][row]
        # requests are ordered (case, neutral), (case, incorrect): acts[2c] and acts[2c+1]
        D = torch.stack([acts[2 * c + 1] - acts[2 * c] for c in range(len(self.cases))])
        report = self._contrast_report(D, (key, layer_index))
        return {"site": key, "layer_index": layer_index, "target_module": target,
                "position": "last prefix token (answer position)", "contrast": "incorrect_suggestion minus neutral",
                "prompt_count": len(prompts), "forward_units": len(batches), **report}

    def _contrast_report(self, D: torch.Tensor, parts: tuple) -> dict:
        n, hidden = D.shape
        norms = D.norm(dim=1)
        mean = D.mean(dim=0)
        mean_norm = float(mean.norm())
        rms = float(torch.sqrt((norms ** 2).mean()))
        report = {"cases": n, "hidden_size": hidden, "mean_difference_norm": mean_norm,
                  "per_case_norm_mean": float(norms.mean()), "per_case_norm_min": float(norms.min()),
                  "per_case_norm_max": float(norms.max())}
        if not math.isfinite(mean_norm) or mean_norm <= 1e-12 or rms <= 1e-12:
            report.update(degenerate=True, direction=None, cosine=None,
                          separation={"statistic": 0.0, "p_value": 1.0, "permutations": 0},
                          random_baseline=None)
            return report
        u = mean / mean_norm
        unit = D / norms.clamp_min(1e-12).unsqueeze(1)
        gram = unit @ unit.T
        off = gram[~torch.eye(n, dtype=torch.bool)]
        cosine = {"cos_to_mean_direction_mean": float((unit @ u).mean()),
                  "pairwise_mean": float(off.mean()) if n > 1 else None,
                  "pairwise_min": float(off.min()) if n > 1 else None}
        observed = mean_norm / rms
        gen = torch.Generator().manual_seed(_seed_for(self.seed, "sign-flip", *parts))
        signs = (torch.randint(0, 2, (self.permutations, n), generator=gen) * 2 - 1).double()
        null = ((signs @ D) / n).norm(dim=1) / rms
        p_value = (1 + int((null >= observed - 1e-12).sum())) / (self.permutations + 1)
        gen_r = torch.Generator().manual_seed(_seed_for(self.seed, "random-direction", *parts))
        R = torch.randn((RANDOM_DIRECTIONS, hidden), generator=gen_r, dtype=torch.float64)
        R = R / R.norm(dim=1, keepdim=True)
        proj = D @ R.T
        random_sep = proj.mean(0).abs() / torch.sqrt((proj ** 2).mean(0)).clamp_min(1e-12)
        proj_u = D @ u
        mean_dir_sep = float(proj_u.mean() / torch.sqrt((proj_u ** 2).mean()).clamp_min(1e-12))
        report.update(
            degenerate=False, direction=u.tolist(), cosine=cosine,
            separation={"statistic": observed, "p_value": p_value, "permutations": self.permutations,
                        "seed": _seed_for(self.seed, "sign-flip", *parts),
                        "statistic_definition": "||mean contrast|| / RMS per-case contrast norm, in [0, 1]",
                        "null": "independent sign flips of the paired contrasts"},
            projection_on_mean_direction={"separation": mean_dir_sep},
            random_baseline={"directions": RANDOM_DIRECTIONS, "mean_separation": float(random_sep.mean()),
                             "max_separation": float(random_sep.max()),
                             "statistic": "|mean projection| / RMS projection on seeded unit directions"})
        return report

    @_logged("attention_to_span")
    def attention_to_span(self, layer: int, head: int | None = None) -> list[dict]:
        """Final-position attention mass on the opinion span, the source span and other tokens.

        Compares the neutral prompt with the misleading (incorrect_suggestion) prompt. Cases whose
        opinion or source span is not a unique match are excluded and counted in match_failures.
        """
        n_layers = self.model.config.num_hidden_layers
        n_heads = self.model.config.num_attention_heads
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < n_layers:
            raise ValueError(f"layer must be an integer in [0, {n_layers})")
        if head is not None and (isinstance(head, bool) or not isinstance(head, int) or not 0 <= head < n_heads):
            raise ValueError(f"head must be None or an integer in [0, {n_heads})")
        heads = list(range(n_heads)) if head is None else [head]
        failures = {"count": 0, "reasons": {}, "case_ids": []}
        usable = []
        for case in self.cases:
            neutral, = self._prepare([(case, "neutral")], reserve=0)
            misleading, = self._prepare([(case, "incorrect_suggestion")], reserve=0)
            n_spans = self._spans(neutral, opinion_required=False)
            m_spans = self._spans(misleading, opinion_required=True)
            reason = n_spans.get("reason") or m_spans.get("reason")
            if reason:
                failures["count"] += 1
                failures["case_ids"].append(case.case_id)
                failures["reasons"][reason] = failures["reasons"].get(reason, 0) + 1
                continue
            usable.append((neutral, n_spans, misleading, m_spans))
        if not usable:
            return [{"layer": layer, "head": h, "status": "no_usable_cases", "match_failures": failures} for h in heads]

        prompts = [item[0] for item in usable] + [item[2] for item in usable]
        spans = [item[1] for item in usable] + [item[3] for item in usable]
        token_lists = [p["tokens"] for p in prompts]
        batches = self._batches(token_lists, self.attention_batch_size)
        self._reserve(len(batches), sum(len(t) for t in token_lists))
        per_prompt: list = [None] * len(prompts)
        previous_impl = self.model.config._attn_implementation
        self.model.set_attn_implementation("eager")  # output_attentions requires eager attention
        try:
            for batch in batches:
                ids, mask, positions, pads = self._inputs(token_lists, batch)
                with torch.no_grad():
                    out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                                     output_attentions=True)
                if out.attentions is None:
                    raise RuntimeError("attention_to_span needs eager attention; the model returned no attentions")
                attn = out.attentions[layer]
                for row, index in enumerate(batch):
                    pad = pads[row]
                    weights = attn[row, :, -1, :].double().cpu()  # (heads, key positions)
                    valid = weights[:, pad:].sum(dim=-1)
                    span = spans[index]
                    opinion = (weights[:, [p + pad for p in span["opinion"]]].sum(dim=-1) if span["opinion"]
                               else torch.zeros(n_heads, dtype=torch.float64))
                    source = weights[:, [p + pad for p in span["source"]]].sum(dim=-1)
                    per_prompt[index] = {"opinion": opinion, "source": source, "other": valid - opinion - source}
        finally:
            self.model.set_attn_implementation(previous_impl)
        count = len(usable)
        neutral_rows = per_prompt[:count]
        misleading_rows = per_prompt[count:]

        def mean_of(rows, key):
            return torch.stack([r[key] for r in rows]).mean(dim=0)

        n_src, n_oth = mean_of(neutral_rows, "source"), mean_of(neutral_rows, "other")
        m_op, m_src, m_oth = (mean_of(misleading_rows, k) for k in ("opinion", "source", "other"))
        results = []
        for h in heads:
            results.append({
                "layer": layer, "head": h, "status": "ok", "cases_used": count,
                "neutral": {"source_mass": float(n_src[h]), "other_mass": float(n_oth[h])},
                "misleading": {"opinion_mass": float(m_op[h]), "source_mass": float(m_src[h]),
                               "other_mass": float(m_oth[h])},
                "source_mass_change_misleading_minus_neutral": float(m_src[h] - n_src[h]),
                "match_failures": failures,
            })
        return results

    @_logged("evaluate_edits")
    def evaluate_edits(self, edits, *, limit_prompts=None, strong=False, strong_variant="trace",
                       wording="reviewer") -> dict:
        """Apply each edit set in turn, run batched greedy generation on discovery prompts, restore, score.

        Metrics use statistics.metrics: invalid JSON or abstention counts as wrong in balanced accuracy.
        Returns per-set balanced accuracies, per-class counts, and restoration hashes.
        """
        sets = _edit_sets(edits)
        plans = [self._plan_edits(edit_set) for edit_set in sets]
        requests = [(case, cond) for case in self.cases for cond in CONDITIONS]
        if limit_prompts is not None:
            if isinstance(limit_prompts, bool) or not isinstance(limit_prompts, int) or limit_prompts < 1:
                raise ValueError("limit_prompts must be a positive integer")
            requests = requests[:limit_prompts]
        prompts = self._prepare(requests, reserve=self.max_new_tokens, strong=strong,
                                strong_variant=strong_variant, wording=wording)
        token_lists = [p["tokens"] for p in prompts]
        batches = self._batches(token_lists, self.batch_size)
        self._reserve(len(batches) * len(plans), sum(len(t) for t in token_lists) * len(plans))
        results = []
        for index, plan in enumerate(plans):
            with self._applied(plan) as applied:
                rows = self._run_generation(prompts, batches)
            results.append(self._set_summary(index, plan, rows, applied))
        return {"edit_sets": results, "prompts_per_set": len(prompts), "forward_units_per_set": len(batches),
                "budget_used": self._units_used, "budget_remaining": self.budget_remaining}

    def _run_generation(self, prompts: list[dict], batches: list[list[int]]) -> list[dict]:
        rows: list = [None] * len(prompts)
        token_lists = [p["tokens"] for p in prompts]
        for batch in batches:
            ids, mask, _, _ = self._inputs(token_lists, batch)
            with torch.no_grad():
                generated = self.model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                                max_new_tokens=self.max_new_tokens, pad_token_id=self._pad_id)
            new_tokens = generated[:, ids.shape[1]:].detach().cpu()
            for row, index in enumerate(batch):
                prompt = prompts[index]
                case = prompt["case"]
                text = PREFILL + self.tokenizer.decode(new_tokens[row], skip_special_tokens=True)
                parsed, error = parse_response(text, category=case.target_category,
                                               line_count=len(case.code.splitlines()))
                predicted = parsed["verdict"] if parsed is not None else "invalid_response"
                rows[index] = {"case_id": case.case_id, "condition": prompt["condition"],
                               "oracle_verdict": case.oracle_verdict, "predicted_verdict": predicted,
                               "correct": predicted == case.oracle_verdict, "parse_error": error,
                               "generation_error": None, "input_tokens": len(prompt["tokens"]),
                               "output_tokens": int(new_tokens.shape[1]), "latency_seconds": 0.0,
                               "response_text": text[:400]}
        return rows

    def _set_summary(self, index: int, plan: list[dict], rows: list[dict], applied: dict) -> dict:
        by_condition = {cond: metrics([r for r in rows if r["condition"] == cond]) for cond in CONDITIONS}
        touched = sum(item["touched"] for item in plan)
        count_keys = ("n_vulnerable", "n_clean", "true_positive", "true_negative", "false_positive",
                      "false_negative", "invalid_count")
        return {
            "edit_set_index": index,
            "tensors": [item["tensor_name"] for item in plan],
            "touched_scalars": touched,
            "touched_fraction": touched / self.total_parameters,
            "changed_scalars": applied["changed_scalars"],
            "changed_fraction": applied["changed_scalars"] / self.total_parameters,
            "misleading_balanced_accuracy": by_condition["incorrect_suggestion"]["balanced_accuracy"],
            "neutral_balanced_accuracy": by_condition["neutral"]["balanced_accuracy"],
            "correct_suggestion_balanced_accuracy": by_condition["correct_suggestion"]["balanced_accuracy"],
            "balanced_accuracy": {c: by_condition[c]["balanced_accuracy"] for c in CONDITIONS},
            "per_class_counts": {c: {k: by_condition[c].get(k) for k in count_keys} for c in CONDITIONS},
            "restoration": {"verified": applied["restored"], "tensor_sha256_before": applied["before_sha256"],
                            "tensor_sha256_after_apply": applied["edited_sha256"]},
            "rows": rows,
        }

    @_logged("propose_random_control")
    def propose_random_control(self, patch, seed=42) -> dict:
        """Same tensors and slice counts as `patch`, random slice indices and values with matched update norm."""
        sets = _edit_sets(patch)
        if len(sets) != 1:
            raise ValueError("propose_random_control takes one edit set")
        plan = self._plan_edits(sets[0])
        target_sq = 0.0
        for item in plan:
            param, axis, index = item["param"], item["axis"], item["index"]
            original = _orient(param.detach().cpu().index_select(axis, index), axis).double()
            new = _orient(item["src"].cpu(), axis).double()
            target_sq += float(((new - original) ** 2).sum())
        target = math.sqrt(target_sq)
        if target <= 0:
            raise ValueError("The edit set has zero update norm; no matched control exists")
        gen = torch.Generator().manual_seed(int(seed))
        drawn, gauss_sq = [], 0.0
        for item in plan:
            param, axis = item["param"], item["axis"]
            k, size = item["index"].numel(), param.shape[axis]
            width = param.shape[1 - axis]
            rand_index = torch.randperm(size, generator=gen)[:k].sort().values
            base = _orient(param.detach().cpu().index_select(axis, rand_index), axis)
            g = torch.randn((k, width), generator=gen, dtype=torch.float64)
            gauss_sq += float((g ** 2).sum())
            drawn.append((item, rand_index, base, g))
        scale = target / math.sqrt(gauss_sq)
        achieved_sq, edits, per_tensor = 0.0, [], {}
        for item, rand_index, base, g in drawn:
            new = (base.double() + scale * g).to(dtype=item["param"].dtype)
            achieved_sq += float(((new.double() - base.double()) ** 2).sum())
            edits.append(Edit(tensor_name=item["tensor_name"], axis=item["axis"], indices=rand_index.tolist(),
                              values=new.double().tolist()))
            per_tensor[item["tensor_name"]] = int(rand_index.numel())
        achieved = math.sqrt(achieved_sq)
        mismatch = abs(achieved - target) / target
        return {"edits": edits, "seed": int(seed), "tensor_names": sorted(per_tensor),
                "changed_slices": per_tensor, "target_update_norm": target, "achieved_update_norm": achieved,
                "relative_norm_mismatch": mismatch, "within_1e-3": bool(mismatch <= 1e-3),
                "note": "values are drawn from a seeded Gaussian and rescaled jointly; the norm is measured after "
                        "casting to the tensor dtype, so dtype rounding is part of the reported mismatch"}

    @_logged("submit")
    def submit(self, edits) -> dict:
        """Validate one edit set against the fraction and submission limits, then persist it as a sparse patch."""
        if self.output_dir is None:
            raise RuntimeError("submit requires output_dir")
        # Count manifests on disk, not in memory, so a restarted process cannot exceed the limit.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.output_dir.glob("submission-*.json"))
        if len(existing) >= self.max_submissions:
            raise ValueError(f"submission limit reached ({self.max_submissions})")
        sets = _edit_sets(edits)
        if len(sets) != 1:
            raise ValueError("submit takes one edit set")
        plan = self._plan_edits(sets[0])
        index = len(existing) + 1
        stem = self.output_dir / f"submission-{index:02d}"
        if Path(f"{stem}.json").exists() or Path(f"{stem}.safetensors").exists():
            raise FileExistsError(f"{stem} already exists")
        original_state, edited_state = {}, {}
        for item in plan:
            param, axis = item["param"], item["axis"]
            base = param.detach().cpu().contiguous().clone()
            edited = base.clone()
            edited.index_copy_(axis, item["index"], item["src"].cpu().to(base.dtype))
            original_state[item["tensor_name"]] = base
            edited_state[item["tensor_name"]] = edited
        patch = build_patch(original_state, edited_state)
        # build_patch only sees the tensors it is given; the protocol fraction is over the whole model.
        patch.total_parameters = self.total_parameters
        manifest = save_patch(patch, stem, {
            "submission_index": index, "discovery_only": True, "edits_sha256": digest(_jsonable(sets[0])),
            "forward_units_used": self._units_used, "tool_calls_logged": self._seq,
            "max_changed_fraction": self.max_changed_fraction})
        return {**manifest, "submission_index": index, "path": f"{stem}.safetensors",
                "manifest_path": f"{stem}.json"}

    def transcript(self) -> list[dict]:
        """Ordered log of every tool call with token estimates. Read-only, not itself logged."""
        return json.loads(json.dumps(self._log))

    def save_transcript(self, path) -> Path:
        atomic_json(path, {"format": TRANSCRIPT_FORMAT, "budget": {
            "forward_pass_budget": self.forward_pass_budget, "used": self._units_used,
            "remaining": self.budget_remaining}, "entries": self.transcript()})
        return Path(path)
