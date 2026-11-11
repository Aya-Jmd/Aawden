"""No-fallback model interfaces, a labeled toy circuit, and local HF inference."""
from __future__ import annotations
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .artifacts import digest, file_hash, read_json
from .data import CLEAN, VULNERABLE
from .prompts import GenerationRequest


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    error: str | None = None
    latency_seconds: float = 0.0


class ModelAdapter(Protocol):
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]: ...
    def score_continuations(self, prefix: str, continuations: list[str]) -> list[float]: ...


class ToyCircuit:
    """Hand-constructed two-input linear circuit, NOT a language model.

    A transparent lexical feature approximates code evidence. A second feature
    carries the reviewer's claim. This deliberately planted mechanism validates
    orchestration and serialization; it provides no evidence about Cohere.
    """
    evidence_kind = "constructed_toy"
    def __init__(self, weights=(1.0, 1.25)):
        self.weights = tuple(float(x) for x in weights)
        self.metadata = {"adapter": "constructed_toy", "weights": list(self.weights), "architecture": "2-input-linear-choice",
                         "trained": False, "model_id": "aawden/constructed-toy-v1", "revision": "1",
                         "token_counts": "whitespace estimates; not LLM tokens"}

    def generate(self, requests):
        results = []
        for request in requests:
            start = time.perf_counter()
            user = request.messages[-1]["content"]
            code = user.split("<source>\n", 1)[1].split("</source>", 1)[0]
            category = re.search(r"Target category: ([a-z_]+)", user).group(1)
            if category == "sql_injection":
                clean = "?" in code or ":name" in code or ":value" in code or ".replace(" in code
            else:
                clean = '["owner_id"]' in code or '["members"]' in code
            evidence = (-1.0 if clean else 1.0) * (0.75 + int(digest(code)[:8], 16) % 70 / 100)
            opinion = re.search(r"verdict is (vulnerable|not_vulnerable_for_target)", user)
            suggestion = 0 if opinion is None else (1 if opinion.group(1) == VULNERABLE else -1)
            if "Independently trace" in request.messages[0]["content"]:
                suggestion *= .75
            score = self.weights[0] * evidence + self.weights[1] * suggestion
            verdict = VULNERABLE if score > 0 else CLEAN
            lines = [i for i, line in enumerate(code.splitlines(), 1)
                     if ("execute(" in line if category == "sql_injection" else "return " in line)]
            payload = {"verdict": verdict, "category": category, "evidence_lines": lines[:2] or [1],
                       "explanation": "Constructed toy circuit: lexical code evidence combined with a reviewer-opinion feature. This is not an LLM security assessment.",
                       "confidence": round(1 / (1 + math.exp(-min(20, abs(score)))), 4)}
            text = json.dumps(payload)
            results.append(GenerationResult(request.request_id, text, sum(len(m["content"].split()) for m in request.messages),
                                            len(text.split()), "stop", latency_seconds=time.perf_counter() - start))
        return results

    def score_continuations(self, prefix, continuations):
        raise NotImplementedError("The constructed toy has no tokenizer or language likelihoods")


def load_lora_into(model, specification, base_hashes, *, merge=False):
    """Attach a LoRA adapter, optionally folding it into dense weights.

    Protocol v3 route (b) edits the merged base: merging replaces each LowRankLinear with its base
    Linear, so tensor names stay model.layers.N.<proj>.weight and a sparse patch applies cleanly.
    Returns the metadata fields to record.
    """
    from .adaptation import load_adapter, merge_dense
    load_adapter(model, specification, base_hashes)
    metadata = {"adaptation": "LoRA", "adapter_sha256": specification["sha256"]}
    if merge:
        from .interventions import tensor_hash
        merged = merge_dense(model)
        if not merged["modules_merged"]:
            raise ValueError("merge_lora requested but no adapter modules were merged")
        model.eval()
        state = {name: tensor_hash(value) for name, value in model.state_dict().items()}
        metadata.update(adaptation="LoRA-merged-dense", merged_modules=merged["modules_merged"],
                        merged_state_sha256=digest(state))
    return metadata


class HuggingFaceAdapter:
    evidence_kind = "cohere_local_checkpoint"
    def __init__(self, config: dict):
        if not config.get("revision") or config["revision"] in ("main", "latest"):
            raise ValueError("An immutable model revision is required")
        if config.get("trust_remote_code", False):
            raise ValueError("This protocol requires trust_remote_code=false")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.config = torch, config
        torch.manual_seed(config.get("seed", 42))
        model_id = config["model_id"]
        local = Path(model_id).is_dir()
        kwargs = {"local_files_only": config.get("local_files_only", False), "trust_remote_code": False}
        if not local:
            kwargs["revision"] = config["revision"]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            dtype = getattr(torch, config.get("dtype", "bfloat16"))
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, use_safetensors=True, **kwargs)
        except OSError as exc:
            raise ValueError(f"Checkpoint load failed for {model_id}: {type(exc).__name__}. Check access and local files.") from None
        if not self.tokenizer.chat_template:
            raise ValueError("The checkpoint has no chat template; implicit prompt substitutions are prohibited")
        if self.model.config.model_type not in ("cohere", "cohere2"):
            raise ValueError("This protocol evaluates dense Cohere checkpoints only; a model change requires an explicit scope decision")
        device = config.get("device", "auto")
        device = "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.model.to(device).eval()
        self.device = device
        from .artifacts import environment
        self.metadata = {"adapter": "huggingface", "model_id": model_id, "revision": config["revision"],
                         "dtype": str(dtype), "device": device, "decoding": {"do_sample": False, "max_new_tokens": config["max_new_tokens"]},
                         "tokenizer_sha256": digest(self.tokenizer.get_vocab()), "chat_template_sha256": digest(self.tokenizer.chat_template),
                         "continuation_scoring": "sum of all continuation token log-probabilities, no length normalization, no added special tokens",
                         "environment": environment()}
        if device.startswith("cuda"):
            props = torch.cuda.get_device_properties(device)
            self.metadata["gpu"] = {"name": props.name, "memory_bytes": props.total_memory, "cuda": torch.version.cuda}
        checkpoint = Path(model_id) if local else cached_weight_directory(model_id, config["revision"])
        self.metadata["checkpoint_hashes"] = weight_file_hashes(checkpoint)
        if config.get("lora_adapter"):
            self.metadata.update(load_lora_into(self.model, config["lora_adapter"],
                                                self.metadata["checkpoint_hashes"],
                                                merge=bool(config.get("merge_lora"))))

    def chat_prefix(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _tokens(self, prefix, reserve=0):
        ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        limit = min(self.config["context_limit"], getattr(self.model.config, "max_position_embeddings", self.config["context_limit"]))
        if ids.shape[1] + reserve > limit:
            raise ValueError("Prompt exceeds context budget; code truncation is prohibited")
        return ids

    def generate(self, requests):
        outputs = []
        for request in requests:
            start = time.perf_counter()
            messages = request.messages
            prefill = ""
            if messages[-1]["role"] == "assistant":
                prefill = messages[-1]["content"]
                messages = messages[:-1]
            ids = self._tokens(self.chat_prefix(messages) + prefill, request.max_new_tokens)
            with self.torch.inference_mode():
                generated = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                                                do_sample=False, max_new_tokens=request.max_new_tokens,
                                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            new = generated[0, ids.shape[1]:]
            outputs.append(GenerationResult(request.request_id, prefill + self.tokenizer.decode(new, skip_special_tokens=True),
                                            ids.shape[1], len(new), "length" if len(new) == request.max_new_tokens else "stop",
                                            latency_seconds=time.perf_counter() - start))
        return outputs

    def score_continuations(self, prefix, continuations):
        prefix_ids = self._tokens(prefix)
        scores = []
        for continuation in continuations:
            full = self._tokens(prefix + continuation)
            length = prefix_ids.shape[1]
            if full.shape[1] <= length or not self.torch.equal(full[:, :length], prefix_ids):
                raise ValueError("Continuation crosses tokenizer boundary or is empty; choose a valid assistant prefix")
            with self.torch.inference_mode():
                logits = self.model(input_ids=full).logits.float().log_softmax(-1)
                target = full[:, length:]
                selected = logits[:, length-1:-1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scores.append(selected.sum().item())
        return scores


def cached_weight_directory(model_id: str, revision: str) -> Path:
    from huggingface_hub import try_to_load_from_cache

    for filename in ("model.safetensors", "model.safetensors.index.json"):
        path = try_to_load_from_cache(model_id, filename, revision=revision)
        if isinstance(path, str) and Path(path).is_file():
            return Path(path).parent
    raise ValueError("The pinned Hugging Face cache is missing its safetensors checkpoint")


def weight_file_hashes(checkpoint):
    single = checkpoint / "model.safetensors"
    if single.is_file():
        return {single.name: file_hash(single)}
    index = checkpoint / "model.safetensors.index.json"
    names = sorted(set(read_json(index)["weight_map"].values()))
    if not names or any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
        raise ValueError("Invalid safetensors shard index")
    return {name: file_hash(checkpoint / name) for name in [index.name, *names]}

