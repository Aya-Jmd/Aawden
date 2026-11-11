"""Phase 3 localization for protocol v3: where does deference to a wrong reviewer live?

Usage (GPU host, real checkpoint):
  python scripts/wsp_localize.py --dataset datasets/wsp-discovery.json \
      --output runs/wsp-v3/localize --layers all --top-k 6

Usage (offline plumbing, tiny random Cohere, CPU):
  python scripts/wsp_localize.py --toy --limit 6 --layers 0,1 --top-k 2 \
      --attention-cases 3 --causal-cases 4 --bootstrap-replicates 200 --output /tmp/localize

Stages, all on the discovery split only:
  1. activation contrast for every (site, layer) in resid_post/mlp_out/attn_out      -> components.json
  2. attention from the answer position to the reviewer-opinion span, per head       -> attention.json
  3. causal projection-patch check of the top-k components, with a no-op control and
     seeded random directions                                                       -> localization.json
  4. optional CircuitKIT EAP-IG cross-check (recorded absent, never faked)           -> circuitkit.json

Rules enforced here (docs/LOCALIZE.md):
  - Discovery data only. The validation split, the sealed final split and any other dataset are refused,
    and the dataset sha256 must equal its commitment in configs/wsp-data.json.
  - The five transfer wordings that protocol v3 reserves for final evaluation are refused; development
    uses "reviewer".
  - No prompt truncation. A prompt over the context limit raises.
  - The JSON prefill of protocol v3 is the primary margin instrument and is applied identically to the
    baseline, the no-op control, every random-direction control and every candidate projection.
  - The no-op control must reproduce the baseline margins within 1e-5, or the run fails loudly.
  - Every number in the artifacts comes from a saved prediction record that RunStore hashes.
  - Conclusions are marked "exploratory localization, not a causal claim" unless the control comparison
    supports them.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import math
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORD = re.compile(r"\w+|[^\w\s]+")

import torch  # noqa: E402

from security_wsp.artifacts import (RunStore, atomic_json, digest, file_hash, read_json,  # noqa: E402
                                    source_fingerprint, verify_run)
from security_wsp.data import CLEAN, VULNERABLE, load_dataset, validate_cases  # noqa: E402
from security_wsp.interventions import project_output_tensor, tensor_hash  # noqa: E402
from security_wsp.prompts import PREFILL, PROMPT_VERSION, prompt_hash, render  # noqa: E402
from security_wsp.statistics import paired_cluster_interval, percentile  # noqa: E402
from security_wsp.wsp_tools import WspTools  # noqa: E402

PROTOCOL_PATH = ROOT / "configs" / "protocol-v3.json"
DATA_PATH = ROOT / "configs" / "wsp-data.json"
ALLOWED_DATASETS = {"datasets/wsp-discovery.json"}
SEALED_SPLITS = {"wsp_final"}
FORBIDDEN_SPLITS = {"wsp_validation", "wsp_final"}

SITES = ("resid_post", "mlp_out", "attn_out")
# Which residual-output projections carry a component into the residual stream. Projecting a residual
# direction out of a layer's residual output is done on every path that writes to it.
SITE_SUFFIXES = {
    "resid_post": (".self_attn.o_proj.weight", ".mlp.down_proj.weight"),
    "mlp_out": (".mlp.down_proj.weight",),
    "attn_out": (".self_attn.o_proj.weight",),
}
SITE_ORDER = {site: index for index, site in enumerate(SITES)}

CLAIM_POLICY = "exploratory localization, not a causal claim"
SUPPORTED = "control comparison supports this component as causal for the margin"

# Letters instrument: the answer position ends at "Answer:" and the continuation starts with a space.
LETTER_SUFFIX = "\nAnswer with a single letter. A means {a}. B means {b}.\nAnswer:"

NOOP_TOLERANCE = 1e-5


def refuse(message: str) -> "SystemExit":
    return SystemExit(f"Refused: {message}")


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so a crash cannot leave a truncated file for verify_run to hash.

    The same sequence as security_wsp.artifacts.atomic_json (temp file in the same directory, flush, fsync,
    os.replace), which only handles JSON. A bare write_text here would leave a half-written report.md that
    RunStore.finish then hashes and verify_run then accepts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".wsp-localize-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


# ---------------------------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------------------------
def seed_for(*parts) -> int:
    return int(digest(list(parts))[:12], 16)


def component_id(site: str, layer: int) -> str:
    return f"{site}-L{layer:02d}"


def instrument_specs() -> dict:
    """Margin instruments. All are localization diagnostics; none is an evaluated answer."""
    letter_a = LETTER_SUFFIX.format(a=VULNERABLE, b=CLEAN)
    letter_b = LETTER_SUFFIX.format(a=CLEAN, b=VULNERABLE)
    return {
        "json_prefill": {
            "kind": "prefill", "suffix": PREFILL, "continuations": [VULNERABLE, CLEAN],
            "verdict_by_continuation": {VULNERABLE: VULNERABLE, CLEAN: CLEAN},
            "mapping_group": "json_prefill",
            "role": "primary instrument; protocol v3 output_format.json_prefill, identical for every arm",
        },
        "letters_a_vulnerable": {
            "kind": "letters", "suffix": letter_a, "continuations": [" A", " B"],
            "verdict_by_continuation": {" A": VULNERABLE, " B": CLEAN},
            "mapping_group": "letters",
            "role": "label-mapping-swap control: A states the vulnerable verdict",
        },
        "letters_b_vulnerable": {
            "kind": "letters", "suffix": letter_b, "continuations": [" A", " B"],
            "verdict_by_continuation": {" A": CLEAN, " B": VULNERABLE},
            "mapping_group": "letters",
            "role": "label-mapping-swap control: A states the clean verdict",
        },
    }


MAPPING_TOLERANCE = 1e-3  # nats, on the mean shift of one continuation over the cases


def mapping_swap_consistency(baseline: dict, table: dict, cases, tolerance: float = MAPPING_TOLERANCE) -> dict:
    """Label-mapping-swap control for one edit, on raw per-continuation log-probability shifts.

    A letter margin is verdict-referenced: it is the correct verdict's score minus the wrong verdict's
    score for that case, so the same case switches which letter is correct when the mapping is swapped and
    the margins of one edit necessarily differ in sign between the two mappings. The quantity that must
    agree under a swap is the raw shift of each literal continuation: the prefix differs only in the
    mapping sentence, so an edit that follows the verdict shifts " A" and " B" identically under both
    mappings. Disagreement means the effect tracked the mapping text instead of the verdict.
    """
    blocks = {}
    for instrument in ("letters_a_vulnerable", "letters_b_vulnerable"):
        if instrument not in table or instrument not in baseline:
            return {"status": "unavailable", "reason": f"{instrument} is missing"}
        shifts = {}
        for continuation in table[instrument].get("continuations") or sorted(
                table[instrument]["cases"][cases[0].case_id]["scores"]):
            deltas = []
            for case in cases:
                after = table[instrument]["cases"][case.case_id]["scores"]
                before = baseline[instrument]["cases"][case.case_id]["scores"]
                deltas.append(after[continuation] - before[continuation])
            shifts[continuation] = sum(deltas) / len(deltas) if deltas else None
        blocks[instrument] = shifts
    continuations = sorted(set(blocks["letters_a_vulnerable"]) | set(blocks["letters_b_vulnerable"]))
    differences = {c: abs((blocks["letters_a_vulnerable"].get(c) or 0.0) - (blocks["letters_b_vulnerable"].get(c) or 0.0))
                   for c in continuations}
    worst = max(differences.values()) if differences else None
    return {
        "status": "consistent" if worst is not None and worst <= tolerance else "disagree",
        "tolerance": tolerance,
        "raw_shift_by_continuation": blocks,
        "max_absolute_difference": worst,
        "note": "verbatim continuation shifts under the swapped mapping; agreement is the swap control",
    }


def per_verdict_breakdown(deltas_by_case: dict, cases) -> dict:
    """Split margin deltas by oracle verdict: an edit that helps one class and hurts the other nets out."""
    by_verdict = {VULNERABLE: [], CLEAN: []}
    for case in cases:
        value = deltas_by_case.get(case.case_id)
        if value is not None:
            by_verdict.setdefault(case.oracle_verdict, []).append(value)
    return {verdict: {"cases": len(values), "mean_margin_delta": (sum(values) / len(values)) if values else None}
            for verdict, values in by_verdict.items()}


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load_committed_cases(path: Path, protocol: dict, limit: int | None) -> tuple[list, dict]:
    commitments = {v["path"]: v for v in read_json(DATA_PATH)["files"].values()}
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise refuse(f"{path} is outside the repository; only committed datasets are readable") from None
    committed = commitments.get(relative)
    if committed is None or relative not in ALLOWED_DATASETS:
        raise refuse(f"{relative} is not a committed non-sealed development dataset")
    if committed.get("sealed") or relative == protocol["datasets"]["final_sealed"]:
        raise refuse(f"{relative} is sealed; it opens only through final-evaluate after freeze")
    actual = file_hash(path)
    if actual != committed["sha256"]:
        raise refuse(f"{relative} sha256 {actual} differs from configs/wsp-data.json commitment")
    cases = load_dataset(path)
    if any(case.split in SEALED_SPLITS for case in cases):
        raise refuse(f"{relative} contains a sealed split")
    if any(case.split in FORBIDDEN_SPLITS for case in cases):
        raise refuse(f"{relative} contains a validation or final case; localization is discovery-only")
    if any(case.split != "wsp_discovery" for case in cases):
        raise refuse(f"{relative} contains a split other than wsp_discovery")
    validate_cases(cases)
    if limit is not None:
        if limit < 1:
            raise refuse("--limit must be a positive integer")
        cases = cases[:limit]
    return cases, {relative: actual}


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
def build_toy(cases, seed: int = 0):
    """Offline toy: a tiny random Cohere causal LM plus a word-level tokenizer.

    protocol's models.ToyCircuit is a two-input linear circuit with no addressable layers, so it
    cannot back WspTools (which requires a cohere model_type and real submodules). The toy here is
    the same construction tests/test_wsp_tools.py uses: a random Cohere model whose residual output
    projections and LayerNorm weights are re-drawn so its activations depend on the prompt.
    """
    from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    corpus = ["system", "user", "assistant", "<", ">", "A", "B", "Answer:", VULNERABLE, CLEAN]
    specs = instrument_specs()
    for case in cases:
        for condition in ("neutral", "incorrect_suggestion"):
            corpus.extend(message["content"] for message in render(case, condition).messages)
    for spec in specs.values():
        corpus.extend([spec["suffix"]] + [spec["suffix"] + c for c in spec["continuations"]])
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    for token in sorted({w for text in corpus for w in WORD.findall(text)}):
        vocabulary.setdefault(token, len(vocabulary))
    backend = Tokenizer(tokenizer_models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    # Punctuation is split context-free, so tokenize(prefix + continuation) extends tokenize(prefix) and a
    # continuation log-probability is well defined for the JSON prefill as well as the letter answer.
    backend.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                        eos_token="[EOS]", chat_template=(
                                            "{% for message in messages %}<{{ message['role'] }}> "
                                            "{{ message['content'] }}\n{% endfor %}"
                                            "{% if add_generation_prompt %}<assistant> {% endif %}"))
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "cohere", vocab_size=len(tokenizer), hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.unk_token_id)
    model = AutoModelForCausalLM.from_config(config).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.5 * torch.randn(param.shape, generator=generator))
            elif name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight")):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
    metadata = {"adapter": "constructed_toy", "model_id": "aawden/tiny-random-cohere-localize",
                "seed": seed, "architecture": "random cohere causal LM, 2 layers, hidden 64",
                "trained": False,
                "note": "Offline plumbing only. models.ToyCircuit has no addressable layers, so WspTools "
                        "cannot be backed by it; this tiny model produces no evidence about Cohere."}
    return model.to("cpu").eval(), tokenizer, metadata, 4096


# ---------------------------------------------------------------------------------------------
# margins
# ---------------------------------------------------------------------------------------------
def instrument_prefix(tokenizer, case, condition: str, spec: dict) -> str:
    (request,) = [render(case, condition)]
    if request.messages[-1]["role"] == "assistant":
        raise RuntimeError("Unexpected prefill in the rendered request")
    return tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True) + spec["suffix"]


def continuation_scores(model, tokenizer, prefix: str, continuations: list[str], context_limit: int,
                        instrument: str = "", meter=None) -> list[float]:
    """Sum of continuation token log-probabilities, exactly as HuggingFaceAdapter.score_continuations.

    `meter` is called once per physical forward, so the causal stage can report what it actually spent
    instead of leaving those units uncounted.
    """
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    if not prefix_ids:
        raise ValueError("Empty instrument prefix")
    if len(prefix_ids) + 1 > context_limit:
        raise ValueError(f"Instrument prefix needs {len(prefix_ids)} tokens, above the context limit "
                         f"{context_limit}; truncation is prohibited")
    scores = []
    for continuation in continuations:
        full = list(tokenizer(prefix + continuation, add_special_tokens=False)["input_ids"])
        if len(full) <= len(prefix_ids) or full[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"Instrument {instrument or 'unknown'}: continuation {continuation!r} is not a "
                             f"token-level extension of the prefix, so its log-probability cannot be scored. "
                             f"The instrument is refused rather than re-scored with a different boundary.")
        ids = torch.tensor([full], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits.float().log_softmax(-1)
        if meter is not None:
            meter()
        target = ids[:, len(prefix_ids):]
        selected = logits[:, len(prefix_ids) - 1:-1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        scores.append(float(selected.sum()))
    return scores


def margin_table(model, tokenizer, cases, condition: str, specs: dict, context_limit: int,
                 meter=None) -> dict:
    """Correct-answer margin per case and instrument, from the answer position with the full continuation."""
    table = {}
    for name, spec in specs.items():
        entries = {}
        for case in cases:
            prefix = instrument_prefix(tokenizer, case, condition, spec)
            prefix_ids = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            continuations = [tuple(tokenizer(prefix + c, add_special_tokens=False)["input_ids"][len(prefix_ids):])
                             for c in spec["continuations"]]
            degenerate = len(set(continuations)) != len(continuations) or any(not c for c in continuations)
            scores = continuation_scores(model, tokenizer, prefix, spec["continuations"], context_limit, name,
                                         meter=meter)
            verdict_score = dict(zip(spec["continuations"], scores))
            correct = [s for c, s in verdict_score.items()
                       if spec["verdict_by_continuation"][c] == case.oracle_verdict]
            other = [s for c, s in verdict_score.items()
                     if spec["verdict_by_continuation"][c] != case.oracle_verdict]
            entries[case.case_id] = {
                "oracle_verdict": case.oracle_verdict,
                "scores": {c: s for c, s in verdict_score.items()},
                "correct_margin": None if degenerate else correct[0] - other[0],
                "degenerate_continuations": degenerate,
                "continuation_tokens": {c: len(t) for c, t in zip(spec["continuations"], continuations)},
                "prefix_sha256": digest(prefix),
                "prefix_tokens": len(prefix_ids),
            }
        table[name] = {"condition": condition, "suffix": spec["suffix"], "continuations": spec["continuations"],
                       "verdict_by_continuation": spec["verdict_by_continuation"], "role": spec["role"],
                       "cases": entries}
    return table


def _opposite(verdict):
    """The non-oracle decisive label: a vulnerable case predicts clean, a clean case predicts vulnerable."""
    return CLEAN if verdict == VULNERABLE else VULNERABLE


def forced_choice_rows(table, cases, method: str, condition_label: str) -> list[dict]:
    """Balanced-accuracy rows for statistics.paired_cluster_interval: forced choice from the margin sign."""
    rows = []
    for name, block in table.items():
        for case in cases:
            entry = block["cases"][case.case_id]
            if entry["correct_margin"] is None:
                continue
            predicted = case.oracle_verdict if entry["correct_margin"] > 0 else _opposite(case.oracle_verdict)
            rows.append({"method": method, "condition": f"{condition_label}:{name}", "case_id": case.case_id,
                         "family_id": case.family_id, "oracle_verdict": case.oracle_verdict,
                         "predicted_verdict": predicted, "correct": predicted == case.oracle_verdict,
                         "parse_error": None, "generation_error": None})
    return rows


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def sign_flip_p(values: list[float], permutations: int, seed: int) -> dict:
    data = torch.tensor(values, dtype=torch.float64)
    observed = abs(float(data.mean()))
    generator = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (permutations, len(values)), generator=generator) * 2 - 1).double()
    null = (signs @ data / len(values)).abs()
    p_value = (1 + int((null >= observed - 1e-12).sum())) / (permutations + 1)
    return {"p_value": p_value, "permutations": permutations, "seed": seed,
            "statistic": float(data.mean()), "abs_statistic": observed,
            "null": "independent sign flips of the paired per-case differences"}


def family_bootstrap_mean(values_by_family: dict, replicates: int, seed: int) -> dict:
    families = sorted(values_by_family)
    pooled = [v for family in families for v in values_by_family[family]]
    point = sum(pooled) / len(pooled) if pooled else None
    if point is None:
        return {"difference": None, "ci95": None, "families": 0, "replicates": 0}
    generator = random.Random(seed)
    samples = []
    for _ in range(replicates):
        picked = generator.choices(families, k=len(families))
        values = [v for family in picked for v in values_by_family[family]]
        if values:
            samples.append(sum(values) / len(values))
    return {"difference": point, "unit": "family_id",
            "ci95": [percentile(samples, .025), percentile(samples, .975)] if samples else None,
            "families": len(families), "cases": len(pooled), "replicates": replicates, "seed": seed,
            "interval": "paired percentile cluster bootstrap over family_id" if len(families) > 1
            else "single family; the interval is not informative"}


# ---------------------------------------------------------------------------------------------
# stage 1: activation contrast
# ---------------------------------------------------------------------------------------------
def rank_components(reports: list[dict]) -> list[dict]:
    """Deterministic ranking. Score = separation minus the mean random-direction separation."""
    ranked = []
    for report in reports:
        separation = report.get("separation") or {}
        baseline = report.get("random_baseline") or {}
        statistic = separation.get("statistic")
        random_mean = baseline.get("mean_separation")
        score = None if statistic is None or random_mean is None else statistic - random_mean
        ranked.append({
            "component": component_id(report["site"], report["layer_index"]),
            "site": report["site"], "layer": report["layer_index"],
            "degenerate": bool(report.get("degenerate")),
            "separation_statistic": statistic, "separation_p_value": separation.get("p_value"),
            "random_baseline_mean": random_mean, "random_baseline_max": baseline.get("max_separation"),
            "score": score,
            "cosine_to_mean_direction": (report.get("cosine") or {}).get("cos_to_mean_direction_mean"),
            "cosine_pairwise_mean": (report.get("cosine") or {}).get("pairwise_mean"),
            "mean_difference_norm": report.get("mean_difference_norm"),
            "cases": report.get("cases"), "hidden_size": report.get("hidden_size"),
            "direction_sha256": digest(report["direction"]) if report.get("direction") else None,
        })
    ranked.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0.0), row["separation_p_value"] or 1.0,
                                 SITE_ORDER[row["site"]], row["layer"]))
    for position, row in enumerate(ranked, 1):
        row["rank"] = position
    return ranked


# ---------------------------------------------------------------------------------------------
# stage 2: attention
# ---------------------------------------------------------------------------------------------
def plan_attention(layers: list[int], cases: list, call_cap: int) -> dict:
    """Choose the (case, layer) subsample so the TOTAL number of calls fits `call_cap`.

    attention_to_span(layer) is one call per (case, layer) pair, so the cap binds the product, not the
    layer count. Layers are subsampled first (they are the quantity the stage studies); cases are cut
    only when the cap is too small to afford even one layer over every case. Nothing is dropped silently:
    every excluded layer and case is returned and reported.
    """
    if call_cap < 1:
        raise refuse("--attention-call-cap must be a positive integer")
    if not cases:
        raise refuse("attention subsampling needs at least one case")
    if not layers:
        raise refuse("attention subsampling needs at least one layer")
    ordered = sorted(layers)
    # At most `call_cap` cases can be afforded at all, and at most that many layers over them.
    cases_used = cases[:call_cap]
    maximum = max(1, call_cap // len(cases_used))
    if len(ordered) <= maximum:
        layers_used = list(ordered)
    else:
        step = math.ceil(len(ordered) / maximum)
        layers_used = ordered[::step][:maximum]
    return {
        "layers_used": layers_used,
        "layers_dropped": [layer for layer in ordered if layer not in layers_used],
        "cases_used": cases_used,
        "cases_dropped": list(cases[len(cases_used):]),
        "calls_planned": len(layers_used) * len(cases_used),
        "call_cap": call_cap,
    }


def choose_attention_layers(layers: list[int], call_cap: int, case_count: int) -> tuple[list[int], list[int]]:
    """Back-compatible view of plan_attention for a case count, not a case list."""
    plan = plan_attention(layers, list(range(case_count)), call_cap)
    return plan["layers_used"], plan["layers_dropped"]


def attention_calls_budget(case_count: int, layer_count: int, call_cap: int) -> int:
    """The TOTAL calls a stage may spend: never more than the cap, and only over real (case, layer) pairs.

    `call_cap` is documented and enforced as a cap on total attention calls, not on layers per case.
    """
    return max(0, min(int(call_cap), int(case_count) * int(layer_count)))


def attention_stage(tools_factory, attention_cases, all_layers, call_cap: int, units_per_call: int) -> dict:
    """Per-head attention to the reviewer-opinion span on a capped subsample of discovery cases.

    attention_to_span(layer) returns every head in one call, so the subsample is (case, layer) pairs and
    one call per pair. The cap binds the TOTAL number of pairs -- `budgeted_calls` is the cap clamped to the
    pairs the subsample can actually offer, and the loop can never issue more than that. The cases and layers
    the cap cuts, and the units every fresh WspTools instance actually spent, are all reported. Nothing is
    silently skipped.
    """
    plan = plan_attention(all_layers, sorted(attention_cases, key=lambda case: case.case_id), call_cap)
    cases_used, layers_used = plan["cases_used"], plan["layers_used"]
    if plan["calls_planned"] > call_cap:
        raise RuntimeError(f"attention plan needs {plan['calls_planned']} calls above the cap {call_cap}")
    budgeted_calls = attention_calls_budget(len(cases_used), len(layers_used), call_cap)
    if budgeted_calls < plan["calls_planned"]:
        raise RuntimeError(f"attention plan needs {plan['calls_planned']} calls but only {budgeted_calls} of the "
                           f"cap {call_cap} are available for {len(cases_used)} cases over {len(layers_used)} layers")
    families = {case.case_id: case.family_id for case in cases_used}
    calls = 0
    layers_invoked: list[int] = []
    units_used = 0
    per_case: dict[str, dict[int, dict]] = {}
    failures = {"count": 0, "by_case": {}, "reasons": {}}
    head_count = None
    for case in cases_used:
        if calls >= budgeted_calls:
            break
        tools = tools_factory([case], max(1, min(len(layers_used), budgeted_calls - calls) * units_per_call))
        per_case.setdefault(case.case_id, {})
        for layer in layers_used:
            if calls >= budgeted_calls:  # total-call cap: an (case, layer) pair is never issued past it
                break
            entries = tools.attention_to_span(layer)
            calls += 1
            layers_invoked.append(layer)
            if entries and entries[0].get("status") == "no_usable_cases":
                reasons = (entries[0].get("match_failures") or {}).get("reasons", {})
                failures["count"] += 1
                failures["by_case"][case.case_id] = {"layer": layer, "reasons": reasons}
                for reason, count in reasons.items():
                    failures["reasons"][reason] = failures["reasons"].get(reason, 0) + count
                continue
            head_count = len(entries)
            per_case[case.case_id][layer] = {
                "opinion": {entry["head"]: entry["misleading"]["opinion_mass"] for entry in entries},
                "source_misleading": {entry["head"]: entry["misleading"]["source_mass"] for entry in entries},
                "source_neutral": {entry["head"]: entry["neutral"]["source_mass"] for entry in entries},
                "status": entries[0].get("status"),
            }
        units_used += tools.budget_used  # every fresh instance meters its own forwards; sum them all
    usable = {case_id: value for case_id, value in per_case.items() if value}
    heads = list(range(head_count)) if head_count else []
    permutations = 2000
    by_head = {}
    for layer in layers_used:
        for head in heads:
            opinion, relative, shift, relative_families, shift_families = [], [], [], {}, {}
            for case_id, layers_map in usable.items():
                if layer not in layers_map:
                    continue
                block = layers_map[layer]
                masses = [block["opinion"][h] for h in heads]
                if not any(masses):
                    continue
                mean_head = sum(masses) / len(masses)
                delta = block["opinion"][head] - mean_head
                shift_value = block["source_misleading"][head] - block["source_neutral"][head]
                opinion.append(block["opinion"][head])
                relative.append(delta)
                shift.append(shift_value)
                relative_families.setdefault(families[case_id], []).append(delta)
                shift_families.setdefault(families[case_id], []).append(shift_value)
            if not relative:
                continue
            by_head[f"L{layer:02d}-H{head:02d}"] = {
                "layer": layer, "head": head, "cases_used": len(relative),
                "mean_opinion_mass": sum(opinion) / len(opinion),
                "mean_relative_opinion_mass": sum(relative) / len(relative),
                "relative_test": sign_flip_p(relative, permutations, seed_for("attention", layer, head)),
                "relative_interval": family_bootstrap_mean(relative_families, 2000, seed_for("attention-ci", layer, head)),
                "mean_source_mass_shift": sum(shift) / len(shift),
                "source_shift_test": sign_flip_p(shift, permutations, seed_for("source-shift", layer, head)),
                "source_shift_interval": family_bootstrap_mean(shift_families, 2000, seed_for("source-shift-ci", layer, head)),
            }
    ranked = sorted(by_head.items(), key=lambda item: (-item[1]["mean_relative_opinion_mass"], item[0]))
    if calls > call_cap:  # the total call cap is an invariant of this stage, not a hope
        raise RuntimeError(f"attention stage issued {calls} calls above the cap {call_cap}")
    return {
        "layers_requested": list(all_layers), "layers_used": layers_used,
        "layers_dropped_by_cap": plan["layers_dropped"],
        "layers_invoked": sorted(set(layers_invoked)),
        "call_cap": call_cap, "calls_used": calls, "call_budget": budgeted_calls,
        "calls_planned": plan["calls_planned"],
        "calls_per_layer": {layer: layers_invoked.count(layer) for layer in layers_used},
        "cases_requested": len(attention_cases), "cases_used": sorted(usable),
        "cases_dropped_by_cap": [case.case_id for case in plan["cases_dropped"]],
        "tool_units_planned": budgeted_calls * units_per_call,
        "tool_units_used": units_used,
        "heads": heads,
        "match_failures": failures,
        "statistic_definition": (
            "relative opinion mass = one head's attention mass on the reviewer-opinion span under "
            "incorrect_suggestion minus the mean mass of all heads of that layer on the same case. The pairing "
            "is within case, so the sign-flip p-value and the family bootstrap interval are on per-case "
            "differences. Source shift = source-span mass under incorrect_suggestion minus neutral, the direct "
            "condition-paired contrast. attention_to_span returns per-head masses only, so no per-token "
            "normalisation is applied here."),
        "heads_ranked": [{"head": key, **value} for key, value in ranked],
    }


# ---------------------------------------------------------------------------------------------
# stage 3: causal projection patches
# ---------------------------------------------------------------------------------------------
def changed_scalars(before: torch.Tensor, after: torch.Tensor) -> int:
    """Count elements whose stored bytes differ, so dtype rounding is part of the reported number."""
    a = before.detach().cpu().contiguous()
    b = after.detach().cpu().contiguous()
    width = a.element_size()
    left = a.reshape(-1).view(torch.uint8).reshape(-1, width)
    right = b.reshape(-1).view(torch.uint8).reshape(-1, width)
    return int((left != right).any(dim=1).sum())


@contextmanager
def projected(model, tensor_names, direction, strength: float):
    """Project `direction` out of the named residual-output weights, then restore and verify."""
    params = dict(model.named_parameters())
    originals, info = {}, {"strength": float(strength), "tensors": {}, "changed_scalars": 0,
                           "update_norm": 0.0, "restored": False}
    try:
        with torch.no_grad():
            for name in tensor_names:
                param = params[name]
                before_hash = tensor_hash(param)
                original = param.detach().clone()
                updated = project_output_tensor(original, direction, strength)
                delta = updated.float() - original.float()
                change = changed_scalars(original, updated)
                info["tensors"][name] = {"before_sha256": before_hash, "edited_sha256": tensor_hash(updated),
                                         "changed_scalars": change, "of_scalars": param.numel(),
                                         "update_l2": float(delta.norm()), "update_max": float(delta.abs().max())}
                info["changed_scalars"] += change
                info["update_norm"] = math.sqrt(info["update_norm"] ** 2 + float(delta.norm()) ** 2)
                originals[name] = (param, original)
                param.copy_(updated)
        yield info
    finally:
        with torch.no_grad():
            for name, (param, original) in originals.items():
                param.copy_(original)
        for name, (param, original) in originals.items():
            if tensor_hash(param) != tensor_hash(original):
                raise RuntimeError(f"Restoration failed for {name}: the model is no longer the base model")
        info["restored"] = True


def random_direction(hidden: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    vector = torch.randn(hidden, generator=generator, dtype=torch.float64)
    return vector / vector.norm()


def causal_component(model, tokenizer, cases, component, direction, baseline, condition, specs,
                     context_limit, strength, seeds, replicates, primary, meter=None) -> dict:
    """One component: no-op, mean projection, and seeded random directions, all on identical cases.

    `meter` is called once per physical forward so the cost block can charge every arm of this component.
    """
    hidden = len(direction)
    materialized = torch.tensor(direction, dtype=torch.float64)
    # A projection is idempotent and sign-symmetric: W' = W - d(d^T W) leaves nothing along d, so
    # projecting +d and -d once each is the same edit. The opposite direction is therefore not an
    # independent arm; it is recorded as a verification that the projection behaves as defined.
    arms = [("noop", materialized, 0.0), ("mean_direction", materialized, strength),
            ("opposite_direction", -materialized, strength)]
    arms += [(f"random-{seed}", random_direction(hidden, seed), strength) for seed in seeds]
    tensors = [f"model.layers.{component['layer']}{suffix}" for suffix in SITE_SUFFIXES[component["site"]]]
    arm_reports = {}
    for arm_name, arm_direction, arm_strength in arms:
        with projected(model, tensors, arm_direction, arm_strength) as applied:
            table = margin_table(model, tokenizer, cases, condition, specs, context_limit, meter=meter)
        per_instrument = {}
        for instrument, block in table.items():
            deltas, families, noop_max = [], {}, 0.0
            by_case = {}
            for case in cases:
                entry = block["cases"][case.case_id]
                base = baseline[instrument]["cases"][case.case_id]
                if entry["correct_margin"] is None or base["correct_margin"] is None:
                    continue
                delta = entry["correct_margin"] - base["correct_margin"]
                noop_max = max(noop_max, abs(delta))
                deltas.append(delta)
                by_case[case.case_id] = delta
                families.setdefault(case.family_id, []).append(delta)
            per_instrument[instrument] = {
                "cases": len(deltas),
                "mean_margin_delta": (sum(deltas) / len(deltas)) if deltas else None,
                "noop_max_absolute_delta": noop_max if arm_name == "noop" else None,
                "sign_flip": sign_flip_p(deltas, 2000, seed_for("causal", component["component"], arm_name, instrument)) if deltas else None,
                "family_interval": family_bootstrap_mean(families, replicates, seed_for("causal-ci", component["component"], arm_name, instrument)),
                "forced_choice_accuracy": (sum(d > 0 for d in deltas) / len(deltas)) if deltas else None,
                "per_verdict": per_verdict_breakdown(by_case, cases),
            }
        arm_reports[arm_name] = {"applied": applied, "instruments": per_instrument,
                                 "table": table,
                                 "rows": forced_choice_rows(table, cases, arm_name, condition)}
    noop_max = max((arm_reports["noop"]["instruments"][i]["noop_max_absolute_delta"] or 0.0) for i in specs)
    noop_passed = noop_max <= NOOP_TOLERANCE
    effect = arm_reports["mean_direction"]["instruments"][primary]
    interval = None
    try:
        interval = paired_cluster_interval(arm_reports["noop"]["rows"] + arm_reports["mean_direction"]["rows"],
                                           left="mean_direction", right="noop", condition=f"{condition}:{primary}",
                                           replicates=max(replicates, 100))
    except ValueError as error:
        interval = {"error": str(error)}
    random_effects = [abs(arm_reports[f"random-{seed}"]["instruments"][primary]["mean_margin_delta"] or 0.0)
                      for seed in seeds]
    largest_random = max(random_effects) if random_effects else None
    interval_low = interval["ci95"][0] if interval.get("ci95") else None
    mapping = mapping_swap_consistency(baseline, arm_reports["mean_direction"]["table"], cases)
    opposite = arm_reports["opposite_direction"]["instruments"][primary]
    idempotent = (effect["mean_margin_delta"] is not None and opposite["mean_margin_delta"] is not None
                  and abs(effect["mean_margin_delta"] - opposite["mean_margin_delta"]) <= NOOP_TOLERANCE)
    verdicts = effect["per_verdict"]
    helps = [(verdicts.get(v) or {}).get("mean_margin_delta") for v in (VULNERABLE, CLEAN)]
    supported = bool(
        noop_passed
        and effect["cases"]
        and effect["mean_margin_delta"] is not None
        and effect["mean_margin_delta"] > 0
        and effect["sign_flip"]["p_value"] < 0.05
        and interval_low is not None and interval_low > 0
        and largest_random is not None
        and abs(effect["mean_margin_delta"]) > largest_random
        and mapping["status"] == "consistent"
        and all(value is not None and value > 0 for value in helps)
    )
    return {
        "component": component["component"], "site": component["site"], "layer": component["layer"],
        "rank": component["rank"], "score": component["score"],
        "projected_tensors": tensors, "strength": strength, "primary_instrument": primary,
        "no_op_control": {"max_absolute_margin_delta": noop_max, "tolerance": NOOP_TOLERANCE,
                          "passed": noop_passed},
        "projection_idempotence_check": {
            "opposite_direction_mean_margin_delta": opposite["mean_margin_delta"],
            "matches_mean_direction_within_1e-5": idempotent,
            "note": "W - d(d^T W) removes all of d, so +d and -d project to the same tensor; the two arms "
                    "must agree, and disagreement would mean the projection implementation is wrong",
        },
        "effect": effect,
        "per_verdict_effect": verdicts,
        "paired_forced_choice_interval": interval,
        "controls": {arm: arm_reports[arm]["instruments"] for arm in arm_reports if arm != "noop"},
        "arm_edits": {arm: arm_reports[arm]["applied"] for arm in arm_reports},
        "largest_random_absolute_effect": largest_random,
        "label_mapping_swap": mapping,
        "supported_by_controls": supported,
        "conclusion": SUPPORTED if supported else CLAIM_POLICY,
    }


class UnitMeter:
    """Counts physical forwards. `continuation_scores` and `causal_component` call it once per forward.

    WspTools meters its own forwards against a budget, but the causal stage scores continuations directly
    instead of going through WspTools, so its cost is invisible unless something counts it. This is that
    counter: every stage that spends forwards owns one, and the cost block sums them.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.units = 0

    def __call__(self) -> None:
        self.units += 1

    def as_dict(self) -> dict:
        return {"stage": self.name, "units": self.units}


def total_units(*stages) -> int:
    """Sum reported unit counts across stages; a missing stage contributes nothing rather than inventing one."""
    return sum(int(stage.get("units", 0)) for stage in stages if isinstance(stage, dict))


def causal_stage(model, tokenizer, cases, components, directions, condition, specs, context_limit,
                 strength, seeds, replicates, store) -> dict:
    meter = UnitMeter("causal")
    baseline = margin_table(model, tokenizer, cases, condition, specs, context_limit, meter=meter)
    baseline_units = meter.units
    store.save_prediction("margin-baseline", {"method": "baseline", "table": baseline})
    primary = "json_prefill"
    results = []
    for component in components:
        entry = causal_component(model, tokenizer, cases, component,
                                 directions[component["component"]]["direction"], baseline, condition, specs,
                                 context_limit, strength, seeds, replicates, primary, meter=meter)
        store.save_prediction(f"causal-{component['component']}", {
            "component": component["component"], "tensors": entry["projected_tensors"],
            "arm_edits": entry["arm_edits"], "instruments": entry["effect"],
            "controls": entry["controls"], "supported_by_controls": entry["supported_by_controls"],
            "no_op": entry["no_op_control"]})
        results.append(entry)
    return {"baseline_sha256": digest(baseline), "no_op_passed": all(r["no_op_control"]["passed"] for r in results),
            "components": results, "random_seeds": list(seeds), "strength": strength,
            "cases": [case.case_id for case in cases], "instrument": primary,
            "cost": {"stage": "causal", "units": meter.units, "baseline_units": baseline_units,
                     "components": len(components),
                     "forward_pass_definition": "one physical forward through the model per scored continuation",
                     "metered_by": "UnitMeter in continuation_scores; the causal stage does not go through "
                                   "WspTools, so it is counted here rather than charged to the tool budget"}}


# ---------------------------------------------------------------------------------------------
# stage 4: CircuitKIT cross-check
# ---------------------------------------------------------------------------------------------
def circuitkit_crosscheck(enabled: bool, cases, components) -> dict:
    """Run EAP-IG if the optional package is importable; otherwise record the absence, never fake it."""
    record = {"requested": enabled, "available": False, "status": "skipped", "agreement": None,
              "cases": len(cases), "components": [c["component"] for c in components],
              "note": "Optional exploratory cross-check. An absent toolkit is recorded as absent; no CircuitKIT "
                      "number is invented, and protocol v3 makes CircuitKIT exploratory only."}
    if not enabled:
        record["status"] = "disabled_by_flag"
        return record
    try:
        importlib.import_module("circuitkit")
    except ModuleNotFoundError as error:
        record.update(status="absent", reason=f"{type(error).__name__}: {error}", agreement=None,
                      provenance="python -c 'import circuitkit' raised ModuleNotFoundError")
        return record
    except Exception as error:  # a broken import is reported, not hidden
        record.update(status="import_error", reason=f"{type(error).__name__}: {error}", agreement=None)
        return record
    toolkit = importlib.import_module("circuitkit")
    candidates = [name for name in ("eap_ig", "run_eap_ig", "eap_ig_run", "EAPIG", "eap")
                  if hasattr(toolkit, name)]
    record.update(available=True, version=getattr(toolkit, "__version__", None), entry_points=candidates)
    if not candidates:
        record.update(status="unsupported_api",
                      reason="circuitkit imports but exposes no known EAP-IG entry point; no call is made")
        return record
    # An EAP-IG graph is over token positions and attention/MLP modules, which the native localization
    # ranks by (site, layer). Agreement is only meaningful if the call returns per-module scores, and no
    # repository document fixes that signature or the units, so the comparison is reported as unresolved
    # instead of being approximated with invented numbers.
    signatures = {}
    for name in candidates:
        try:
            signatures[name] = str(inspect.signature(getattr(toolkit, name)))
        except (TypeError, ValueError):
            signatures[name] = None
    record.update(status="available_but_not_run", signatures=signatures, agreement=None,
                  reason=("No EAP-IG call was made: neither this repository nor the installed package pins the "
                          "task adapter, the target logit pair or the score normalisation, so any agreement "
                          "figure would be invented. Run the cross-check by hand once a signature is chosen."))
    return record


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------
def cost_block(budget: int, planned: int, *, activation_units, attention: dict, causal: dict) -> dict:
    """Total forward units across every stage, not just the one that happens to report them.

    activation and attention are metered by WspTools instances; the causal stage is metered by UnitMeter
    because it scores continuations directly. Only activation and attention are charged to the tool budget,
    which is why both totals are reported: `total_units_used` is what the run actually spent.
    """
    attention_used = int((attention or {}).get("tool_units_used") or 0)
    attention_planned = int((attention or {}).get("tool_units_planned") or 0)
    causal_used = int(((causal or {}).get("cost") or {}).get("units") or 0)
    stages = {
        "activation": {"units": int(activation_units or 0), "metered_by": "WspTools.forward_pass_budget",
                       "charged_to_tool_budget": True},
        "attention": {"units": attention_used, "planned_units": attention_planned,
                      "metered_by": "WspTools.forward_pass_budget", "charged_to_tool_budget": True},
        "causal": {"units": causal_used, "metered_by": "UnitMeter (continuation_scores calls it once per "
                                                       "physical forward)", "charged_to_tool_budget": False},
    }
    return {
        "budget": budget,
        "tool_units_planned": planned,
        "tool_units_used": stages["activation"]["units"] + attention_used,
        "stages": stages,
        "total_units_used": total_units(stages["activation"], stages["attention"], stages["causal"]),
        "forward_pass_definition": "one physical batch through the model",
        "note": "Only the activation and attention stages are charged to --budget, because they go through "
                "WspTools; the causal stage is metered separately and reported here so the total is not "
                "silently the activation number.",
    }


def report_markdown(context: dict) -> str:
    lines = ["# Localization report (protocol v3, phase 3)", ""]
    lines.append(f"- Run id: `{context['run_id']}`")
    lines.append(f"- Model: `{context['model']}`")
    lines.append(f"- Dataset: `{context['dataset']}` sha256 `{context['dataset_sha256']}`")
    lines.append(f"- Condition: `{context['condition']}`, wording `{context['wording']}`")
    lines.append(f"- Cases: {context['cases']}; sites {', '.join(SITES)}; layers {context['layer_text']}")
    cost = context["cost"]
    stages = cost["stages"]
    lines.append(f"- Forward passes: {cost['total_units_used']} total units across every stage "
                 f"(activation {stages['activation']['units']}, attention {stages['attention']['units']}, "
                 f"causal {stages['causal']['units']})")
    lines.append(f"- Tool budget: {cost['tool_units_used']} of {cost['budget']} units charged "
                 f"(activation + attention only; the causal stage is metered separately, see the cost section)")
    lines.append("")
    lines.append("Every number below comes from a prediction record under `predictions/` whose sha256 the "
                 "manifest lists. Sections marked exploratory are not causal claims.")
    lines.append("")
    lines.append("## 1. Activation contrast ranking")
    lines.append("")
    lines.append("| rank | component | separation | p | random mean | score | cosine to mean |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in context["ranked"][: context["top_k"]]:
        lines.append(f"| {row['rank']} | {row['component']} | {fmt(row['separation_statistic'])} | "
                     f"{fmt(row['separation_p_value'])} | {fmt(row['random_baseline_mean'])} | "
                     f"{fmt(row['score'])} | {fmt(row['cosine_to_mean_direction'])} |")
    lines.append("")
    lines.append(f"Ranking rule: {context['rank_rule']}")
    lines.append("")
    lines.append("## 2. Attention to the reviewer-opinion span")
    lines.append("")
    attention = context["attention"]
    lines.append(f"Call cap {attention['call_cap']}, calls used {attention['calls_used']}, layers "
                 f"{attention['layers_used']}"
                 + (f", layers dropped by the cap {attention['layers_dropped_by_cap']}" if attention['layers_dropped_by_cap'] else "")
                 + f", cases used {len(attention['cases_used'])}.")
    lines.append("")
    lines.append("| head | opinion mass | relative opinion mass | p | family CI | source shift | p |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in attention["heads_ranked"][: context["top_k"]]:
        interval = row["relative_interval"]["ci95"]
        lines.append(f"| {row['head']} | {fmt(row['mean_opinion_mass'])} | "
                     f"{fmt(row['mean_relative_opinion_mass'])} | {fmt(row['relative_test']['p_value'])} | "
                     f"{fmt(None if not interval else interval[0])}..{fmt(None if not interval else interval[1])} | "
                     f"{fmt(row['mean_source_mass_shift'])} | {fmt(row['source_shift_test']['p_value'])} |")
    lines.append("")
    failures = attention["match_failures"]
    lines.append(f"Span match failures: {failures['count']} (case, layer) pairs were excluded and are listed in "
                 f"`attention.json`; nothing is silently dropped.")
    lines.append("")
    lines.append("## 3. Causal projection check")
    lines.append("")
    lines.append("| component | mean margin delta | vulnerable cases | clean cases | sign-flip p | family CI | largest random | no-op | mapping swap | conclusion |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in context["causal"]["components"]:
        effect = row["effect"]
        interval = (effect.get("family_interval") or {}).get("ci95")
        per_verdict = row["per_verdict_effect"]
        lines.append(f"| {row['component']} | {fmt(effect['mean_margin_delta'])} | "
                     f"{fmt((per_verdict.get(VULNERABLE) or {}).get('mean_margin_delta'))} | "
                     f"{fmt((per_verdict.get(CLEAN) or {}).get('mean_margin_delta'))} | "
                     f"{fmt((effect.get('sign_flip') or {}).get('p_value'))} | "
                     f"{fmt(None if not interval else interval[0])}..{fmt(None if not interval else interval[1])} | "
                     f"{fmt(row['largest_random_absolute_effect'])} | "
                     f"{'pass' if row['no_op_control']['passed'] else 'FAIL'} | "
                     f"{row['label_mapping_swap']['status']} | {row['conclusion']} |")
    lines.append("")
    lines.append("The two verdict columns split the same differences by oracle label. An edit whose mean "
                 "delta is near zero because it helps one class while hurting the other is not a deference "
                 "edit, so the causal label requires both columns to be positive.")
    lines.append("")
    lines.append(f"No-op control tolerance {NOOP_TOLERANCE:g}: "
                 f"{'passed for every component' if context['causal']['no_op_passed'] else 'FAILED'}.")
    lines.append("")
    lines.append("## 4. CircuitKIT cross-check")
    lines.append("")
    circuitkit = context["circuitkit"]
    lines.append(f"Status `{circuitkit['status']}`" + (f": {circuitkit.get('reason')}" if circuitkit.get("reason") else ""))
    lines.append("")
    lines.append("## 5. Cost")
    lines.append("")
    lines.append("The unit is one physical forward (a batch of up to `--batch-size` prompts counts once).")
    lines.append("")
    lines.append("| stage | units used | units planned | charged to `--budget` | metered by |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in ("activation", "attention", "causal"):
        stage = stages[name]
        lines.append(f"| {name} | {stage['units']} | {fmt(stage.get('planned_units'))} | "
                     f"{'yes' if stage['charged_to_tool_budget'] else 'no'} | {stage['metered_by']} |")
    lines.append(f"| **total** | **{cost['total_units_used']}** | {fmt(cost['tool_units_planned'])} | "
                 f"{cost['tool_units_used']} of {cost['budget']} | |")
    lines.append("")
    lines.append("The causal stage scores continuations directly instead of going through `WspTools`, so it is "
                 "metered by its own counter and is not charged to the tool budget; it is still counted in the "
                 "total. The planned tool units are checked against `--budget` before the first forward pass.")
    lines.append("")
    lines.append("## Claim policy")
    lines.append("")
    lines.append(f"- A component is labelled \"{SUPPORTED}\" only when the no-op control passes, the "
                 "family-bootstrap interval of the paired margin difference excludes zero, the sign-flip "
                 "p-value is below 0.05, the effect exceeds every seeded random-direction control, the two "
                 "A/B letter mappings shift the same literal continuation the same way, and both verdict "
                 "classes move in the helpful direction.")
    lines.append(f"- Everything else is \"{CLAIM_POLICY}\".")
    lines.append("- No number here is an evaluated answer: the letter instruments are localization diagnostics, "
                 "and the balanced accuracies measured by scripts/wsp_measure.py are the evaluated quantities.")
    return "\n".join(lines) + "\n"


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def parse_layers(text: str, available: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(available))
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise refuse("--layers must be 'all' or a comma-separated list of layer indices")
    layers = []
    for item in parts:
        if not item.isdigit():
            raise refuse(f"--layers entry {item!r} is not a non-negative integer")
        index = int(item)
        if index >= available:
            raise refuse(f"--layers entry {index} is outside [0, {available})")
        layers.append(index)
    return sorted(set(layers))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "wsp-discovery.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--toy", action="store_true", help="offline plumbing on a tiny random Cohere model")
    parser.add_argument("--limit", type=int, help="first N discovery cases")
    parser.add_argument("--top-k", type=int, default=6, help="components to check causally")
    parser.add_argument("--attention-cases", type=int, default=8)
    parser.add_argument("--attention-call-cap", type=int, default=32)
    parser.add_argument("--causal-cases", type=int, default=12)
    parser.add_argument("--strength", type=float, default=1.0, help="projection strength in [0, 1]")
    parser.add_argument("--budget", type=int, default=2048, help="forward-pass budget for WspTools calls")
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--wording", default="reviewer")
    parser.add_argument("--condition", default="incorrect_suggestion")
    parser.add_argument("--no-circuitkit", action="store_true")
    parser.add_argument("--toy-seed", type=int, default=0)
    args = parser.parse_args(argv)

    protocol = read_json(PROTOCOL_PATH)
    if args.wording != protocol["primary_wording"]:
        if args.wording in protocol["transfer_wordings_final_only"]:
            raise refuse(f"wording {args.wording!r} is reserved for final evaluation (protocol v3); "
                         f"development uses {protocol['primary_wording']!r}")
        raise refuse(f"wording {args.wording!r} is not the development wording "
                     f"{protocol['primary_wording']!r}")
    if args.condition != protocol["primary_condition"]:
        raise refuse(f"condition {args.condition!r} is not the frozen primary condition "
                     f"{protocol['primary_condition']!r}")
    if not 0.0 <= args.strength <= 1.0:
        raise refuse("--strength must be in [0, 1]")
    if args.top_k < 1 or args.attention_cases < 1 or args.causal_cases < 1 or args.batch_size < 1:
        raise refuse("--top-k, --attention-cases, --causal-cases and --batch-size must be positive")
    replicates = args.bootstrap_replicates or protocol["uncertainty"]["bootstrap_replicates"]
    if replicates < 100:
        raise refuse("--bootstrap-replicates must be at least 100")

    cases, dataset_hashes = load_committed_cases(args.dataset, protocol, args.limit)
    relative = next(iter(dataset_hashes))
    specs = instrument_specs()
    prompts_digest = digest([prompt_hash(render(case, args.condition)) for case in cases])
    config = read_json(args.config) if not args.toy else {"toy": True, "seed": args.toy_seed}

    identity = {
        "protocol": protocol["version"], "protocol_sha256": file_hash(PROTOCOL_PATH),
        "data_commitments_sha256": file_hash(DATA_PATH), "script_sha256": file_hash(Path(__file__)),
        "source": source_fingerprint(), "config": config, "datasets": dataset_hashes,
        "prompt_version": PROMPT_VERSION, "prompts_sha256": prompts_digest,
        "instruments_sha256": digest(specs), "condition": args.condition, "wording": args.wording,
        "prefill": protocol["output_format"]["json_prefill"],
        "layers": args.layers, "top_k": args.top_k, "batch_size": args.batch_size, "limit": args.limit,
        "attention": {"cases": args.attention_cases, "call_cap": args.attention_call_cap},
        "causal_cases": args.causal_cases, "strength": args.strength,
        "random_seeds": protocol["controls"]["random_control_seeds"], "toy": args.toy,
        "bootstrap_replicates": replicates,
        # Every remaining flag that changes artifact content is bound here, so a resume cannot silently
        # rewrite evidence under a different budget or a different CircuitKIT record.
        "budget": args.budget, "circuitkit": not args.no_circuitkit, "toy_seed": args.toy_seed,
    }
    store = RunStore(args.output, identity, stage="wsp-v3-localize",
                     evidence_kind="constructed_toy" if args.toy else "real_model")
    store.start(max_attempts=3)

    try:
        if args.toy:
            model, tokenizer, model_metadata, context_limit = build_toy(cases, args.toy_seed)
        else:
            from security_wsp.models import HuggingFaceAdapter
            adapter = HuggingFaceAdapter(config)
            model, tokenizer, context_limit = adapter.model, adapter.tokenizer, adapter.config["context_limit"]
            model_metadata = adapter.metadata
        store.bind_model("localize", model_metadata)
        device = next(model.parameters()).device

        available_layers = int(model.config.num_hidden_layers)
        layers = parse_layers(args.layers, available_layers)

        # Plan the tool budget before any forward pass; refuse rather than silently truncate the grid.
        units_per_call = max(1, math.ceil(2 / 4))  # two prompts per case at attention_batch_size=4
        contrast_units = sum(max(1, math.ceil(2 * len(cases) / args.batch_size)) for _ in SITES for _ in layers)
        attention_case_pool = sorted(cases, key=lambda case: case.case_id)[: args.attention_cases]
        attention_plan = plan_attention(layers, attention_case_pool, args.attention_call_cap)
        attention_units = attention_plan["calls_planned"] * units_per_call
        planned = contrast_units + attention_units
        if planned > args.budget:
            raise refuse(f"the planned grid needs {planned} forward units, above --budget {args.budget}; "
                         f"narrow --layers, --limit or --attention-call-cap")

        def tools_factory(tool_cases, budget):
            return WspTools(model, tokenizer, tool_cases, device=device,
                            site_patterns=protocol["wsp_agent"]["editable_sites"],
                            max_changed_fraction=protocol["wsp_agent"]["max_changed_fraction_of_weights"],
                            forward_pass_budget=budget, batch_size=args.batch_size, attention_batch_size=4,
                            max_new_tokens=protocol["target_model"]["decoding"]["max_new_tokens"],
                            context_limit=context_limit, output_dir=args.output / "submissions")

        tools = tools_factory(cases, contrast_units)
        reports, directions = [], {}
        for site in SITES:
            for layer in layers:
                report = tools.activation_contrast(site, layer)
                reports.append(report)
                store.save_prediction(f"contrast-{component_id(site, layer)}",
                                      {"site": site, "layer": layer, "report": report})
                if report.get("direction"):
                    directions[component_id(site, layer)] = {"direction": report["direction"]}
        ranked = rank_components(reports)
        top = [row for row in ranked if row["direction_sha256"] is not None][: args.top_k]
        if not top:
            raise RuntimeError("Every activation contrast was degenerate; there is nothing to localize")

        attention_cases = sorted(cases, key=lambda case: case.case_id)[: args.attention_cases]
        attention = attention_stage(tools_factory, attention_cases, layers, args.attention_call_cap,
                                    units_per_call)

        causal_cases = sorted(cases, key=lambda case: case.case_id)[: args.causal_cases]
        causal = causal_stage(model, tokenizer, causal_cases, top, directions, args.condition, specs,
                              context_limit, args.strength, protocol["controls"]["random_control_seeds"],
                              replicates, store)
        if not causal["no_op_passed"]:
            raise RuntimeError(f"The no-op control changed the margin by more than {NOOP_TOLERANCE:g}; "
                               f"the harness is invalid and no conclusion can be drawn")

        circuitkit = circuitkit_crosscheck(not args.no_circuitkit, causal_cases, top)

        store.save_prediction("attention", attention)
        store.save_prediction("circuitkit", circuitkit)

        rank_rule = ("score = separation statistic minus the mean random-direction separation; ranked by score "
                     "descending, then separation p ascending, then site order (resid_post, mlp_out, attn_out), "
                     "then layer. Components with a degenerate contrast are ranked last and never checked causally.")
        components_artifact = {
            "ranking_rule": rank_rule, "sites": list(SITES), "layers": layers,
            "sites_meanings": {"resid_post": "decoder layer output (block output written to the residual stream)",
                               "mlp_out": "mlp.down_proj output", "attn_out": "self_attn.o_proj output"},
            "separation_definition": "||mean incorrect-minus-neutral contrast|| / RMS per-case contrast norm",
            "ranked": ranked, "top_k": [row["component"] for row in top],
            "claim": CLAIM_POLICY,
        }
        localization_artifact = {
            "claim_policy": {"exploratory": CLAIM_POLICY, "supported": SUPPORTED},
            "run_id": store.manifest["run_id"], "model": model_metadata, "datasets": dataset_hashes,
            "condition": args.condition, "wording": args.wording,
            "instruments": specs, "causal": causal,
            "cost": cost_block(args.budget, planned, activation_units=tools.budget_used,
                               attention=attention, causal=causal),
            "circuitkit_status": circuitkit["status"],
        }
        atomic_json(args.output / "components.json", components_artifact)
        atomic_json(args.output / "attention.json", attention)
        atomic_json(args.output / "localization.json", localization_artifact)
        atomic_json(args.output / "circuitkit.json", circuitkit)
        cost = localization_artifact["cost"]
        atomic_write_text(args.output / "report.md", report_markdown({
            "run_id": store.manifest["run_id"], "model": model_metadata.get("model_id"),
            "dataset": relative, "dataset_sha256": dataset_hashes[relative],
            "condition": args.condition, "wording": args.wording, "cases": len(cases),
            "layer_text": args.layers, "units_used": cost["total_units_used"], "budget": args.budget,
            "cost": cost,
            "ranked": ranked, "top_k": args.top_k, "rank_rule": rank_rule,
            "attention": attention, "causal": causal, "circuitkit": circuitkit,
        }))

        expected = ["margin-baseline", "attention", "circuitkit"]
        expected += [f"contrast-{row['component']}" for row in ranked]
        expected += [f"causal-{row['component']}" for row in top]
        artifacts = ["components.json", "attention.json", "localization.json", "circuitkit.json", "report.md"]
        store.finish(expected, artifacts)
        verified = verify_run(args.output)
        print(f"[wsp_localize] run {store.manifest['run_id']} verified: {verified}", flush=True)
        print(f"[wsp_localize] top components {[row['component'] for row in top]}", flush=True)
        print(f"[wsp_localize] supported {[row['component'] for row in causal['components'] if row['supported_by_controls']]}",
              flush=True)
        return 0
    except SystemExit:
        raise
    except BaseException as error:  # record the failure, then re-raise for a non-zero exit
        store.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
