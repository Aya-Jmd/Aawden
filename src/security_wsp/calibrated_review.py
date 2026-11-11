"""Calibrate a first-token logit bias on exposed cases before strict Aya generation."""
from dataclasses import replace
import math

from .adapter_review import evaluate
from .artifacts import atomic_json, read_json, source_fingerprint
from .data import CLEAN, VULNERABLE, validate_cases
from .rigorous_review import checked_dataset, error_report, rigorous_request


VERSION = "aawden-rigorous-review-v7"
PREFILL = '{"verdict":"'
METHOD = "calibrated"
BIAS_GRID = (-3.0, -1.5, -.75, 0.0, .75, 1.5, 3.0)


def calibrated_request(case, routing):
    request = rigorous_request(case, routing[case.target_category])
    if case.target_category == "access_control":
        request = replace(request, messages=[*request.messages, {"role": "assistant", "content": PREFILL}])
    return request


def verdict_logits(cases, adapter, routing):
    rows = []
    token_pair = None
    for case in cases:
        request = calibrated_request(case, routing)
        prefix = adapter.chat_prefix(request.messages[:-1]) + PREFILL
        ids = adapter._tokens(prefix)
        candidates = []
        for verdict in (VULNERABLE, CLEAN):
            continuation = adapter._tokens(prefix + verdict + '"')
            start = ids.shape[1]
            if continuation.shape[1] <= start or not adapter.torch.equal(continuation[:, :start], ids):
                raise ValueError("The verdict prefill crosses a tokenizer boundary")
            candidates.append(continuation[0, start].item())
        if candidates[0] == candidates[1] or (token_pair is not None and candidates != token_pair):
            raise ValueError("The two verdicts require stable, distinct first tokens")
        token_pair = candidates
        with adapter.torch.inference_mode():
            logits = adapter.model(input_ids=ids, attention_mask=adapter.torch.ones_like(ids),
                                   logits_to_keep=1, use_cache=False).logits[0, -1].float()
            positive, negative = [logits[index].item() for index in candidates]
            logits[candidates] = -float("inf")
            other = logits.max().item()
        if not all(map(math.isfinite, (positive, negative, other))):
            raise ValueError("Nonfinite verdict scores")
        rows.append({"case_id": case.case_id, "positive": positive, "negative": negative, "other": other,
                     "expected": case.oracle_verdict, "request_id": request.request_id})
    return rows, token_pair


def bias_score(rows, bias):
    correct, false_alarms = 0, 0
    for row in rows:
        scores = {VULNERABLE: row["positive"] + bias, CLEAN: row["negative"], None: row["other"]}
        ranked = sorted(scores, key=scores.get, reverse=True)
        prediction = ranked[0] if scores[ranked[0]] > scores[ranked[1]] else None
        correct += prediction == row["expected"]
        false_alarms += prediction == VULNERABLE and row["expected"] == CLEAN
    return correct, false_alarms


class CalibratedAdapter:
    def __init__(self, adapter, bias, positive_token):
        self.adapter = adapter
        prefix_tokens = adapter.tokenizer.encode(PREFILL, add_special_tokens=False)
        self.sequence = (*prefix_tokens, positive_token)
        self.bias = bias
        self.metadata = dict(adapter.metadata, verdict_calibration={
            "prefill": PREFILL, "sequence": list(self.sequence), "positive_token_bias": bias,
            "scope": "The bias applies when generating the first verdict token after the fixed JSON prefix.",
        })

    def generate(self, requests):
        config = self.adapter.model.generation_config
        original = config.sequence_bias
        if original is not None:
            raise ValueError("Existing generation biases must be isolated")
        results = []
        try:
            for request in requests:
                calibrated = request.messages[-1] == {"role": "assistant", "content": PREFILL}
                config.sequence_bias = {self.sequence: self.bias} if calibrated and self.bias else None
                results.extend(self.adapter.generate([request]))
        finally:
            config.sequence_bias = original
        return results


def calibrated_study(root, adapter, protocol, output):
    if (adapter.metadata["adapter_sha256"] != protocol["adapter_sha256"]
            or protocol["bias_grid"] != list(BIAS_GRID) or protocol["target_accuracy"] != 1.0):
        raise ValueError("Calibration settings differ from the frozen protocol")
    base_routing = protocol["source_method"]
    validation = checked_dataset(root, protocol["development"]) + checked_dataset(root, protocol["regression"])
    access = [case for case in validation if case.target_category == "access_control"]
    output.mkdir(parents=True, exist_ok=False)
    identity = {"protocol": protocol, "source": source_fingerprint()}
    trials = {}
    for access_method in protocol["access_methods"]:
        routing = dict(base_routing, access_control=access_method)
        logits, tokens = verdict_logits(access, adapter, routing)
        grid = [(bias, *bias_score(logits, bias)) for bias in BIAS_GRID]
        trials[access_method] = {"scores": logits, "grid": grid, "token_ids": tokens}
    access_method, chosen = max(((method, item) for method, trial in trials.items() for item in trial["grid"]),
                                key=lambda pair: (pair[1][1], -abs(pair[1][0]), -pair[1][2]))
    bias = chosen[0]
    routing = dict(base_routing, access_control=access_method)
    tuned = CalibratedAdapter(adapter, bias, trials[access_method]["token_ids"][0])
    calibration = {"trials": trials, "chosen_bias": bias, "source_method": routing}
    atomic_json(output / "calibration.json", calibration)
    make_request = lambda case, _: calibrated_request(case, routing)
    evaluate(validation, tuned, (METHOD,), output / "development", identity,
             make_request=make_request, prompt_version=VERSION)
    result = error_report(read_json(output / "development/results.json"))
    outcome = {"selected_prompt": METHOD, "development": result, "confirmation": None, "target_reached": False}
    if result["overall"]["accuracy"] >= protocol["minimum_dev_accuracy"]:
        atomic_json(output / "frozen.json", dict(identity, adapter_sha256=protocol["adapter_sha256"],
                    selected_prompt=METHOD, confirmation_sha256=protocol["confirmation"]["sha256"], source_method=routing,
                    verdict_calibration=tuned.metadata["verdict_calibration"]))
        fresh = checked_dataset(root, protocol["confirmation"])
        validate_cases(validation + fresh)
        evaluate(fresh, tuned, (METHOD,), output / "confirmation", identity,
                 make_request=make_request, prompt_version=VERSION)
        report = error_report(read_json(output / "confirmation/results.json"))
        outcome.update(confirmation=report, target_reached=report["overall"]["accuracy"] == 1.0)
    else:
        outcome["reason"] = "Development score below the preregistered gate"
    atomic_json(output / "outcome.json", outcome)
    return outcome
