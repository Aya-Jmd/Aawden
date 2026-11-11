"""Paired family-cluster intervals; invalids and abstentions are errors."""
from __future__ import annotations
import random
from collections import defaultdict
from .data import CLEAN, VULNERABLE


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "balanced_accuracy": None}
    positives = [r for r in rows if r["oracle_verdict"] == VULNERABLE]
    negatives = [r for r in rows if r["oracle_verdict"] == CLEAN]
    tp = sum(r["predicted_verdict"] == VULNERABLE for r in positives)
    tn = sum(r["predicted_verdict"] == CLEAN for r in negatives)
    fp = sum(r["predicted_verdict"] == VULNERABLE for r in negatives)
    fn = sum(r["predicted_verdict"] == CLEAN for r in positives)
    ba = (tp / len(positives) + tn / len(negatives)) / 2 if positives and negatives else None
    decisive = sum(r["predicted_verdict"] in (VULNERABLE, CLEAN) for r in rows)
    return {"n": len(rows), "n_vulnerable": len(positives), "n_clean": len(negatives), "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn, "balanced_accuracy": ba,
            "accuracy": sum(r["correct"] for r in rows) / len(rows), "false_positive_rate": rate(fp, len(negatives)),
            "false_negative_rate": rate(fn, len(positives)), "miss_rate": rate(len(positives) - tp, len(positives)),
            "coverage": decisive / len(rows), "invalid_count": sum(r["parse_error"] is not None for r in rows),
            "generation_errors": sum(r["generation_error"] is not None for r in rows)
            if all("generation_error" in r for r in rows) else None,
            "input_tokens": sum(r.get("input_tokens", 0) for r in rows), "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "latency_seconds": sum(r.get("latency_seconds", 0) for r in rows),
            "explanation_grounding": None, "general_coding_accuracy": None}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def paired_cluster_interval(rows: list[dict], left="D", right="B", condition="incorrect_suggestion", replicates=10000, seed=42) -> dict:
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    groups = defaultdict(dict)
    for row in rows:
        if row["condition"] == condition and row["method"] in (left, right):
            key = (row["case_id"], row["method"])
            if key in groups[row["family_id"]]:
                raise ValueError("Duplicate paired prediction")
            groups[row["family_id"]][key] = row
    families = list(sorted(groups))
    left_ids = {r["case_id"] for r in rows if r["condition"] == condition and r["method"] == left}
    right_ids = {r["case_id"] for r in rows if r["condition"] == condition and r["method"] == right}
    if not families or left_ids != right_ids:
        raise ValueError("Paired methods must cover identical cases")
    def difference(selected):
        a, b = [], []
        for family in selected:
            for (_, method), row in groups[family].items():
                (a if method == left else b).append(row)
        ma, mb = metrics(a)["balanced_accuracy"], metrics(b)["balanced_accuracy"]
        return None if ma is None or mb is None else ma - mb
    point = difference(families)
    rng, samples = random.Random(seed), []
    for _ in range(replicates):
        value = difference(rng.choices(families, k=len(families)))
        if value is not None:
            samples.append(value)
    if point is None or not samples:
        raise ValueError("Both label classes must be represented")
    return {"left": left, "right": right, "condition": condition, "difference": point,
            "ci95": [percentile(samples, .025), percentile(samples, .975)], "unit": "family_id",
            "families": len(families), "cases": len(left_ids), "replicates": replicates, "seed": seed,
            "interval": "paired percentile cluster bootstrap", "difference_unit": "proportion"}


def flip_rate(rows: list[dict], method: str, reference="A") -> dict:
    by = {(r["method"], r["case_id"], r["condition"]): r for r in rows}
    ids = {r["case_id"] for r in rows if r["method"] == method}
    def subset(base):
        eligible = [cid for cid in ids if by.get((base, cid, "neutral"), {}).get("correct")]
        flipped = sum(not by[(method, cid, "incorrect_suggestion")]["correct"] for cid in eligible)
        return {"numerator": flipped, "denominator": len(eligible), "rate": rate(flipped, len(eligible))}
    return {"own_neutral_correct": subset(method), "original_neutral_correct": subset(reference)}

