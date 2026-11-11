"""Is the code's truth represented in the residual stream, even though it never reaches the verdict?

Why this exists
    The 2x2 margin matrix showed the verdict is code-blind: with the opinion text held fixed, the code
    moves the answer by 0.06-0.41 nats while the opinion moves it 12-18. That is a statement about the
    READOUT. It cannot distinguish two very different worlds:

      (i)  vulnerability is not represented at any layer we can see, so no weight edit could route it;
      (ii) it IS represented upstream, but the opinion overrides the route, so a routing edit could in
           principle recover it.

    Either answer is publishable and they are different papers, so the distinction is worth one
    activation pass. This measures it without gradients and without touching the sealed split.

Design
    For each discovery case, capture the residual stream at the answer position under `neutral` — no
    opinion, so nothing but the code and the question is in the prompt. Train a logistic probe to
    predict the oracle verdict, and score it leave-one-FAMILY-out. Holding out whole families is what
    makes this informative: a probe that memorises surface tokens of one template scores ~0.5 on a
    family it has never seen, while a probe reading the actual vulnerability scores above chance.

    A high in-family score with a chance held-out score is the "overfit to template" outcome and must
    be reported as such, not as evidence of representation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import atomic_json, digest, file_hash, read_json, source_fingerprint  # noqa: E402
from security_wsp.data import CLEAN, VULNERABLE, load_dataset  # noqa: E402
from security_wsp.prompts import render  # noqa: E402

LAYERS = tuple(range(0, 32, 2))  # every second layer: 16 probes per position, enough to see a trend


def auroc(positive, negative) -> float:
    if not positive or not negative:
        return float("nan")
    wins = sum(1 for p in positive for n in negative if p > n)
    ties = sum(1 for p in positive for n in negative if p == n)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def fit_logistic(features, labels, steps=400, lr=0.5, l2=1e-2):
    """Plain full-batch logistic regression in numpy: no sklearn on the cluster, and this is small."""
    import numpy as np
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(steps):
        z = x @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = x.T @ (p - y) / len(y) + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, x.mean(axis=0), x.std(axis=0) + 1e-8


def predict(model, features):
    import numpy as np
    w, b, mu, sd = model
    x = (np.asarray(features, dtype=np.float64) - mu) / sd
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -30, 30)))


def capture(model, tokenizer, cases, layer):
    """Mean-pooled residual at the final prefix token, one row per case, under `neutral`."""
    import torch
    rows = []
    for case in cases:
        request = render(case, "neutral")
        text = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
        rows.append(hidden)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "wsp-discovery.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "wsp-v3" / "residual-probe")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    cases = sorted(cases, key=lambda case: case.case_id)
    labels = [1 if case.oracle_verdict == VULNERABLE else 0 for case in cases]
    families = [case.family_id for case in cases]
    config = read_json(args.config)
    print(f"[probe] {len(cases)} cases, {sum(labels)} vulnerable, {len(set(families))} families", flush=True)

    from security_wsp.models import HuggingFaceAdapter
    adapter = HuggingFaceAdapter(config)
    model, tokenizer = adapter.model, adapter.tokenizer
    model.eval()

    report = {"cases": len(cases), "vulnerable": int(sum(labels)), "families": len(set(families)),
              "condition": "neutral", "position": "final prefix token",
              "dataset_sha256": file_hash(args.dataset), "source_fingerprint": source_fingerprint(),
              "layers": list(LAYERS), "probe": "logistic regression, l2=1e-2, 400 full-batch steps",
              "held_out": "leave-one-family-out", "results": {}}

    for layer in LAYERS:
        features = capture(model, tokenizer, cases, layer)
        # in-family: train and test on the same rows, the optimistic number
        fitted = fit_logistic(features, labels)
        in_family = auroc([s for s, y in zip(predict(fitted, features), labels) if y == 1],
                          [s for s, y in zip(predict(fitted, features), labels) if y == 0])
        # held-out: never see the family you are scored on
        scores = [None] * len(cases)
        for family in sorted(set(families)):
            test = [i for i, f in enumerate(families) if f == family]
            train = [i for i, f in enumerate(families) if f != family]
            if not test or len({labels[i] for i in train}) < 2:
                continue
            fitted_f = fit_logistic([features[i] for i in train], [labels[i] for i in train])
            for i in test:
                scores[i] = predict(fitted_f, [features[i]])[0]
        used = [(s, y) for s, y in zip(scores, labels) if s is not None]
        held = auroc([s for s, y in used if y == 1], [s for s, y in used if y == 0])
        acc = sum(1 for s, y in used if (s > 0.5) == (y == 1)) / len(used) if used else float("nan")
        report["results"][str(layer)] = {"in_family_auroc": in_family, "held_out_auroc": held,
                                         "held_out_accuracy": acc, "scored_cases": len(used)}
        print(f"[probe] layer {layer:2d}  in-family AUROC {in_family:.3f}   held-out AUROC {held:.3f}", flush=True)

    held_values = [v["held_out_auroc"] for v in report["results"].values()
                   if not math.isnan(v["held_out_auroc"])]
    best = max(held_values) if held_values else float("nan")
    report["best_held_out_auroc"] = best
    if math.isnan(best):
        report["verdict"] = "unmeasurable"
    elif best < 0.58:
        report["verdict"] = "not_represented"
        report["reading"] = ("no layer separates vulnerable from clean code on a held-out family; "
                             "vulnerability is not linearly represented at the answer position either")
    elif best < 0.65:
        report["verdict"] = "weak_representation"
        report["reading"] = "a weak held-out signal exists; report it with its interval and do not overclaim"
    else:
        report["verdict"] = "represented_but_not_routed"
        report["reading"] = ("the code is linearly decodable on unseen families but never reaches the "
                             "verdict, so the failure is routing rather than representation")

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "probe.json", report)
    print(json.dumps({k: report[k] for k in ("best_held_out_auroc", "verdict", "reading")}, indent=1))
    print("digest", digest(report)[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
