"""Training rows for the anti-deference adapter (Arm 1 of docs/FIX_PROTOCOL.md).

Why this exists
    The distillation arm trained one answer format and was measured in another, which left its null
    ambiguous. This arm trains the EXACT measured format: the study's own prompt, rendered by
    `security_wsp.prompts.render`, followed by the json_prefill `{"verdict":"`, with the oracle verdict
    as the only supervised tokens. That is what `wsp_cross_model.render_for(..., PREFILL)` scores.

What is written
    One JSONL row per (case, condition): `system`, `prompt` (the user turn), `prefill`, `completion`
    (the oracle verdict string), `label`, `condition`, `family`. The key is `label`, not `oracle`, so the
    teacher-trace rejection filter in `distill_sft.load_traces` leaves these rows alone: the target IS
    the oracle, there is nothing to reject.

    --conditions all      neutral + correct_suggestion + incorrect_suggestion (the main recipe)
    --conditions neutral  neutral only (the control: gets better at review, never sees an opinion)
    --prior-answer        round 3 (R9): adds two rows per case in which the conversation already holds
                          an answer, once the right verdict and once the wrong one, followed by a plain
                          request to check again. No reviewer appears in these rows and the wording
                          `pushback` is never rendered, so `pushback` stays a held-out wording.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.data import Case, validate_cases  # noqa: E402
from security_wsp.prompts import CLEAN, PREFILL, PUSHBACK_PRIOR, VULNERABLE, render  # noqa: E402

EVAL_FILES = ("plain-pooled.json", "wsp-discovery.json")
SEALED = ("wsp-final.json", "wsp-validation.json")
# Second user turn of the prior-answer rows. Deliberately not the pushback sentence: no reviewer, no verdict.
RECHECK = "Check the source again and give your final security assessment as one JSON object."


def prior_answer_rows(case: Case) -> list[dict]:
    """Two rows: the same neutral prompt, an earlier answer (right, then wrong), a request to re-check."""
    request = render(case, "neutral")
    rows = []
    for tag, prior in (("prior_right", case.oracle_verdict),
                       ("prior_wrong", CLEAN if case.oracle_verdict == VULNERABLE else VULNERABLE)):
        messages = [dict(m) for m in request.messages] + [
            {"role": "assistant", "content": PUSHBACK_PRIOR.format(
                verdict=prior, category=case.visible()["target_category"])},
            {"role": "user", "content": RECHECK}]
        rows.append({"key": f"{case.case_id}:{tag}", "family": case.family_id, "condition": tag,
                     "system": messages[0]["content"], "prompt": messages[1]["content"],
                     "messages": messages, "prefill": PREFILL, "completion": case.oracle_verdict,
                     "label": case.oracle_verdict})
    return rows


def load_cases(path: Path) -> list[Case]:
    """`distill-review-train.json` carries a `_source` provenance key `Case` does not accept."""
    items = json.loads(path.read_text())["cases"]
    cases = [Case.from_dict({k: v for k, v in item.items() if not k.startswith("_")}) for item in items]
    validate_cases(cases)
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "distill-review-train.json")
    parser.add_argument("--conditions", choices=["all", "neutral"], default="all")
    parser.add_argument("--wording", default="reviewer")
    parser.add_argument("--wordings", default=None,
                        help="comma list (round 5, R11): each case gets its neutral row once and both reviewer "
                             "conditions under every listed wording; overrides --wording")
    parser.add_argument("--prior-answer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.dataset.name in SEALED or args.dataset.name in EVAL_FILES:
        raise SystemExit(f"{args.dataset.name} is an evaluation or sealed file; refusing to train on it")
    cases = load_cases(args.dataset)
    train_families = {c.family_id for c in cases}
    for name in EVAL_FILES:
        eval_families = {c["family_id"] for c in json.loads((ROOT / "datasets" / name).read_text())["cases"]}
        overlap = train_families & eval_families
        if overlap:
            raise SystemExit(f"training shares {len(overlap)} families with {name}")

    conditions = ("neutral",) if args.conditions == "neutral" else \
        ("neutral", "correct_suggestion", "incorrect_suggestion")
    wordings = [w.strip() for w in args.wordings.split(",")] if args.wordings else [args.wording]
    rows = []
    for case in cases:
        for condition, wording in [(c, w) for c in conditions for w in (wordings if c != "neutral" else wordings[:1])]:
            request = render(case, condition, wording=wording)
            system, user = request.messages[0]["content"], request.messages[1]["content"]
            suffix = f":{wording}" if len(wordings) > 1 and condition != "neutral" else ""
            rows.append({"key": f"{case.case_id}:{condition}{suffix}", "family": case.family_id,
                         "condition": condition, "system": system, "prompt": user,
                         "prefill": PREFILL, "completion": case.oracle_verdict,
                         "label": case.oracle_verdict})
        if args.prior_answer:
            rows.extend(prior_answer_rows(case))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[antidef] {len(rows)} rows ({len(cases)} cases x {len(conditions)} conditions"
          f"{' + 2 prior-answer rows' if args.prior_answer else ''}, "
          f"{len(train_families)} families, 0 overlap with {EVAL_FILES}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
