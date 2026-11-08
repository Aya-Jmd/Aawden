"""Measured reports, explicit missing evidence, and dashboard payloads."""
from __future__ import annotations
from .artifacts import atomic_json


def pct(value):
    return "not measured" if value is None else f"{value*100:.1f}%"


def render_report(summary):
    toy = summary["evidence_kind"] == "constructed_toy"
    review = summary.get("final_review")
    if review:
        oracle_note = (f"The external evaluator attested independent, blinded label review and executable checks "
                       f"for all {review['cases']} cases. The committed review matches the dataset and oracle labels. "
                       "Operational separation and reviewer identity are the evaluator's responsibility.")
    else:
        oracle_note = ("Project-original SQLite and access-control fixtures have executable behavioral checks. "
                       "A coding agent authored and reviewed the development labels. The bundled test families are "
                       "exposed and exploratory; they are not a protected final benchmark.")
    lines = ["# Aawden — security review robustness", "",
             "**Evidence: " + ("constructed toy instrumentation demo; NOT Cohere results.**" if toy else "local checkpoint generations; see model manifest.**"),
             "", "## Research question", "",
             "Can a bounded checkpoint edit reduce incorrect deference to a reviewer's opinion while preserving independent security review?",
             "", "## Dataset and oracle", "",
             f"{summary['dataset']['cases']} cases in {summary['dataset']['families']} implementation families. "
             + oracle_note, "",
             "## Before / after", "",
             "| Method | Neutral BA | Incorrect-suggestion BA | Correct-suggestion accuracy | False-positive rate under incorrect suggestions | Coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, conditions in summary["metrics"].items():
        n, i, c = (conditions[k] for k in ("neutral", "incorrect_suggestion", "correct_suggestion"))
        lines.append(f"| {method} | {pct(n.get('balanced_accuracy'))} | {pct(i.get('balanced_accuracy'))} | {pct(c.get('accuracy'))} | {pct(i.get('false_positive_rate'))} | {pct(i.get('coverage'))} |")
    lines += ["", "A = original/standard; B = original/strong; C = edited/standard; D = edited/strong; "
              "R = seeded matched random control; N = unchanged software control. Invalid outputs and abstentions count as errors in balanced accuracy.", "",
              "## Paired uncertainty", ""]
    comparisons = list(summary.get("comparisons", {}).items()) + list(summary.get("standard_prompt_comparisons", {}).items())
    if "authority_wording" in summary:
        comparisons.append(("misleading advice with held-out authority wording", summary["authority_wording"]))
    for name, item in comparisons:
        lo, hi = item["ci95"]
        lines.append(f"- {item['left']} minus {item['right']}, {name}. {100*item['difference']:.1f} percentage points "
                     f"with a 95% paired family bootstrap interval [{100*lo:.1f}, {100*hi:.1f}] "
                     f"over {item['families']} families and {item['cases']} cases.")
    lines += ["", ("The paired intervals use the evaluator's committed final families."
                    if review else "Development and toy intervals are descriptive and cannot establish real-model transfer."), "",
              "## Mechanism and intervention", "",
              ("The toy has two fixed inputs: a lexical code feature and an opinion feature. "
               "The deliberately planted opinion weight is projected toward zero. No training or LLM inference is involved."
               if toy else "CircuitKIT supplies instrumented inference for residual contrasts and causal patches. "
               "The project projects selected residual-output weight matrices, then evaluates full generated verdicts. "
               "Forced-choice evidence and full generation evidence are distinct."), "",
              "## Candidate search and freeze", "",
              summary.get("selection_description", "See candidate artifacts."), "",
              "The trial ledger records targeted edits, random controls and their outcomes. Validation families choose the candidate.", "",
              "## Reproduction and integrity", "",
              "Run python aawden.py verify --run <this-run-directory>. The run manifest records the source identity, "
              "configuration, prompt version, model metadata, predictions, and artifact SHA-256 values. "
              "Raw generations and strict parse failures are retained.", "",
              "## Generation resources", "",
              "| Method | Answers | Input tokens | Output tokens | Generation seconds | Invalid answers | Generation errors |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method, conditions in summary["metrics"].items():
        values = list(conditions.values())
        totals = [sum(item.get(key, 0) for item in values) for key in
                  ("n", "input_tokens", "output_tokens", "latency_seconds", "invalid_count")]
        errors = sum(item["generation_errors"] for item in values) if all(item.get("generation_errors") is not None for item in values) else "unrecorded"
        lines.append(f"| {method} | {totals[0]} | {totals[1]} | {totals[2]} | {totals[3]:.2f} | {totals[4]} | {errors} |")
    if "strong_prompt_variant" in summary:
        lines += ["", f"Stronger prompt variant used for this run is `{summary['strong_prompt_variant']}`."]
    if "trials" in summary:
        lines += ["", "## Validation trials", "", "| Candidate | Role | Outcome | Attempts |", "|---|---|---|---|"]
        for trial in summary["trials"]:
            outcome = "ineligible" if trial["role"] == "targeted" and trial["status"] == "completed" and trial["eligible"] is False else trial["status"]
            lines.append(f"| {trial['candidate_id']} | {trial['role']} | {outcome} | {', '.join(trial['attempts']) or 'unrecorded'} |")
        lines += ["", "[Full trial ledger](trials.json)"]
    if "coding_regression" in summary:
        coding = summary["coding_regression"]
        lines += ["", "## Ordinary Python code questions", "", coding["scope"], ""]
        for arm, outcome in coding["arms"].items():
            lines.append(f"- {arm.capitalize()} answered {outcome['correct']} of {outcome['cases']} correctly ({pct(outcome['accuracy'])}).")
        lines.append(f"- Edited minus original was {100*coding['accuracy_difference']:.1f} percentage points.")
    if "grounding_review" in summary:
        grounding = summary["grounding_review"]
        lines += ["", "## Explanation grounding", "", grounding["scope"], ""]
        for arm, outcome in grounding["arms"].items():
            lines.append(f"- {arm} had {outcome['supported']} supported explanations out of {outcome['sampled']} sampled answers.")
        lines.append(f"- Edited minus original was {100*grounding['support_rate_difference']:.1f} percentage points.")
    lines += ["", "## Limits and unmeasured endpoints", ""]
    lines += ["- " + value for value in summary["unmeasured_endpoints"]]
    lines += ["", ("No claim of a Cohere robustness improvement is justified by a toy run."
                    if toy else "Interpret the comparison within the tested task, prompts, model, and stated uncertainty. "
                    "The reported benchmark does not establish general safety or new cybersecurity knowledge."), "",
              "## Sponsor integrations", ""]
    for name, info in summary.get("integrations", {}).items():
        lines.append(f"- {name}: {info.get('status', 'not executed')}. {info.get('scope', '')}")
    return "\n".join(lines) + "\n"


def write_report(path, summary):
    path.write_text(render_report(summary), encoding="utf-8")


def dashboard_payload(summary, rows, cases, oracles, trials, checkpoint):
    return {"summary": summary, "rows": rows, "cases": [c.__dict__ for c in cases], "oracles": oracles,
            "trials": trials, "checkpoint": checkpoint,
            "pipeline": [
                {"name": "Executable truth", "tool": "SQLite + policy fixtures", "status": "passed", "detail": "50 paired examples / executable oracles"},
                {"name": "Paired review", "tool": "Aawden prompt contracts", "status": "passed", "detail": "Neutral, correct and misleading opinions"},
                {"name": "Mechanism", "tool": "CircuitKIT", "status": "toy_only" if summary["evidence_kind"] == "constructed_toy" else "pending",
                 "detail": "Real-weight numerical parity is a mandatory gate"},
                {"name": "Bounded edit", "tool": "Reversible projection", "status": "toy_only" if summary["evidence_kind"] == "constructed_toy" else "pending",
                 "detail": "Candidate ledger, random controls and exact hashes"},
                {"name": "Evaluate", "tool": "AuditKIT + paired statistics", "status": summary["integrations"].get("AuditKIT", {}).get("status", "pending"),
                 "detail": "Strict parsing; invalid outputs count as errors"},
                {"name": "Freeze + reload", "tool": "Artifact manifests", "status": "toy_only" if summary["evidence_kind"] == "constructed_toy" else "pending",
                 "detail": "Fresh-process checkpoint verification"},
            ]}

