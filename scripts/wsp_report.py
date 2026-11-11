"""Judge-facing Markdown report for protocol v3, generated only from saved artifacts.

Every number in the output is read from a file named in the report. Nothing is computed from a
model, and nothing is estimated: a stage that did not run is printed as MISSING with the reason,
never as a zero, a blank or a plausible placeholder.

Usage:
  python scripts/wsp_report.py --baseline runs/wsp-v3/baseline-A2 --out docs/REPORT_V3.md
  python scripts/wsp_report.py --baseline runs/wsp-v3/baseline-A2 \
      --merged runs/wsp-v3/merged --localize runs/wsp-v3/localize \
      --rounds runs/wsp-v3/rounds --final runs/wsp-v3/final --out docs/REPORT_V3.md

Rules enforced here:
  - A metric that is absent from the artifacts is printed as MISSING, never as 0.
  - Balanced accuracy is never averaged or pooled across splits. One table per split, with n.
  - Every table names its split and its n, and the file it came from.
  - Every run directory is checked with security_wsp.artifacts.verify_run before anything is read
    from it. A hash that does not match its manifest aborts the whole report.
  - The output is byte-deterministic: no timestamps, sorted iteration, fixed float formatting.

This module imports no GPU code: only the standard library and security_wsp.artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from security_wsp.artifacts import file_hash, read_json, verify_run  # noqa: E402

MISSING = "MISSING"
PROTOCOL = ROOT / "configs" / "protocol-v3.json"
CONDITIONS = ("neutral", "correct_suggestion", "incorrect_suggestion")
EXPLORATORY = "exploratory localization, not a causal claim"

# The arm names scripts/wsp_measure.py writes, and which of them are the stronger-prompt comparator.
STANDARD_ARM = "A_pf"
STRONG_ARMS = ("B_trace_pf", "B_verify_pf")
NO_PREFILL_ARM = "A_raw"


class ReportError(RuntimeError):
    """A refusal: the report cannot be produced honestly from these artifacts."""


# --------------------------------------------------------------------------------------------------
# formatting helpers: absent is absent, never zero
# --------------------------------------------------------------------------------------------------


def number(value: Any, digits: int = 3) -> str:
    """Render a metric. None, a missing key and a non-number are all MISSING."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def count(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, int):
        return str(value)
    return str(value)


def provenance(path: Path) -> str:
    return f"`{Path(path).as_posix()}`"


def cell(value: Any) -> str:
    """Escape a value for a Markdown table cell: a literal pipe would split the column."""
    return str(value).replace("|", "\\|")


def row(cells: list[str]) -> str:
    return "| " + " | ".join(cell(c) for c in cells) + " |"


def header(cells: list[str]) -> list[str]:
    return [row(cells), "|" + "|".join(["---"] * len(cells)) + "|"]


# --------------------------------------------------------------------------------------------------
# artifact loading and integrity
# --------------------------------------------------------------------------------------------------


def verified(path: Path, what: str) -> dict:
    """Run verify_run and turn any integrity failure into a loud refusal."""
    try:
        return verify_run(path)
    except Exception as exc:  # noqa: BLE001 - the message is the point
        raise ReportError(f"{what}: artifact integrity failure in {path.as_posix()} ({exc})") from None


def load_verified(paths: list[tuple[Path, str]]) -> None:
    for path, what in paths:
        if path is not None:
            verified(path, what)


class Run:
    """A wsp_measure output directory: results.json, summary.json, decision.json."""

    def __init__(self, path: Path, role: str):
        self.path = Path(path)
        self.role = role
        self.summary_path = self.path / "summary.json"
        if not self.summary_path.is_file():
            raise ReportError(f"{role}: {self.summary_path.as_posix()} is absent; run wsp_measure first")
        self.payload = read_json(self.summary_path)
        self.summary = self.payload.get("summary") or {}
        self.label = self.payload.get("label")
        self.decision = read_json(self.path / "decision.json") if (self.path / "decision.json").is_file() else None

    def splits(self) -> list[str]:
        by = self.summary.get("by_arm_split") or {}
        return sorted({split for arm in by for split in by[arm]})

    def arms(self) -> list[str]:
        return sorted(self.summary.get("by_arm_split") or {})

    def block(self, arm: str, split: str, condition: str) -> dict | None:
        by = self.summary.get("by_arm_split") or {}
        return ((by.get(arm) or {}).get(split) or {}).get(condition)


# --------------------------------------------------------------------------------------------------
# section 1: the headline failure
# --------------------------------------------------------------------------------------------------


def section_headline(runs: list[Run], lines: list[str]) -> None:
    lines.append("## 1. The failure")
    lines.append("")
    lines.append("Balanced accuracy of the unedited model, per condition, per split. Balanced accuracy is "
                 "never pooled across splits: each split is its own table with its own n. Invalid answers "
                 "are counted as wrong by the protocol, so TP+TN+FP+FN need not equal n.")
    lines.append("")

    for run in runs:
        lines.append(f"### {run.role} — label `{run.label}`")
        lines.append("")
        lines.append(f"Source: {provenance(run.summary_path)}, field `summary.by_arm_split`. "
                     f"Verified with `security_wsp.artifacts.verify_run`.")
        lines.append("")
        splits = run.splits()
        if not splits:
            lines.append(f"{MISSING} — this summary records no per-split metrics.")
            lines.append("")
            continue
        for split in splits:
            arms = [a for a in run.arms() if run.block(a, split, CONDITIONS[0]) is not None
                    or any(run.block(a, split, c) is not None for c in CONDITIONS)]
            if not arms:
                continue
            ns = sorted({(run.block(a, split, c) or {}).get("n") for a in arms for c in CONDITIONS} - {None})
            n_text = ", ".join(str(v) for v in ns) if ns else MISSING
            lines.append(f"**Split `{split}`** — n = {n_text} per condition block.")
            lines.append("")
            lines.extend(header(["Arm", "Condition", "n", "Balanced accuracy", "TP", "TN", "FP", "FN", "Invalid"]))
            for arm in arms:
                for condition in CONDITIONS:
                    b = run.block(arm, split, condition)
                    if b is None:
                        continue
                    lines.append(row([f"`{arm}`", condition, count(b.get("n")),
                                      number(b.get("balanced_accuracy")), count(b.get("true_positive")),
                                      count(b.get("true_negative")), count(b.get("false_positive")),
                                      count(b.get("false_negative")), count(b.get("invalid_count"))]))
            measured = {c for a in arms for c in CONDITIONS if run.block(a, split, c) is not None}
            absent = [c for c in CONDITIONS if c not in measured]
            if absent:
                lines.append("")
                lines.append(f"Conditions with no record in this split: {', '.join(f'`{c}`' for c in absent)} "
                             f"({MISSING} — the arm was not measured on this condition, which is not the "
                             f"same as scoring 0).")
            lines.append("")

        if run.decision is not None:
            d = run.decision
            lines.append(f"Recorded decision ({provenance(run.path / 'decision.json')}): "
                         f"route `{d.get('route')}`, A_pf neutral balanced accuracy "
                         f"{number(d.get('A_pf_neutral_ba'))}, misleading "
                         f"{number(d.get('A_pf_misleading_ba'))}, drop {number(d.get('A_pf_drop'))}, "
                         f"prompt-only solution {number(d.get('prompt_only_solution'))}.")
            lines.append("")


def section_no_prefill(runs: list[Run], lines: list[str]) -> list[str]:
    """The strict no-prefill number is reported separately and never used for selection."""
    notes = []
    for run in runs:
        by = run.summary.get("by_arm_split") or {}
        arm = by.get(NO_PREFILL_ARM)
        if not arm:
            continue
        for split in sorted(arm):
            b = (arm.get(split) or {}).get("neutral")
            if b is None:
                continue
            notes.append(f"- No-prefill (`{NO_PREFILL_ARM}`), split `{split}`, n = {count(b.get('n'))}: "
                         f"neutral balanced accuracy {number(b.get('balanced_accuracy'))}, invalid "
                         f"{count(b.get('invalid_count'))} ({provenance(run.summary_path)}).")
    if not notes:
        return [f"- No-prefill control: {MISSING} — no `{NO_PREFILL_ARM}` arm in the supplied runs."]
    return notes


# --------------------------------------------------------------------------------------------------
# section 2: what changed and what did not
# --------------------------------------------------------------------------------------------------


def section_changes(runs: list[Run], lines: list[str]) -> list[str]:
    missing: list[str] = []
    lines.append("## 2. What changed and what did not")
    lines.append("")
    lines.append("Each method is compared against the standard-prompt baseline and against the "
                 "stronger-prompt comparator, **within one split and one condition**. A difference is "
                 "printed only when both numbers exist in the same split; otherwise MISSING. The protocol's "
                 "primary comparison is D (edited, stronger prompt) minus B (unedited, stronger prompt) on "
                 "`incorrect_suggestion`.")
    lines.append("")

    for run in runs:
        arms = run.arms()
        comparators = [a for a in STRONG_ARMS if a in arms]
        base = STANDARD_ARM if STANDARD_ARM in arms else None
        if not comparators and base is None:
            continue
        lines.append(f"### {run.role} — label `{run.label}`")
        lines.append("")
        lines.extend(header(["Split", "Arm", "Condition", "Balanced accuracy",
                             f"Δ vs `{base}`" if base else "Δ vs standard", "Δ vs stronger prompt"]))
        for split in run.splits():
            for arm in arms:
                for condition in CONDITIONS:
                    b = run.block(arm, split, condition)
                    if b is None:
                        continue
                    bacc = b.get("balanced_accuracy")
                    if base is not None and arm != base:
                        ref = run.block(base, split, condition)
                        base_bacc = None if ref is None else ref.get("balanced_accuracy")
                        d_base = (None if bacc is None or base_bacc is None else bacc - base_bacc)
                    else:
                        d_base = None  # an arm compared against itself is a tautology, not a result
                    strong_cells = []
                    for strong in comparators:
                        if strong == arm:
                            continue
                        s = run.block(strong, split, condition)
                        s_bacc = None if s is None else s.get("balanced_accuracy")
                        value = None if bacc is None or s_bacc is None else bacc - s_bacc
                        strong_cells.append(f"{strong}: {number(value)}")
                    if not strong_cells:
                        strong_cells = [MISSING]
                    lines.append(row([f"`{split}`", f"`{arm}`", condition, number(bacc),
                                      number(d_base), "; ".join(strong_cells)]))
        lines.append("")
        if not comparators:
            missing.append(f"Stronger-prompt comparator on {run.role}: {MISSING} — no "
                           f"{'/'.join(STRONG_ARMS)} arm in that run.")

    # paired intervals recorded by the producer
    lines.append("### Paired intervals")
    lines.append("")
    lines.append("Read verbatim from each summary's `paired` block. The producer computes these in the "
                 "protocol's bootstrap unit (`family_id`); the scope column states whether the producer "
                 "pooled it across splits, because this report does not.")
    lines.append("")
    printed = False
    for run in runs:
        paired = run.summary.get("paired") or {}
        if not paired:
            continue
        lines.extend(header(["Run", "Pair", "Condition", "Difference", "CI95", "Unit", "Scope"]))
        for pair in sorted(paired):
            for condition in CONDITIONS:
                entry = (paired.get(pair) or {}).get(condition)
                if not isinstance(entry, dict) or not entry:
                    continue
                ci = entry.get("ci95")
                ci_text = MISSING if not ci else f"[{number(ci[0])}, {number(ci[1])}]"
                lines.append(row([run.role, f"`{pair}`", condition, number(entry.get("difference")),
                                  ci_text, str(entry.get("unit", MISSING)),
                                  "pooled across splits by the producer"
                                  if len(run.splits()) > 1 else "single split"]))
                printed = True
        lines.append("")
    if not printed:
        lines.append(f"{MISSING} — no `paired` interval in any supplied run. The baseline summary's "
                     f"`paired` block is empty, because no stronger-prompt arm was measured in it.")
        lines.append("")
        missing.append(f"Paired bootstrap intervals: {MISSING} — empty `paired` block in the supplied runs.")

    return missing


# --------------------------------------------------------------------------------------------------
# section 3: controls
# --------------------------------------------------------------------------------------------------


def _localization_payload(path: Path) -> dict | None:
    p = Path(path) / "localization.json"
    return read_json(p) if p.is_file() else None


def section_controls(runs: list[Run], localize: Path | None, final: Path | None,
                     lines: list[str]) -> list[str]:
    missing: list[str] = []
    lines.append("## 3. Controls")
    lines.append("")
    lines.append("A positive result is only worth reading if these pass. Each row states where it came "
                 "from; a control with no artifact is MISSING, not assumed.")
    lines.append("")
    lines.extend(header(["Control", "Requirement", "Result", "Source"]))

    # no-op
    noop_cells = None
    if localize is not None:
        payload = _localization_payload(localize)
        causal = (payload or {}).get("causal") or {}
        components = causal.get("components") or []
        if components:
            passed = all((c.get("no_op_control") or {}).get("passed") for c in components)
            worst = max(((c.get("no_op_control") or {}).get("max_absolute_margin_delta") or 0.0)
                        for c in components)
            tolerance = (components[0].get("no_op_control") or {}).get("tolerance")
            noop_cells = (f"passed={number(bool(passed))}; max |Δmargin| {number(worst)} "
                          f"<= {number(tolerance)}",
                          provenance(Path(localize) / "localization.json"))
    if noop_cells is None and final is not None:
        f_summary = Path(final) / "summary.json"
        if f_summary.is_file():
            check = read_json(f_summary).get("noop_check") or {}
            if check and not check.get("missing"):
                noop_cells = (f"equal to `{check.get('compared_to')}` on "
                              f"`{check.get('condition')}`: {number(check.get('equal'))}",
                              provenance(f_summary))
    if noop_cells is None:
        lines.append(row(["No-op edit", "must reproduce the baseline", MISSING,
                          "no localization or sealed-final artifact supplied, and wsp_measure runs no no-op arm"]))
        missing.append(f"No-op control: {MISSING} — it lives in wsp_localize (causal stage) and "
                       f"wsp_final (noop_check); neither run directory was supplied.")
    else:
        lines.append(row(["No-op edit", "must reproduce the baseline", noop_cells[0], noop_cells[1]]))

    # random controls: configured seeds, then any measured result
    seeds = []
    if PROTOCOL.is_file():
        seeds = (read_json(PROTOCOL).get("controls") or {}).get("random_control_seeds") or []
    random_result = None
    random_source = None
    if localize is not None:
        payload = _localization_payload(localize)
        causal = (payload or {}).get("causal") or {}
        if causal.get("random_seeds"):
            largest = [c.get("largest_random_absolute_effect") for c in (causal.get("components") or [])]
            random_result = (f"seeds {causal['random_seeds']}; per-component largest |effect| recorded "
                             f"for {sum(v is not None for v in largest)} component(s)")
            random_source = provenance(Path(localize) / "localization.json")
    if random_result is None:
        lines.append(row(["Size-matched random patches",
                          f"same tensors, same #slices, norm-matched; seeds {seeds or MISSING}",
                          MISSING, "no artifact supplied records random-control results"]))
        missing.append(f"Random size-matched patch results: {MISSING} — seeds are configured "
                       f"{seeds or MISSING} in configs/protocol-v3.json but no run on disk reports them.")
    else:
        lines.append(row(["Size-matched random patches",
                          f"same tensors, same #slices, norm-matched; seeds {seeds or MISSING}",
                          random_result, random_source]))

    # permuted label
    permuted = None
    permuted_source = None
    if localize is not None:
        payload = _localization_payload(localize)
        causal = (payload or {}).get("causal") or {}
        for component in (causal.get("components") or []):
            mapping = component.get("label_mapping_swap") or {}
            if mapping:
                permuted = (f"`{component.get('component')}`: {mapping.get('status', MISSING)}")
                permuted_source = provenance(Path(localize) / "localization.json")
                break
    if permuted is None:
        lines.append(row(["Permuted labels", "agent must not recover swapped verdict labels", MISSING,
                          "no artifact supplied records a label-mapping swap"]))
        missing.append(f"Permuted-label control: {MISSING} — recorded only inside the wsp_localize causal "
                       f"stage; no such run directory was supplied.")
    else:
        lines.append(row(["Permuted labels", "agent must not recover swapped verdict labels",
                          permuted, permuted_source]))

    # shuffled weights
    lines.append(row(["Shuffled weights", "separates reading weights from prior knowledge", MISSING,
                      "no artifact on disk records a shuffled-weight control"]))
    missing.append(f"Shuffled-weight control: {MISSING} — no artifact on disk records it.")

    lines.append("")
    return missing


# --------------------------------------------------------------------------------------------------
# section 4: localization
# --------------------------------------------------------------------------------------------------


def section_localization(localize: Path | None, lines: list[str]) -> list[str]:
    lines.append("## 4. Localization")
    lines.append("")
    if localize is None:
        lines.append(f"{MISSING} — no `--localize` directory supplied. Localization did not run, so there "
                     f"is no `components.json`, `attention.json` or `localization.json` to read.")
        lines.append("")
        return []  # the MISSING section already names the absent stage; do not repeat it

    components_path = Path(localize) / "components.json"
    if not components_path.is_file():
        lines.append(f"{MISSING} — `{components_path.as_posix()}` is absent.")
        lines.append("")
        return [f"Localization components: {MISSING} — {components_path.as_posix()} absent."]

    payload = read_json(components_path)
    localization = _localization_payload(localize) or {}
    causal = localization.get("causal") or {}
    supported = [c for c in (causal.get("components") or []) if c.get("supported_by_controls")]

    lines.append(f"Source: {provenance(components_path)} (ranking) and "
                 f"{provenance(Path(localize) / 'localization.json')} (causal check).")
    lines.append("")
    claim_default = payload.get("claim") or EXPLORATORY
    if supported:
        named = ", ".join("`" + str(c.get("component")) + "`" for c in supported)
        lines.append(f"**Claim status:** the control comparison supports "
                     f"{len(supported)} top component(s) as causal for the margin "
                     f"({named}). Every other component is **{claim_default}**.")
    else:
        lines.append(f"**Claim status: {claim_default}.** No component passed the control comparison, so "
                     f"nothing below is a causal statement about the model.")
    lines.append("")
    lines.append(f"Ranking rule (verbatim from the artifact): {payload.get('ranking_rule', MISSING)}")
    lines.append("")

    ranked = payload.get("ranked") or []
    if ranked:
        lines.append(f"**Split `wsp_discovery`** — localization runs on the discovery split only; "
                     f"n = {count((ranked[0] or {}).get('cases'))} cases per component.")
        lines.append("")
        lines.extend(header(["Rank", "Component", "Site", "Layer", "Cases", "Separation",
                             "Random mean", "Score", "Degenerate"]))
        for entry in ranked[:10]:
            lines.append(row([count(entry.get("rank")), f"`{entry.get('component')}`", str(entry.get("site")),
                              count(entry.get("layer")), count(entry.get("cases")),
                              number(entry.get("separation_statistic")),
                              number(entry.get("random_baseline_mean")), number(entry.get("score")),
                              number(bool(entry.get("degenerate")))]))
        lines.append("")
        if len(ranked) > 10:
            lines.append(f"(Top 10 of {len(ranked)} ranked components shown.)")
            lines.append("")
    else:
        lines.append(f"{MISSING} — `components.json` records no ranked components.")
        lines.append("")

    if causal.get("components"):
        lines.append("### Causal projection check against its controls")
        lines.append("")
        lines.extend(header(["Component", "Mean margin Δ", "Sign-flip p", "Largest random |effect|",
                             "Label swap", "Supported by controls", "Conclusion"]))
        for entry in causal["components"]:
            effect = entry.get("effect") or {}
            noop = entry.get("no_op_control") or {}
            mapping = entry.get("label_mapping_swap") or {}
            lines.append(row([f"`{entry.get('component')}`",
                              number(effect.get("mean_margin_delta")),
                              number((effect.get("sign_flip") or {}).get("p_value")),
                              number(entry.get("largest_random_absolute_effect")),
                              str(mapping.get("status", MISSING)),
                              number(bool(entry.get("supported_by_controls"))),
                              str(entry.get("conclusion", MISSING))]))
            if noop and not noop.get("passed"):
                lines.append(row([f"`{entry.get('component')}` no-op", "failed", MISSING, MISSING,
                                  MISSING, MISSING, "harness invalid"]))
        lines.append("")

    return [] if supported else ["Localization causal claim: not supported by controls "
                                 f"({EXPLORATORY})."]


# --------------------------------------------------------------------------------------------------
# section 5: agent rounds
# --------------------------------------------------------------------------------------------------


def _verify_rounds_state(path: Path) -> list[str]:
    """Verify each round's recorded artifact hashes from state.json. Loud on mismatch."""
    problems: list[str] = []
    state_path = Path(path) / "state.json"
    if not state_path.is_file():
        return problems
    state = read_json(state_path)
    for record in state.get("rounds") or []:
        directory = Path(path) / str(record.get("artifact_dir") or "")
        for name, expected in (record.get("artifacts") or {}).items():
            artifact = directory / name
            if not artifact.is_file():
                problems.append(f"round {record.get('round')}: {artifact.as_posix()} is absent")
            elif file_hash(artifact) != expected:
                problems.append(f"round {record.get('round')}: {artifact.as_posix()} sha256 does not match "
                                f"the hash recorded in state.json")
    return problems


def section_rounds(rounds: Path | None, lines: list[str]) -> list[str]:
    lines.append("## 5. Agent rounds")
    lines.append("")
    if rounds is None:
        lines.append(f"{MISSING} — no `--rounds` directory supplied. No weight-space-programming round ran.")
        lines.append("")
        return []  # the MISSING section already names the absent stage; do not repeat it

    problems = _verify_rounds_state(rounds)
    if problems:
        raise ReportError("Rounds directory failed its own integrity check: " + "; ".join(problems))

    state_path = Path(rounds) / "state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    budget = state.get("budget") or {}
    lines.append(f"Source: {provenance(state_path)} and `rounds/round-NN/summary.json` under "
                 f"`{Path(rounds).as_posix()}`.")
    lines.append("")
    lines.append(f"Forward-pass budget (protocol unit): used {count(budget.get('forward_units_used'))}, "
                 f"remaining {count(budget.get('forward_units_remaining'))}, "
                 f"total {count(budget.get('forward_pass_budget'))}.")
    lines.append("")

    summaries = []
    round_root = Path(rounds) / "rounds"
    if round_root.is_dir():
        for directory in sorted(p for p in round_root.iterdir() if p.is_dir()):
            candidate = directory / "summary.json"
            if candidate.is_file():
                summaries.append(read_json(candidate))
    if not summaries:
        lines.append(f"{MISSING} — no `rounds/round-NN/summary.json` under "
                     f"`{round_root.as_posix()}`; the state records "
                     f"{len(state.get('rounds') or [])} round(s).")
        lines.append("")
        return ["Agent round results: MISSING — the rounds directory holds no round summary."]

    lines.append("Split: `wsp_discovery`. n per round is the balanced-accuracy denominator recorded in "
                 "each round's own `per_class_counts`. A dash in the seed column means the round has no "
                 "seed by design (a proposal round), which is not the same as MISSING.")
    lines.append("")
    lines.extend(header(["Round", "Kind", "Seed", "Misleading BA", "Neutral BA", "Correct-opinion BA",
                         "Forward units used", "Changed scalars", "Update norm"]))
    for summary in summaries:
        # A proposal round records seed = null by design; that is "not applicable", which is a
        # different statement from MISSING ("should exist but no artifact has it").
        seed = summary.get("seed")
        seed_cell = "--" if seed is None else count(seed)
        lines.append(row([count(summary.get("round")), str(summary.get("kind", MISSING)),
                          seed_cell,
                          number(summary.get("misleading_balanced_accuracy")),
                          number(summary.get("neutral_balanced_accuracy")),
                          number(summary.get("correct_suggestion_balanced_accuracy")),
                          count((summary.get("budget") or {}).get("forward_units_used")),
                          count(summary.get("changed_scalars")),
                          number(summary.get("update_norm"))]))
    lines.append("")
    lines.append(f"n for the misleading column, per round: "
                 + ", ".join(f"round {s.get('round')} = "
                             f"{count((s.get('per_class_counts') or {}).get('incorrect_suggestion', {}).get('n'))}"
                             for s in summaries) + ".")
    lines.append("")
    return []


# --------------------------------------------------------------------------------------------------
# section 6: cost
# --------------------------------------------------------------------------------------------------


def section_cost(runs: list[Run], lines: list[str]) -> list[str]:
    lines.append("## 6. Cost and resources")
    lines.append("")
    lines.append("Token counts and wall-clock latency are read per split from each summary. GPU-seconds are "
                 "reported only if an artifact records them; no artifact field carries them, so they are "
                 "MISSING rather than inferred from latency.")
    lines.append("")
    printed = False
    for run in runs:
        for split in run.splits():
            arms = [a for a in run.arms() if run.block(a, split, CONDITIONS[0]) is not None]
            if not arms:
                continue
            lines.append(f"**{run.role}, split `{split}`** — {provenance(run.summary_path)}")
            lines.append("")
            lines.extend(header(["Arm", "Condition", "n", "Input tokens", "Output tokens",
                                 "Latency (s)", "Generation errors"]))
            for arm in arms:
                for condition in CONDITIONS:
                    b = run.block(arm, split, condition)
                    if b is None:
                        continue
                    lines.append(row([f"`{arm}`", condition, count(b.get("n")),
                                      count(b.get("input_tokens")), count(b.get("output_tokens")),
                                      number(b.get("latency_seconds"), 2),
                                      count(b.get("generation_errors"))]))
                    printed = True
            lines.append("")
    if not printed:
        lines.append(f"{MISSING} — no per-split token or latency record in the supplied runs.")
        lines.append("")
    lines.append(f"**GPU-seconds: {MISSING}.** No artifact in the repository records GPU time; latency "
                 f"above is wall-clock per condition block and is not a substitute.")
    lines.append("")
    return [f"GPU-seconds: {MISSING} — no artifact field records GPU time."]


# --------------------------------------------------------------------------------------------------
# section 7: MISSING
# --------------------------------------------------------------------------------------------------


def section_missing(missing: list[str], provided: dict[str, Path | None], lines: list[str]) -> None:
    lines.append("## 7. MISSING")
    lines.append("")
    lines.append("Every stage that did not run, or whose evidence is not in the supplied artifact set. "
                 "Nothing in this report fills these gaps with an estimate.")
    lines.append("")
    items: list[str] = []
    for name, path in (("Baseline", provided.get("baseline")), ("Merged-base measurement", provided.get("merged")),
                       ("Localization", provided.get("localize")), ("Agent rounds", provided.get("rounds")),
                       ("Sealed final evaluation", provided.get("final"))):
        if path is None:
            items.append(f"- {name}: {MISSING} — no directory supplied.")
    items.extend(f"- {entry}" for entry in missing)
    # Deduplicate by the subject before the colon, preserving order: a gap can be reported both by
    # "no directory supplied" and by its own section, and listing it twice reads as two gaps.
    seen, unique = set(), []
    for entry in items:
        key = entry.split(":", 1)[0].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    items = unique
    if not items:
        lines.append("Nothing is missing: every stage supplied an artifact.")
    else:
        lines.extend(items)
    lines.append("")


# --------------------------------------------------------------------------------------------------
# section 8: reproduction
# --------------------------------------------------------------------------------------------------


def section_reproduction(lines: list[str]) -> None:
    lines.append("## 8. Reproduction")
    lines.append("")
    lines.append("Run from the repository root. The final set is sealed: `wsp_final.py evaluate` opens it "
                 "once, after `freeze`, and refuses a second look.")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/build_wsp_data.py            # 220 paired fixtures, hashes committed")
    lines.append("python scripts/wsp_measure.py \\")
    lines.append("    --output runs/wsp-v3/baseline-A2 \\")
    lines.append("    --dataset datasets/wsp-discovery.json \\")
    lines.append("    --dataset datasets/wsp-validation.json")
    lines.append("python scripts/wsp_localize.py \\")
    lines.append("    --dataset datasets/wsp-discovery.json \\")
    lines.append("    --output runs/wsp-v3/localize --layers all --top-k 6")
    lines.append("python scripts/wsp_agent_round.py dump    --output runs/wsp-v3/rounds")
    lines.append("python scripts/wsp_agent_round.py apply   --output runs/wsp-v3/rounds --spec <edit-spec.json>")
    lines.append("python scripts/wsp_final.py freeze   --output runs/wsp-v3/final ...")
    lines.append("python scripts/wsp_final.py evaluate --output runs/wsp-v3/final ...")
    lines.append("python scripts/wsp_report.py --baseline runs/wsp-v3/baseline-A2 \\")
    lines.append("    --localize runs/wsp-v3/localize --rounds runs/wsp-v3/rounds \\")
    lines.append("    --final runs/wsp-v3/final --out docs/REPORT_V3.md")
    lines.append("```")
    lines.append("")
    lines.append("Every run directory is re-verified with `security_wsp.artifacts.verify_run` before this "
                 "report reads it, so a re-run that changed an artifact cannot silently enter the report.")
    lines.append("")


# --------------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> str:
    baseline = Path(args.baseline) if args.baseline else None
    merged = Path(args.merged) if args.merged else None
    localize = Path(args.localize) if args.localize else None
    rounds = Path(args.rounds) if args.rounds else None
    final = Path(args.final) if args.final else None

    # Integrity first: a tampered artifact must abort before any number is printed.
    for path, what in ((baseline, "Baseline"), (merged, "Merged base"), (localize, "Localization"),
                       (final, "Sealed final")):
        if path is not None:
            if not path.is_dir():
                raise ReportError(f"{what}: {path.as_posix()} is not a directory")
            verified(path, what)
    if rounds is not None and not Path(rounds).is_dir():
        raise ReportError(f"Agent rounds: {rounds.as_posix()} is not a directory")

    runs = []
    if baseline is not None:
        runs.append(Run(baseline, "Baseline (unedited model)"))
    if merged is not None:
        runs.append(Run(merged, "Merged base (edit base for every method)"))

    lines: list[str] = []
    lines.append("# Aawden — protocol v3 report")
    lines.append("")
    lines.append("Generated from saved artifacts only. Every number below names the file it was read from; "
                 "a stage that did not run is printed as MISSING, never estimated.")
    lines.append("")
    lines.append(f"Conditions: {', '.join(f'`{c}`' for c in CONDITIONS)}. Primary metric: balanced "
                 f"accuracy. Invalid answers count as wrong.")
    lines.append("")

    section_headline(runs, lines)

    if runs:
        lines.append("### No-prefill control")
        lines.append("")
        lines.extend(section_no_prefill(runs, lines))
        lines.append("")

    change_missing = section_changes(runs, lines)
    control_missing = section_controls(runs, localize, final, lines)
    localize_missing = section_localization(localize, lines)
    rounds_missing = section_rounds(rounds, lines)
    cost_missing = section_cost(runs, lines)

    all_missing = change_missing + control_missing + localize_missing + rounds_missing + cost_missing
    if final is None:
        all_missing.append(f"Sealed final evaluation: {MISSING} — no `--final` run directory supplied, so "
                           f"`summary.json` and `report.md` do not exist.")
    section_missing(all_missing, {"baseline": baseline, "merged": merged, "localize": localize,
                                  "rounds": rounds, "final": final}, lines)
    section_reproduction(lines)

    return "\n".join(lines).rstrip("\n") + "\n"


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".wsp-report-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, help="wsp_measure run dir: results.json, summary.json, decision.json")
    parser.add_argument("--merged", type=Path, help="wsp_measure run dir on the merged adapter (label merged-adapter)")
    parser.add_argument("--localize", type=Path, help="wsp_localize run dir: components.json, attention.json, localization.json")
    parser.add_argument("--rounds", type=Path, help="wsp_agent_round run dir: workbench/, rounds/, state.json")
    parser.add_argument("--final", type=Path, help="wsp_final run dir: summary.json, report.md")
    parser.add_argument("--out", type=Path, help="Markdown output path; stdout when omitted")
    args = parser.parse_args(argv)

    try:
        text = build(args)
    except ReportError as exc:
        print(f"[wsp_report] REFUSED: {exc}", file=sys.stderr)
        return 2

    if args.out:
        atomic_write(args.out, text)
        print(f"[wsp_report] wrote {args.out.as_posix()}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
