"""Aawden demo: saved outputs from real runs. No live inference and no network calls.

The page shows one case at a time. Each case is the same code under a misleading reviewer opinion,
answered by three roles: base Aya (prefilled JSON, standard prompt), prompt-only (a stronger prompt
on the same base) and patched (a weight patch applied to the base). The executable oracle verdict is
shown next to each answer, and a summary of which weights changed comes from the patch manifest.

examples.json is built by `scripts/hf_publish.py build-space-examples` from wsp_measure results and
excludes the sealed wsp_final split by default. If it is absent, the page says results are pending.
"""
from __future__ import annotations

import json
from pathlib import Path

try:  # The helpers below stay importable without gradio, for tests and checks.
    import gradio as gr
except ImportError:  # pragma: no cover - exercised only when gradio is not installed
    gr = None

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples.json"
CONDITION = "incorrect_suggestion"
PENDING = (
    "**Results pending.** `examples.json` is not in this Space, so no model outputs are shown. "
    "Outputs appear only after the owner-approved run summaries are converted with "
    "`python scripts/hf_publish.py build-space-examples`. Nothing on this page is estimated or filled in."
)


def load_examples(path: Path = EXAMPLES) -> dict | None:
    """Return the saved payload, or None when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("examples"), list):
        raise ValueError("examples.json does not match schema_version 1")
    return payload


def case_ids(payload: dict | None, category: str | None = None) -> list[str]:
    if not payload:
        return []
    ids = {e["case_id"] for e in payload["examples"]
           if e["condition"] == CONDITION and (category in (None, "All") or e.get("category") == category)}
    return sorted(ids)


def categories(payload: dict | None) -> list[str]:
    if not payload:
        return ["All"]
    return ["All"] + sorted({e["category"] for e in payload["examples"] if e.get("category")})


def pick_roles(payload: dict, case_id: str) -> dict[str, dict | None]:
    """Choose one misleading-reviewer answer per role. Patched rows are those whose label names a patch."""
    rows = [e for e in payload["examples"] if e["case_id"] == case_id and e["condition"] == CONDITION]
    is_patch = lambda e: "patch" in str(e.get("label", "")).lower()  # noqa: E731
    base = next((e for e in rows if e["method"] == "A_pf" and not is_patch(e)), None)
    prompt = next((e for e in rows if str(e["method"]).startswith("B_") and not is_patch(e)), None)
    patched = next((e for e in rows if is_patch(e)), None)
    return {"base": base, "prompt_only": prompt, "patched": patched}


def _verdict(row: dict | None) -> str:
    if row is None:
        return "not in saved outputs"
    predicted = row.get("predicted_verdict") or "unparsed"
    mark = "correct" if predicted == row.get("oracle_verdict") else "wrong"
    return f"{predicted} ({mark})"


def case_markdown(payload: dict | None, case_id: str | None) -> str:
    if not payload:
        return PENDING
    if not case_id:
        return "No case selected."
    rows = [e for e in payload["examples"] if e["case_id"] == case_id and e["condition"] == CONDITION]
    if not rows:
        return f"No saved misleading-reviewer outputs for `{case_id}`."
    sample = rows[0]
    roles = pick_roles(payload, case_id)
    lines = [
        f"### `{case_id}`",
        f"**Category:** {sample.get('category')}  |  **Split:** {sample.get('split')}  |  "
        f"**Executable oracle verdict:** `{sample.get('oracle_verdict')}`",
        f"**Threat model:** {sample.get('threat_model') or 'not recorded'}",
        "**Condition:** the same code with a misleading reviewer opinion.",
        "",
        "```python",
        (sample.get("code") or "# code not recorded").rstrip(),
        "```",
        "",
        "| Role | Source | Predicted verdict | Oracle |",
        "|---|---|---|---|",
    ]
    names = {"base": "Base Aya", "prompt_only": "Prompt only", "patched": "Patched weights"}
    for role in ("base", "prompt_only", "patched"):
        row = roles[role]
        source = "not in saved outputs" if row is None else f"method `{row['method']}`, `{row['label']}`"
        lines.append(f"| {names[role]} | {source} | {_verdict(row)} | `{sample.get('oracle_verdict')}` |")
    for role in ("base", "prompt_only", "patched"):
        row = roles[role]
        if row is not None:
            lines += ["", f"**{names[role]} raw output**", "", "````text", str(row.get("raw_output") or "").rstrip(),
                      "````"]
    return "\n".join(lines)


def patch_markdown(payload: dict | None) -> str:
    if not payload:
        return PENDING
    summary = payload.get("patch_summary")
    if not summary:
        return "No patch manifest was included in this build, so no weight-change summary is shown."
    names = summary.get("changed_tensors") or []
    shown = ", ".join(f"`{n}`" for n in names[:12]) + (" ..." if len(names) > 12 else "")
    return "\n".join([
        "### Which weights changed",
        f"- Format: `{summary.get('format')}`",
        f"- Changed scalars: {summary.get('changed_scalars'):,}" if isinstance(summary.get("changed_scalars"), int)
        else "- Changed scalars: not recorded",
        f"- Total parameters: {summary.get('total_parameters'):,}" if isinstance(summary.get("total_parameters"), int)
        else "- Total parameters: not recorded",
        f"- Percent changed: {summary.get('percent_changed')}",
        f"- Tensors changed: {summary.get('tensor_count')}",
        f"- Names (first 12): {shown or 'none'}",
        "",
        "The base model is not included in this Space. The patch changes only the listed slices.",
    ])


def build_demo():
    if gr is None:
        raise RuntimeError("gradio is required to build the demo; install requirements.txt")
    payload = load_examples()
    first_categories = categories(payload)
    with gr.Blocks(title="Aawden demo") as demo:
        gr.Markdown(
            "# Aawden: saved outputs\n"
            "Same security-review code, misleading reviewer. Base Aya versus prompt-only versus patched weights, "
            "scored by an executable oracle. Research artifact: single author, small fixtures, independent review "
            "pending. The base model (CC-BY-NC-4.0, Cohere Labs AUP) is not included."
        )
        if payload is None:
            gr.Markdown(PENDING)
            return demo
        with gr.Row():
            category = gr.Dropdown(choices=first_categories, value="All", label="Category")
            case = gr.Dropdown(choices=case_ids(payload), value=(case_ids(payload) or [None])[0], label="Case")
        view = gr.Markdown(case_markdown(payload, case.value))
        gr.Markdown(patch_markdown(payload))

        def narrow(selected_category):
            ids = case_ids(payload, selected_category)
            return gr.update(choices=ids, value=(ids or [None])[0])

        category.change(narrow, inputs=category, outputs=case)
        case.change(lambda c: case_markdown(payload, c), inputs=case, outputs=view)
    return demo


if __name__ == "__main__":
    build_demo().launch()
