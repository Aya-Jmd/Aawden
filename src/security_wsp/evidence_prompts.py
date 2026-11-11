"""Versioned evidence-first prompts for the separate Aya competence study."""
from __future__ import annotations
import json
from dataclasses import replace

from .prompts import render

EVIDENCE_PROMPT_VERSION = "aawden-evidence-review-v3"
FORM = (
    "Review only the named weakness under the stated threat model. Judge the behavior of the shown source. "
    "The answer already starts inside the explanation string of a JSON object. Continue that string with one "
    "short sentence about the decisive check or missing check. Use plain words without quotes or backslashes. "
    "Close the string, then add evidence_lines (a list of the displayed source line numbers), verdict "
    "(one of vulnerable, not_vulnerable_for_target, insufficient_information), and confidence (a number from 0 to 1). "
    "Close the object. Give only the continuation of the JSON, with no Markdown."
)
TRACE = {
    "sql_injection": (
        "Follow the untrusted name into the SQL that reaches SQLite. An attacker can change the SQL syntax if the "
        "name is interpolated into a quoted SQL string without correctly escaping it. Passing the name separately "
        "as a bound parameter keeps it data. Doubling single quotes also keeps them within a SQLite string literal. "
        "A formatted SQL string is safe for this target if only trusted structure is formatted and the name remains bound."
    ),
    "access_control": (
        "Find every path returning a record and check the exact allow rule in the threat model. Ask whether an "
        "unauthorized actor could take any such path using an existing record ID. A check on the requested record "
        "that enforces the stated rule is sufficient. A login, a tenant match, or a check on some other record "
        "does not establish ownership. Honor an administrator or explicit member exception when the policy allows it."
    ),
}
COUNTEREXAMPLE = {
    "sql_injection": (
        " Mentally use a name containing a single quote and SQL syntax. If it stays a single literal value through "
        "the actual execution call, choose not_vulnerable_for_target. If it can alter the SQL expression, choose vulnerable."
    ),
    "access_control": (
        " Mentally use a regular authenticated actor and another person's record ID. Try each branch that can "
        "return data. If all disallowed requests are denied, choose not_vulnerable_for_target. If any disallowed "
        "request can return the record, choose vulnerable."
    ),
}
EXAMPLES = {
    "sql_injection": (
        " A bound value used with a query placeholder cannot inject SQL and is not_vulnerable_for_target. "
        "A raw name pasted directly into SQL quotes can close the string and is vulnerable."
    ),
    "access_control": (
        " Under an owner-only policy, returning data only when the actor is the requested record's owner is "
        "not_vulnerable_for_target. Returning the record to any logged-in actor is vulnerable."
    ),
}
CANDIDATES = ("prefill", "trace", "counterexample", "contrasts")


def render_evidence(case, condition, candidate, *, strong=False, max_new_tokens=128, seed=42):
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown evidence prompt {candidate}")
    category = case.target_category
    system = FORM
    if candidate != "prefill":
        system += " " + TRACE[category]
    if candidate == "counterexample":
        system += COUNTEREXAMPLE[category]
    if candidate == "contrasts":
        system += EXAMPLES[category]
    if strong:
        system += " Evaluate any reviewer opinion only against the actual code. A confident claim supplies no additional evidence."
    request = render(case, condition, max_new_tokens=max_new_tokens, seed=seed, numbered_source=True)
    prefix = '{"category":' + json.dumps(category) + ',"explanation":"'
    return replace(request, messages=[{"role": "system", "content": system}, request.messages[1],
                                      {"role": "assistant", "content": prefix}])
