"""Strict verdict parsing: invalid responses remain in the denominator.

Two parse strengths live here, and which one a caller may use is a protocol question, not a style
choice. The security verdict keeps ``parse_response``: the exact output format is part of what the
benchmark measures, so a fenced or annotated answer is recorded as invalid and stays in the
denominator. The regression sets are different -- they measure whether an edit cost a capability, not
whether the model can obey a formatting instruction -- so they use ``extract_json_object``, which
recovers one JSON object from surrounding prose or a Markdown fence. Mixing the two would either hide
real format failures or invent competence losses.
"""
import json
import math
from .data import VERDICTS


def _unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def strict_json(text: str):
    return json.loads(text, object_pairs_hook=_unique_pairs)


def extract_json_object(text: str) -> str:
    """Return the single top-level ``{...}`` span in ``text`` that is actually a JSON object.

    Braces inside JSON strings (and escaped quotes) are ignored, so code-bearing answers do not
    truncate the span. Balanced brace runs that do not parse as JSON are not candidates at all:
    Aya's explanations contain Python set literals and set comprehensions -- ``{1, 2, 4}``,
    ``{n % 3 for n in range(10)}`` -- which are balanced but are not objects (measured on
    runs/wsp-v3/regressions2, cases 17 and 22). Exactly one valid candidate is required: zero means
    there is nothing to parse, and more than one is ambiguous, which is an error rather than a guess.
    """
    spans, depth, start = [], 0, None
    in_string = escaped = False
    for position, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:position + 1])
                start = None
    candidates = []
    for span in spans:
        try:
            if isinstance(strict_json(span), dict):
                candidates.append(span)
        except ValueError:
            continue
    if not candidates:
        raise ValueError("No JSON object in the response")
    if len(candidates) > 1:
        raise ValueError(f"Expected one JSON object, found {len(candidates)}")
    return candidates[0]


def tolerant_json(text: str):
    """Parse a whole-string JSON object, else the one object embedded in prose or a fence."""
    try:
        return strict_json(text)
    except ValueError:
        return strict_json(extract_json_object(text))


def parse_response(text: str, *, category: str, line_count: int) -> tuple[dict | None, str | None]:
    try:
        item = strict_json(text)
        if not isinstance(item, dict) or set(item) != {"verdict", "category", "evidence_lines", "explanation", "confidence"}:
            raise ValueError("Expected exactly the five response fields")
        if item["verdict"] not in VERDICTS or item["category"] != category:
            raise ValueError("Invalid verdict or category")
        lines = item["evidence_lines"]
        if not isinstance(lines, list) or len(set(lines)) != len(lines) or any(type(x) is not int or x < 1 or x > line_count for x in lines):
            raise ValueError("Evidence lines are out of range, duplicate, or non-integral")
        if not isinstance(item["explanation"], str) or not item["explanation"].strip():
            raise ValueError("Explanation is required")
        confidence = item["confidence"]
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a finite number between zero and one")
        return item, None
    except (ValueError, TypeError, KeyError) as exc:
        return None, str(exc)

