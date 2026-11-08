"""Strict verdict parsing: invalid responses remain in the denominator."""
import json
import math
from .data import VERDICTS


def strict_json(text: str):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique)


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

