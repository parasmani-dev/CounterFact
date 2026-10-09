"""Deterministic scoring for the versioned JSON answer contract."""

import json
import math
import re
from typing import Any

SCORER_VERSION = "1"

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```\Z", re.DOTALL | re.IGNORECASE)


def _parse(raw_text: str) -> Any:
    text = raw_text.strip()
    match = _FENCE.fullmatch(text)
    if match:
        text = match.group(1).strip()

    def reject_constant(value):
        raise ValueError("Nonfinite constants are not JSON")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON keys are ambiguous")
            obj[key] = value
        return obj

    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {"answer"}:
        raise ValueError("response must be an object with only an answer key")
    return value["answer"]


def score_response(raw_text: str, question_type: str, expected_answer: Any) -> dict[str, Any]:
    """Parse without repairs and score against reference logic."""
    try:
        parsed = _parse(raw_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"status": "invalid_output", "parsed_answer": None}

    correct = False
    if question_type == "largest_category":
        correct = (
            isinstance(parsed, str)
            and isinstance(expected_answer, str)
            and parsed.strip().casefold() == expected_answer.strip().casefold()
        )
    elif question_type == "value_lookup":
        correct = (
            isinstance(parsed, (int, float))
            and not isinstance(parsed, bool)
            and isinstance(expected_answer, (int, float))
            and not isinstance(expected_answer, bool)
            and math.isclose(float(parsed), float(expected_answer), rel_tol=1e-6, abs_tol=1e-9)
        )
    elif question_type == "above_threshold":
        normalized = parsed.casefold() if isinstance(parsed, str) else parsed
        if normalized is True or normalized == "yes":
            parsed = True
        elif normalized is False or normalized == "no":
            parsed = False
        correct = (
            isinstance(parsed, bool)
            and isinstance(expected_answer, bool)
            and parsed is expected_answer
        )
    else:
        return {"status": "invalid_output", "parsed_answer": parsed}
    return {"status": "correct" if correct else "incorrect", "parsed_answer": parsed}
