"""Offline guards for the investigated threshold family's evaluation contract."""

import pytest

from counterfact.reference import answer, question_text
from counterfact.runner import pair_status
from counterfact.schemas import PairSpec
from counterfact.score import SCORER_VERSION, score_response
from counterfact.suites import build_suite


def failure_spec():
    family = next(
        f for f in build_suite(seed=42).families if f.family_id == "hard-above_threshold-01"
    )
    return PairSpec.model_validate(family.spec.model_dump())


def test_threshold_family_keeps_question_and_changes_only_target_value():
    spec = failure_spec()
    original, transformed = spec.chart, spec.transformed()
    assert spec.question.target_id == "group_10"
    assert spec.question.threshold == 1050.4
    assert not spec.show_value_labels
    assert question_text(original, spec.question) == question_text(transformed, spec.question)
    assert question_text(original, spec.question) == (
        'Is the value of category "Group J" strictly greater than 1050.4?'
    )
    assert [c.id for c in original.categories] == [c.id for c in transformed.categories]
    changes = [
        (a, b) for a, b in zip(original.categories, transformed.categories, strict=True) if a != b
    ]
    assert len(changes) == 1
    before, after = changes[0]
    assert before.id == after.id == "group_10"
    assert before.label == after.label == "Group J"
    assert before.value == 1040.0
    assert after.value == 1071.2
    assert answer(original, spec.question) is False
    assert answer(transformed, spec.question) is True


@pytest.mark.parametrize("expected,status", [(False, "incorrect"), (True, "correct")])
def test_same_yes_response_is_scored_against_each_sides_ground_truth(expected, status):
    # Parser input only: this is not a model acquisition or demo result.
    result = score_response('```json\n{"answer": "yes"}\n```', "above_threshold", expected)
    assert SCORER_VERSION == "1"
    assert result == {"parsed_answer": True, "status": status}


def test_threshold_equality_is_false_for_this_target():
    spec = failure_spec()
    payload = spec.model_dump()
    payload["question"]["threshold"] = 1040.0
    boundary = PairSpec.model_validate(payload)
    assert answer(boundary.chart, boundary.question) is False
    assert answer(boundary.transformed(), boundary.question) is True


def test_failure_followed_by_pass_is_not_correct_to_incorrect():
    assert pair_status("fails", "passes") == "mixed"
    assert pair_status("passes", "fails") == "correct_to_incorrect"
    assert pair_status("fails", "inconclusive") == "inconclusive"
