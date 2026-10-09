import copy

import pytest
from pydantic import ValidationError

from counterfact.reference import answer, question_text
from counterfact.schemas import PairSpec


def test_changed_winner_has_computable_answers(pair_payload):
    spec = PairSpec.model_validate(pair_payload)
    assert answer(spec.chart, spec.question) == "Lab B"
    assert answer(spec.transformed(), spec.question) == "Lab A"
    assert spec.chart.categories[0].value == 40  # Source remains unchanged.


@pytest.mark.parametrize(
    "kind,target,threshold,expected",
    [
        ("value_lookup", "lab_a", None, 40),
        ("above_threshold", "lab_a", 40, False),
        ("above_threshold", "lab_b", 40, True),
    ],
)
def test_three_question_types_and_strict_threshold(pair_payload, kind, target, threshold, expected):
    question = {"type": kind, "target_id": target}
    if threshold is not None:
        question["threshold"] = threshold
    pair_payload["question"] = question
    spec = PairSpec.model_validate(pair_payload)
    assert answer(spec.chart, spec.question) == expected
    assert spec.chart.categories[[c.id for c in spec.chart.categories].index(target)].label in (
        question_text(spec.chart, spec.question)
    )


def test_reorder_preserves_label_value_binding(pair_payload):
    pair_payload["mutations"] = [{"type": "reorder", "order": ["lab_c", "lab_a", "lab_b"]}]
    spec = PairSpec.model_validate(pair_payload)
    assert [c.value for c in spec.transformed().categories] == [25, 40, 60]
    assert answer(spec.transformed(), spec.question) == answer(spec.chart, spec.question)


def test_value_change_can_preserve_answer(pair_payload):
    pair_payload["mutations"][0]["value"] = 45
    spec = PairSpec.model_validate(pair_payload)
    assert answer(spec.chart, spec.question) == answer(spec.transformed(), spec.question)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 1_000_001, True, "40"])
def test_rejects_invalid_numbers(pair_payload, bad):
    pair_payload["chart"]["categories"][0]["value"] = bad
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)


@pytest.mark.parametrize(
    "mutation",
    [
        {"type": "change_value", "category_id": "missing", "value": 3},
        {"type": "change_value", "category_id": "lab_a", "value": 40},
        {"type": "change_value", "category_id": "lab_a", "value": 60},  # Tied maxima.
        {"type": "reorder", "order": ["lab_a", "lab_b", "lab_b"]},
        {"type": "reorder", "order": ["lab_a", "lab_b"]},
        {"type": "reorder", "order": ["lab_a", "lab_b", "lab_c"]},
    ],
)
def test_rejects_invalid_mutations(pair_payload, mutation):
    pair_payload["mutations"] = [mutation]
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "value_lookup", "target_id": "missing"},
        {"type": "largest_category", "target_id": "lab_a"},
        {"type": "above_threshold", "target_id": "lab_a"},
        {"type": "value_lookup", "target_id": "lab_a", "threshold": 4},
    ],
)
def test_rejects_invalid_question_bindings(pair_payload, question):
    pair_payload["question"] = question
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)


@pytest.mark.parametrize("field,value", [("id", "lab_a"), ("label", "LAB A")])
def test_rejects_duplicate_categories(pair_payload, field, value):
    pair_payload["chart"]["categories"][1][field] = value
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)


def test_rejects_empty_and_net_identity_transforms(pair_payload):
    for mutations in (
        [],
        [
            {"type": "change_value", "category_id": "lab_a", "value": 80},
            {"type": "change_value", "category_id": "lab_a", "value": 40},
        ],
    ):
        candidate = copy.deepcopy(pair_payload)
        candidate["mutations"] = mutations
        with pytest.raises(ValidationError):
            PairSpec.model_validate(candidate)


@pytest.mark.parametrize("label", ["", "a" * 41, "Lab\nA", "数学"])
def test_rejects_unrenderable_labels(pair_payload, label):
    pair_payload["chart"]["categories"][0]["label"] = label
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)


def test_rejects_unknown_fields_and_chart_types(pair_payload):
    pair_payload["api_key"] = "test-only-sensitive-value"
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)
    del pair_payload["api_key"]
    pair_payload["chart"]["type"] = "pie"
    with pytest.raises(ValidationError):
        PairSpec.model_validate(pair_payload)
