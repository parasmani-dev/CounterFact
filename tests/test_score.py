import pytest

from counterfact.score import score_response


@pytest.mark.parametrize("fence", ['```json\n{"answer":"Lab A"}\n```', '```{"answer":"Lab A"}```'])
def test_one_outer_code_fence_is_accepted(fence):
    assert score_response(fence, "largest_category", "lab a") == {
        "status": "correct",
        "parsed_answer": "Lab A",
    }


@pytest.mark.parametrize(
    "raw",
    [
        'The answer is {"answer": 1}',
        '{"answer": 1} trailing',
        "{bad",
        "{}",
        '{"answer":1,"reason":"x"}',
    ],
)
def test_non_contract_response_is_invalid(raw):
    assert score_response(raw, "value_lookup", 1)["status"] == "invalid_output"


def test_category_comparison_is_trimmed_and_casefolded():
    assert (
        score_response('{"answer":"  gRoUp a  "}', "largest_category", "Group A")["status"]
        == "correct"
    )


@pytest.mark.parametrize("raw", ['{"answer":true}', '{"answer":"YES"}', '{"answer":"no"}'])
def test_threshold_accepts_json_boolean_and_yes_no(raw):
    expected = False if '"no"' in raw else True
    assert score_response(raw, "above_threshold", expected)["status"] == "correct"


def test_numeric_tolerance_and_boolean_rejection():
    assert score_response('{"answer":100.00005}', "value_lookup", 100)["status"] == "correct"
    assert score_response('{"answer":0.000000001}', "value_lookup", 0)["status"] == "correct"
    assert score_response('{"answer":true}', "value_lookup", 1)["status"] == "incorrect"
    assert score_response('{"answer":100.001}', "value_lookup", 100)["status"] == "incorrect"
