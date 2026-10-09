import json

import httpx

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget
from counterfact.check import cache_key, run_check
from counterfact.score import SCORER_VERSION, score_response
from counterfact.store import Store
from counterfact.suites import build_suite
from scripts.pilot import balanced_order
from tests.test_adapter import FakeTransportForTests
from tests.test_phase2 import FakeAdapterForTests


def test_google_thinking_minimal_is_sent_and_metadata_returned():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"A"}'}}],
                "usage": {"completion_tokens": 12},
            },
        )

    adapter = OpenAICompatibleAdapter(
        InferenceConfig(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemma-4-31b-it",
            "secret",
            thinking_level="minimal",
        ),
        transport=FakeTransportForTests(handler),
    )
    result = adapter.complete(b"pixels", "question")
    assert (
        captured[0]["extra_body"]
        == result["request_params"]["extra_body"]
        == {"google": {"thinking_config": {"thinking_level": "minimal"}}}
    )
    assert result["response_metadata"] == {
        "finish_reason": "stop",
        "usage": {"completion_tokens": 12},
    }
    assert adapter.config.timeout_seconds == 120
    assert captured[0]["max_tokens"] == 1024


def test_rejected_thinking_parameter_is_not_removed_or_retried():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "Parameter rejected; secret"}})

    adapter = OpenAICompatibleAdapter(
        InferenceConfig("https://test.example/v1", "test", "secret", thinking_level="minimal"),
        transport=FakeTransportForTests(handler),
    )
    result = adapter.complete(b"pixels", "question")
    assert len(requests) == result["attempts_used"] == 1
    assert "extra_body" in requests[0]
    assert result["error_type"] == "bad_request"
    assert result["error_message"] == "Parameter rejected; [REDACTED]"


def test_thinking_parameter_changes_cache_identity():
    first = OpenAICompatibleAdapter(InferenceConfig("https://test.example/v1", "test", "secret"))
    second = OpenAICompatibleAdapter(
        InferenceConfig("https://test.example/v1", "test", "secret", thinking_level="minimal")
    )
    assert cache_key(b"pixels", "question", first) != cache_key(b"pixels", "question", second)


def test_balanced_sample_keeps_all_families_and_their_specs():
    original = build_suite()
    balanced = balanced_order(original)
    assert [f.spec.question.type for f in balanced.families[:6]] == [
        "largest_category",
        "largest_category",
        "value_lookup",
        "value_lookup",
        "above_threshold",
        "above_threshold",
    ]
    assert {f.family_id for f in original.families} == {f.family_id for f in balanced.families}
    assert len(balanced.families) == 20
    assert all(
        next(f for f in original.families if f.family_id == b.family_id) == b
        for b in balanced.families
    )


def test_cache_scores_are_computed_on_read_without_overwriting_old_rows(tmp_path):
    store = Store(tmp_path)
    store.initialize()
    adapter = FakeAdapterForTests()
    first = run_check(b"pixels", "question", "A", "largest_category", adapter, store, Budget(3))
    with store.connect() as connection:
        connection.execute("UPDATE observation SET status='invalid_output'")
    replay = run_check(b"pixels", "question", "A", "largest_category", adapter, store, Budget(0))
    assert replay["verdict"] == "passes"
    assert replay["scorer_version"] == SCORER_VERSION
    assert all(o["acquisition_status"] == "invalid_output" for o in replay["observations"])
    assert all(o["scorer_version"] == SCORER_VERSION for o in replay["observations"])
    assert [o["id"] for o in first["observations"]] == [o["id"] for o in replay["observations"]]
    with store.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM observation WHERE status='invalid_output'"
            ).fetchone()[0]
            == 3
        )


def test_option_a_keeps_strict_scorer_unchanged():
    assert SCORER_VERSION == "1"
    assert (
        score_response('<thought>closed</thought>{"answer":"A"}', "largest_category", "A")["status"]
        == "invalid_output"
    )


def test_offline_rescore_keeps_raw_rows_and_makes_no_adapter_calls(tmp_path):
    from counterfact.runner import EvaluationProfile, run_suite
    from scripts.rescore import rescore_stored

    store = Store(tmp_path)
    suite = build_suite()
    adapter = FakeAdapterForTests(['<thought>closed</thought>{"answer":"A"}'] * 6)
    run_suite(suite, EvaluationProfile("test", adapter, store), False, Budget(6), limit=1)
    with store.connect() as connection:
        before = [tuple(r) for r in connection.execute("SELECT * FROM observation ORDER BY id")]
    result = rescore_stored(store, suite)
    with store.connect() as connection:
        after = [tuple(r) for r in connection.execute("SELECT * FROM observation ORDER BY id")]
    assert before == after
    assert adapter.calls == 6
    assert result["network_calls"] == 0
    assert result["observation_counts"] == {"invalid_output": 6}
    assert result["verdict_counts"] == {"mixed": 2}
