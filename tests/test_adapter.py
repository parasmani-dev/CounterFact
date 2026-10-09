import json

import httpx
import pytest

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter


class FakeTransportForTests(httpx.MockTransport):
    """Synthetic HTTP responses used only by adapter unit tests."""


def config(retries=0):
    return InferenceConfig("https://api.example/v1", "open-model", "test-secret", 1, retries)


def adapter_for(status, retries=0):
    return OpenAICompatibleAdapter(
        config(retries),
        transport=FakeTransportForTests(lambda request: httpx.Response(status)),
        sleep_fn=lambda _: None,
        jitter_fn=lambda: 0,
    )


@pytest.mark.parametrize(
    "status,error",
    [
        (401, "auth"),
        (403, "auth"),
        (408, "timeout"),
        (429, "rate_limit"),
        (500, "server"),
        (400, "bad_request"),
    ],
)
def test_http_error_classification(status, error):
    result = adapter_for(status).complete(b"png", "question")
    assert result["error_type"] == error
    assert not result["ok"]
    assert "test-secret" not in json.dumps(result)


def test_retry_count_is_bounded_and_counted_before_dispatch():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    adapter = OpenAICompatibleAdapter(
        config(2),
        transport=FakeTransportForTests(handler),
        sleep_fn=lambda _: None,
        jitter_fn=lambda: 0,
    )
    result = adapter.complete(b"png", "question")
    assert len(calls) == result["attempts_used"] == 3


def test_success_uses_fixed_prompt_image_data_url_and_zero_temperature():
    captured = {}

    def handler(request):
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "A"}'}}]})

    result = OpenAICompatibleAdapter(config(), transport=FakeTransportForTests(handler)).complete(
        b"png", "Question"
    )
    assert result["ok"] is True
    assert result["raw_text"] == '{"answer": "A"}'
    assert captured["auth"] == "Bearer test-secret"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["messages"][0]["content"].startswith("Answer the question")
    image_url = captured["body"]["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_timeout_and_network_failures_retry():
    for error_class, expected in ((httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "network")):
        calls = []

        def handler(request, error_type=error_class, request_log=calls):
            request_log.append(request)
            raise error_type("synthetic test error")

        adapter = OpenAICompatibleAdapter(
            config(1), transport=FakeTransportForTests(handler), sleep_fn=lambda _: None
        )
        result = adapter.complete(b"png", "question")
        assert result["error_type"] == expected
        assert result["attempts_used"] == len(calls) == 2


def test_adapter_rejects_more_than_two_retries():
    with pytest.raises(ValueError, match="MAX_RETRIES"):
        OpenAICompatibleAdapter(config(3))
