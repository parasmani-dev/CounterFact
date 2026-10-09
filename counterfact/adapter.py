"""Small OpenAI-compatible vision adapter with explicit, bounded retries."""

import base64
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from counterfact.budget import Budget, BudgetExceeded

PROMPT_VERSION = "chart-answer-json-v1"
SYSTEM_PROMPT = 'Answer the question about the chart. Reply with JSON only: {"answer": <value>}.'


@dataclass(frozen=True)
class InferenceConfig:
    base_url: str
    model_id: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 120.0
    max_retries: int = 2
    config_revision: str = "1"
    thinking_level: str | None = None

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        load_dotenv(".env", override=False)
        base_url = os.getenv(
            "GEMMA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        model_id = os.getenv("GEMMA_MODEL_ID", "gemma-4-31b-it")
        return cls(
            base_url=base_url,
            model_id=model_id,
            api_key=os.getenv("GEMMA_API_KEY", ""),
            timeout_seconds=float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            config_revision=os.getenv("GEMMA_CONFIG_REVISION", "1"),
            thinking_level=(
                "minimal"
                if urlsplit(base_url).hostname == "generativelanguage.googleapis.com"
                and model_id.startswith("gemma-4-")
                else None
            ),
        )


class OpenAICompatibleAdapter:
    def __init__(
        self,
        config: InferenceConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        jitter_fn: Callable[[], float] = random.random,
    ):
        parsed = urlsplit(config.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
        ):
            raise ValueError("GEMMA_BASE_URL must be an HTTP(S) URL without embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("GEMMA_BASE_URL cannot contain a query or fragment")
        if not config.model_id.strip() or not config.api_key.strip():
            raise ValueError("GEMMA_MODEL_ID and GEMMA_API_KEY are required")
        if (
            not math.isfinite(config.timeout_seconds)
            or config.timeout_seconds <= 0
            or not 0 <= config.max_retries <= 2
        ):
            raise ValueError("Timeout must be positive and MAX_RETRIES must be between 0 and 2")
        if not config.config_revision.strip():
            raise ValueError("GEMMA_CONFIG_REVISION cannot be empty")
        if config.thinking_level not in (None, "minimal", "high"):
            raise ValueError("Gemma thinking_level must be minimal or high")
        self.config = config
        self.endpoint = config.base_url.rstrip("/") + "/chat/completions"
        self.provider = parsed.hostname or "unknown"
        self.transport = transport
        self.sleep_fn = sleep_fn
        self.jitter_fn = jitter_fn

    def request_params(self) -> dict:
        params = {
            "model": self.config.model_id,
            "temperature": 0,
            "prompt_version": PROMPT_VERSION,
            "max_tokens": 1024,
            "endpoint": self.endpoint,
            "config_revision": self.config.config_revision,
        }
        if self.config.thinking_level is not None:
            params["extra_body"] = {
                "google": {"thinking_config": {"thinking_level": self.config.thinking_level}}
            }
        return params

    def complete(self, image: bytes, question: str, *, budget: Budget | None = None) -> dict:
        started = time.monotonic()
        attempts = 0
        error_type = "none"
        raw_text = None
        request_params = self.request_params()
        budget = budget if budget is not None else Budget(self.config.max_retries + 1)
        budget_exhausted = False
        error_counts = {}
        response_metadata = {}
        error_message = None
        payload = {
            "model": self.config.model_id,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(image).decode("ascii")
                            },
                        },
                    ],
                },
            ],
        }
        if "extra_body" in request_params:
            payload["extra_body"] = request_params["extra_body"]
        for attempt in range(self.config.max_retries + 1):
            try:
                budget.reserve(1)
            except BudgetExceeded:
                budget_exhausted = True
                if error_type == "none":
                    error_type = "budget"
                break
            attempts += 1
            try:
                with httpx.Client(
                    timeout=self.config.timeout_seconds,
                    transport=self.transport,
                    follow_redirects=False,
                ) as client:
                    response = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.config.api_key}"},
                        json=payload,
                    )
                if response.status_code in (401, 403):
                    error_type = "auth"
                elif response.status_code == 408:
                    error_type = "timeout"
                elif response.status_code == 429:
                    error_type = "rate_limit"
                elif response.status_code >= 500:
                    error_type = "server"
                elif response.status_code >= 400:
                    error_type = "bad_request"
                else:
                    try:
                        data = response.json()
                        raw_text = data["choices"][0]["message"]["content"]
                        if not isinstance(raw_text, str):
                            raise (KeyError("content"))
                        response_metadata = {
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "usage": data.get("usage"),
                        }
                        error_type = "none"
                        break
                    except (ValueError, KeyError, IndexError, TypeError):
                        error_type = "bad_request"
                if response.status_code >= 400:
                    try:
                        message = response.json().get("error", {}).get("message", "")
                        if isinstance(message, str):
                            error_message = message.replace(self.config.api_key, "[REDACTED]")[
                                :1000
                            ]
                    except (ValueError, AttributeError):
                        pass
                retryable = error_type in {"rate_limit", "server", "timeout"}
            except httpx.TimeoutException:
                error_type, retryable = "timeout", True
            except httpx.RequestError:
                error_type, retryable = "network", True
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            if not retryable or attempt >= self.config.max_retries:
                break
            self.sleep_fn(min(2**attempt, 4) + self.jitter_fn() * 0.25)
        return {
            "ok": error_type == "none",
            "raw_text": raw_text,
            "error_type": error_type,
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "attempts_used": attempts,
            "provider": self.provider,
            "model": self.config.model_id,
            "request_params": request_params,
            "budget_exhausted": budget_exhausted,
            "error_counts": error_counts,
            "response_metadata": response_metadata,
            "error_message": error_message,
        }
