"""Small OpenAI-compatible vision adapter with explicit, bounded retries."""

import base64
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

PROMPT_VERSION = "chart-answer-json-v1"
SYSTEM_PROMPT = 'Answer the question about the chart. Reply with JSON only: {"answer": <value>}.'


@dataclass(frozen=True)
class InferenceConfig:
    base_url: str
    model_id: str
    api_key: str
    timeout_seconds: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        load_dotenv(".env", override=False)
        return cls(
            base_url=os.getenv(
                "GEMMA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            model_id=os.getenv("GEMMA_MODEL_ID", "gemma-4-31b-it"),
            api_key=os.getenv("GEMMA_API_KEY", ""),
            timeout_seconds=float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
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
        if config.timeout_seconds <= 0 or not 0 <= config.max_retries <= 2:
            raise ValueError("Timeout must be positive and MAX_RETRIES must be between 0 and 2")
        self.config = config
        self.endpoint = config.base_url.rstrip("/") + "/chat/completions"
        self.provider = parsed.hostname or "unknown"
        self.transport = transport
        self.sleep_fn = sleep_fn
        self.jitter_fn = jitter_fn

    def complete(self, image: bytes, question: str) -> dict:
        started = time.monotonic()
        attempts = 0
        error_type = "none"
        raw_text = None
        request_params = {
            "model": self.config.model_id,
            "temperature": 0,
            "prompt_version": PROMPT_VERSION,
            "max_tokens": 128,
        }
        payload = {
            "model": self.config.model_id,
            "temperature": 0,
            "max_tokens": 128,
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
        for attempt in range(self.config.max_retries + 1):
            attempts += 1  # reserve/count before dispatch
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
                        error_type = "none"
                        break
                    except (ValueError, KeyError, IndexError, TypeError):
                        error_type = "bad_request"
                retryable = error_type in {"rate_limit", "server"}
            except httpx.TimeoutException:
                error_type, retryable = "timeout", True
            except httpx.RequestError:
                error_type, retryable = "network", True
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
        }
