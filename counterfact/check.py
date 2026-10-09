"""Three independent observations with a deterministic vote and durable cache."""

import hashlib
import json
import uuid
from collections import Counter

from counterfact.adapter import PROMPT_VERSION, OpenAICompatibleAdapter
from counterfact.budget import Budget, DispatchStopped
from counterfact.score import SCORER_VERSION, score_response
from counterfact.store import Store


def generation_params(adapter: OpenAICompatibleAdapter) -> dict:
    allowed = {
        "model",
        "temperature",
        "prompt_version",
        "max_tokens",
        "endpoint",
        "config_revision",
        "extra_body",
    }
    return {key: value for key, value in adapter.request_params().items() if key in allowed}


def cache_key(image_bytes: bytes, question: str, adapter: OpenAICompatibleAdapter) -> str:
    params = generation_params(adapter)
    identity = {
        "question": question.strip(),
        "prompt_version": PROMPT_VERSION,
        "generation_config": params,
        "provider": adapter.provider,
        "model": adapter.config.model_id,
        "config_revision": adapter.config.config_revision,
    }
    digest = hashlib.sha256(image_bytes)
    digest.update(b"\0" + json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def run_check(
    image_bytes: bytes,
    question: str,
    expected_answer: str | float | bool,
    question_type: str,
    adapter: OpenAICompatibleAdapter,
    store: Store,
    budget: Budget,
    fresh: bool = False,
) -> dict:
    base_key = cache_key(image_bytes, question, adapter)
    # A fresh acquisition gets its own namespace: old observations and timestamps survive.
    group_key = base_key + ":" + uuid.uuid4().hex if fresh else base_key
    cached = {} if fresh else store.cached_slots(base_key)
    observations = [None, None, None]
    for slot, row in cached.items():
        derived = score_response(row["raw_text"], question_type, expected_answer)
        observations[slot] = {
            **row,
            "acquisition_status": row["status"],
            "status": derived["status"],
            "scorer_version": SCORER_VERSION,
            "cached": True,
        }
    cached_count = len(cached)
    live_count = attempts_used = live_requests = 0
    errors = Counter()
    budget_exhausted = False
    terminal_error = "none"
    for slot in range(3):
        if slot in cached:
            continue
        before = budget.used
        try:
            result = adapter.complete(image_bytes, question.strip(), budget=budget)
        except DispatchStopped as exc:
            terminal_error = exc.reason
            attempts_used += budget.used - before
            live_requests += int(budget.used > before)
            break
        attempts_used += result["attempts_used"]
        live_requests += int(result["attempts_used"] > 0)
        budget_exhausted = result.get("budget_exhausted", False)
        errors.update(result.get("error_counts", {}))
        if not result["ok"]:
            terminal_error = result["error_type"]
            if not result.get("error_counts"):
                errors[terminal_error] += 1
            break
        score = score_response(result["raw_text"], question_type, expected_answer)
        observation = store.save_observation(
            {
                "group_key": group_key,
                "slot": slot,
                "raw_text": result["raw_text"],
                "status": score["status"],
                "error_type": "none",
                "provider": adapter.provider,
                "model": adapter.config.model_id,
                "latency_ms": result["latency_ms"],
                "request_params_json": json.dumps(
                    {
                        **generation_params(adapter),
                        "response_metadata": result.get("response_metadata", {}),
                        "scorer_version": SCORER_VERSION,
                    },
                    sort_keys=True,
                ),
            }
        )
        observations[slot] = {**observation, "cached": False, "scorer_version": SCORER_VERSION}
        live_count += 1
    counts = Counter(o["status"] for o in observations if o is not None)
    verdict = "inconclusive"
    if all(o is not None for o in observations):
        if counts["incorrect"] >= 2:
            verdict = "fails"
        elif counts["correct"] >= 2:
            verdict = "passes"
        else:
            verdict = "mixed"
    return {
        "group_key": group_key,
        "verdict": verdict,
        "observations": observations,
        "cached_count": cached_count,
        "live_count": live_count,
        "live_requests": live_requests,
        "attempts_used": attempts_used,
        "correct_count": counts["correct"],
        "incorrect_count": counts["incorrect"],
        "invalid_output_count": counts["invalid_output"],
        "errors": dict(errors),
        "budget_exhausted": budget_exhausted,
        "terminal_error": terminal_error,
        "scorer_version": SCORER_VERSION,
    }
