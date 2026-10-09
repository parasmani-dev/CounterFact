"""Sequential suite evaluation with durable partial results and a hard request budget."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget
from counterfact.check import run_check
from counterfact.config import Settings
from counterfact.minimize import DispatchStopped
from counterfact.reference import answer, question_text
from counterfact.score import SCORER_VERSION
from counterfact.store import Store
from counterfact.suites import HardSuite


@dataclass
class EvaluationProfile:
    name: str
    adapter: OpenAICompatibleAdapter
    store: Store


def pair_status(original: str, transformed: str) -> str:
    if "inconclusive" in (original, transformed):
        return "inconclusive"
    if original == "passes" and transformed == "fails":
        return "correct_to_incorrect"
    if original == transformed == "fails":
        return "both_fail"
    if original == transformed == "passes":
        return "both_pass"
    return "mixed"


def run_suite(
    suite: HardSuite,
    profile: EvaluationProfile | str,
    fresh: bool,
    budget: Budget,
    limit: int | None = None,
    on_progress: Callable[[dict], None] | None = None,
    *,
    run_id: str | None = None,
    cancel_flag=None,
) -> dict:
    if isinstance(profile, str):
        if profile != "gemma":
            raise ValueError("Only the gemma profile is configured in Phase 2")
        profile = EvaluationProfile(
            "gemma",
            OpenAICompatibleAdapter(InferenceConfig.from_env()),
            Store(Settings.from_env().data_dir),
        )
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    store = profile.store
    store.initialize()
    if run_id is None:
        run_id = store.start_run(suite.suite_id, profile.name, fresh, budget.report())
    else:
        with store.connect() as connection:
            connection.execute("UPDATE run SET status='running' WHERE id=?", (run_id,))
    results = []
    errors = Counter()
    cached_count = live_count = live_requests = invalid_count = 0
    attempts_start = budget.used
    status, message = "completed", "Suite completed."
    failures = []
    try:
        for family in suite.families[:limit]:
            if cancel_flag is not None and cancel_flag.is_set():
                status, message = "cancelled", "Run cancelled; partial results saved."
                break
            manifest = store.create(family.spec)
            sides = {}
            for side, chart in (
                ("original", family.spec.chart),
                ("transformed", family.spec.transformed()),
            ):
                image = (store.root / "pairs" / manifest["id"] / f"{side}.png").read_bytes()
                try:
                    check = run_check(
                        image,
                        question_text(chart, family.spec.question),
                        answer(chart, family.spec.question),
                        family.spec.question.type,
                        profile.adapter,
                        store,
                        budget,
                        fresh=fresh,
                    )
                except DispatchStopped:
                    status, message = "cancelled", "Run cancelled; partial results saved."
                    break
                sides[side] = check
                store.save_case(run_id, family.family_id, side, check, budget.report())
                errors.update(check["errors"])
                cached_count += check["cached_count"]
                live_count += check["live_count"]
                live_requests += check["live_requests"]
                invalid_count += check["invalid_output_count"]
                if check["verdict"] == "fails":
                    failures.append(
                        {
                            "case_id": family.family_id,
                            "side": side,
                            "incorrect_count": check["incorrect_count"],
                        }
                    )
                if check["terminal_error"] == "rate_limit":
                    status, message = (
                        "rate_limited",
                        "Persistent rate limit; stopped with partial results saved.",
                    )
                elif check["terminal_error"] == "cancelled":
                    status, message = "cancelled", "Run cancelled; partial results saved."
                elif check["budget_exhausted"]:
                    status, message = (
                        "budget_exhausted",
                        "Attempt cap reached; partial results saved.",
                    )
                elif check["terminal_error"] in {
                    "auth",
                    "bad_request",
                    "network",
                    "timeout",
                    "server",
                }:
                    status, message = (
                        "provider_error",
                        "Provider request failed after bounded attempts; partial results saved.",
                    )
                if status != "completed":
                    break
            if "original" not in sides:
                break
            original = sides["original"]
            transformed = sides.get("transformed")
            result = {
                "case_id": family.family_id,
                "question_type": family.spec.question.type,
                "artifacts": {m["variant"]: m["image_hash"] for m in manifest["members"]},
                "original": original,
                "transformed": transformed,
                "pair_status": pair_status(
                    original["verdict"], transformed["verdict"] if transformed else "inconclusive"
                ),
                "single_image_failure": original["verdict"] == "fails",
            }
            results.append(result)
            if on_progress is not None:
                on_progress(result)
            if status != "completed":
                break
        if status == "completed" and any(r["pair_status"] == "inconclusive" for r in results):
            status, message = "inconclusive", "Suite finished with incomplete observation groups."
    except Exception:
        store.finish_run(run_id, "error", budget.report())
        raise
    store.finish_run(
        run_id,
        status,
        {
            **budget.report(),
            "error_counts": dict(errors),
            "cached_count": cached_count,
            "live_count": live_count,
            "live_requests": live_requests,
            "scorer_version": SCORER_VERSION,
        },
    )
    return {
        "run_id": run_id,
        "status": status,
        "message": message,
        "cases": results,
        "attempts_used": budget.used - attempts_start,
        "budget": budget.report(),
        "cached_count": cached_count,
        "live_count": live_count,
        "live_requests": live_requests,
        "error_counts": dict(errors),
        "invalid_output_count": invalid_count,
        "failures": failures,
    }
