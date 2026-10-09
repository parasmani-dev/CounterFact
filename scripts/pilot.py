"""Real Gemma pilot; no substitute model outputs or hidden retry runs."""

import argparse
import json

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget
from counterfact.config import Settings
from counterfact.runner import EvaluationProfile, run_suite
from counterfact.score import SCORER_VERSION
from counterfact.store import Store
from counterfact.suites import HardSuite, generate_suite


def balanced_order(suite: HardSuite) -> HardSuite:
    selected = []
    for kind in ("largest_category", "value_lookup", "above_threshold"):
        selected.extend([f for f in suite.families if f.spec.question.type == kind][:2])
    chosen = {f.family_id for f in selected}
    remaining = [f for f in suite.families if f.family_id not in chosen]
    return HardSuite.model_validate(
        suite.model_copy(update={"families": selected + remaining}).model_dump()
    )


def print_case(case: dict) -> None:
    def cell(check):
        return (
            "not_run (0/3 correct)"
            if check is None
            else (f"{check['verdict']} ({check['correct_count']}/3 correct)")
        )

    sides = [c for c in (case["original"], case["transformed"]) if c is not None]
    cached = sum(c["cached_count"] for c in sides)
    live = sum(c["live_count"] for c in sides)
    print(
        f"{case['case_id']} | {case['question_type']} | {cell(case['original'])} | "
        f"{cell(case['transformed'])} | {case['pair_status']} | {cached}/{live}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--cap", type=int, default=80)
    args = parser.parse_args()
    if args.limit < 1 or args.cap < 0:
        parser.error("--limit must be positive and --cap nonnegative")
    try:
        config = InferenceConfig.from_env()
        if not config.api_key.strip():
            print("Gemma API key missing. Set GEMMA_API_KEY in the ignored local .env file.")
            return 2
        adapter = OpenAICompatibleAdapter(config)
    except ValueError:
        print(
            "Invalid inference configuration. "
            "Check base URL, model ID, timeout and retries in .env."
        )
        return 2
    settings = Settings.from_env()
    suite_path = settings.data_dir / "suites" / "hard_v1.json"
    if suite_path.exists():
        suite = HardSuite.model_validate_json(suite_path.read_text(encoding="utf-8"))
    else:
        suite = generate_suite(data_dir=settings.data_dir, output_path=suite_path)
    suite = balanced_order(suite)
    print(f"Provider: {adapter.provider}; model: {config.model_id}; fresh: {args.fresh}")
    print(f"Scorer version: {SCORER_VERSION}; balanced sample: 2 largest / 2 lookup / 2 threshold")
    print("case_id | question_type | original | transformed | pair_status | cached/live")
    result = run_suite(
        suite,
        EvaluationProfile("gemma", adapter, Store(settings.data_dir)),
        args.fresh,
        Budget(args.cap),
        args.limit,
        print_case,
    )
    print(result["message"])
    print(
        f"Total attempts: {result['attempts_used']}; budget: "
        f"{result['budget']['used']}/{result['budget']['cap']}"
    )
    print(
        f"Observations cached/live: {result['cached_count']}/{result['live_count']}; "
        f"live requests: {result['live_requests']}"
    )
    print(
        f"Error counts: {json.dumps(result['error_counts'], sort_keys=True)}; "
        f"invalid_output: {result['invalid_output_count']}"
    )
    if result["failures"]:
        print("Observed failures: " + json.dumps(result["failures"], sort_keys=True))
    else:
        print("no failure observed")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
