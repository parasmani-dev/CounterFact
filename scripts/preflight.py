"""Make the single, bounded real-model call used to verify local configuration."""

import json
import sys
from pathlib import Path

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.config import Settings
from counterfact.reference import answer, question_text
from counterfact.score import score_response


def main() -> int:
    config = InferenceConfig.from_env()
    if not config.api_key.strip():
        print(
            "Gemma API key is missing. Add GEMMA_API_KEY=your_key to the local .env file, "
            "then rerun scripts/preflight.py."
        )
        return 2
    suite_path = Path("data/suites/hard_v1.json")
    if not suite_path.is_file():
        print("Hard suite is missing. Run: python scripts/generate_suite.py")
        return 2
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    family = next(
        f for f in suite["families"] if f["spec"]["question"]["type"] == "largest_category"
    )
    if not family.get("pair_id"):
        print("Hard suite has no rendered pair IDs. Regenerate it with scripts/generate_suite.py.")
        return 2
    settings = Settings.from_env()
    image_path = settings.data_dir / "pairs" / family["pair_id"] / "original.png"
    if not image_path.is_file():
        print("Suite chart image is missing. Run: python scripts/generate_suite.py")
        return 2
    spec = family["spec"]
    from counterfact.schemas import PairSpec

    pair_spec = PairSpec.model_validate(spec)
    question = question_text(pair_spec.chart, pair_spec.question)
    result = OpenAICompatibleAdapter(config).complete(image_path.read_bytes(), question)
    print(f"Raw response: {result['raw_text'] if result['raw_text'] is not None else '<none>'}")
    score = score_response(
        result["raw_text"] or "",
        pair_spec.question.type,
        answer(pair_spec.chart, pair_spec.question),
    )
    print(f"Parsed score: {json.dumps(score, sort_keys=True)}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Attempts: {result['attempts_used']}")
    print(f"Provider/model: {result['provider']} / {result['model']}")
    print(f"Request params: {json.dumps(result['request_params'], sort_keys=True)}")
    print(f"Response metadata: {json.dumps(result['response_metadata'], sort_keys=True)}")
    if not result["ok"]:
        print(f"Preflight error: {result['error_type']}")
        print(f"Provider error message: {result['error_message'] or '<none>'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
