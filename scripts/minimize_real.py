"""One real reduction run, selected only from stored strict-scored failure evidence."""

import json

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget
from counterfact.config import Settings
from counterfact.evidence import stored_failure
from counterfact.minimize import minimize
from counterfact.store import Store
from counterfact.suites import HardSuite


def main():
    settings = Settings.from_env()
    store = Store(settings.data_dir)
    store.initialize()
    suite = HardSuite.model_validate_json((settings.data_dir / "suites/hard_v1.json").read_text())
    selected = stored_failure(store, suite)
    if not selected:
        print("no failure observed, nothing to minimize")
        return 0
    config = InferenceConfig.from_env()
    if not config.api_key:
        print("Gemma API key missing: set GEMMA_API_KEY in the ignored local .env file.")
        return 2
    print("Selected:", json.dumps({k: v for k, v in selected.items() if k != "case"}))
    result = minimize(
        selected["case"],
        selected["predicate"],
        OpenAICompatibleAdapter(config),
        store,
        Budget(60),
        max_calls=60,
        on_step=lambda step: print("Step:", json.dumps(step, sort_keys=True), flush=True),
    )
    print("Final report:", json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
