# Current State
- `preflight.py` is configured and passing.
- Google Generative Language API is reachable and correctly configured.
- `max_tokens` was increased to 1024 in `counterfact/adapter.py` to allow reasoning models enough space for generating `<thought>` blocks.
- `counterfact/score.py` was updated to correctly extract JSON answers from models that output `<thought>` blocks or other text around the JSON markdown fences.

# Next Steps
- Move forward with running suite evaluations or adding new benchmark suites as needed.
