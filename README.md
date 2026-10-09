# CounterFact

Turn repeatable visual AI failures into smaller, valid regression tests.

**Phases 1 and 1b are implemented:** validated synthetic bar-chart pairs, a seeded
20-family hard suite, deterministic answer scoring, an OpenAI-compatible vision adapter,
bounded retries and a one-call preflight command. Charts and reference answers are
synthetic test inputs; only a successful preflight can provide model evidence.
The minimizer and comparison workflow have not started.

## Setup

Python 3.11+ is required. Phase 1 was tested on Python 3.14.0 on Windows;
all source files also pass a Python 3.11 grammar check (not a 3.11 runtime test).
Runtime packages are pinned in `pyproject.toml`, including
`matplotlib==3.10.8`; rendering uses Agg and bundled DejaVu Sans.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.venv/Scripts/python.exe -m pytest
```

On Linux/macOS use `.venv/bin/python` and `cp .env.example .env`.
If Windows sandbox temp permissions block venv creation, set `TEMP` and `TMP` to
an existing writable folder inside the workspace before creating the venv.

## Generate the synthetic hard suite

```powershell
.venv/Scripts/python.exe scripts/generate_suite.py
```

This creates `data/suites/hard_v1.json` and the rendered pairs under `data/pairs/`.
The fixed seed generates 7 largest-category, 7 value-lookup and 6 threshold families.
All 20 are labelled synthetic. Their designed transformations change 13 reference
answers and preserve 7. No model output is included.

## Try the preflight

Copy `.env.example` to `.env` and set `GEMMA_API_KEY`; adjust `GEMMA_BASE_URL`,
`GEMMA_MODEL_ID`, `INFERENCE_TIMEOUT_SECONDS`, or `MAX_RETRIES` for the compatible
provider. `MAX_RETRIES` is limited to 0–2, so one preflight uses at most 3 attempts.
Then run the single real inference check:

```powershell
.venv/Scripts/python.exe scripts/preflight.py
```

It sends one chart image and fixed JSON-only prompt at temperature 0, then prints
the raw response, deterministic score, latency and attempts. It does not save the
response as demo evidence. The adapter includes no SDK retries and does not log keys.
Keep `.env` private; never commit provider credentials.

## Try Phase 1 pair generation

Generate a pair from the supplied synthetic input specification:

```powershell
.venv/Scripts/python.exe -m counterfact tests/fixtures/bar_pair.json
```

This writes two PNGs, chart JSON, a manifest, and `chartographer-family.jsonl`
under `data/pairs/<content-hash>/` and stores the manifest in SQLite. Repeating
the command returns the same pair. `model_evaluated` is explicitly `false`.

Start the API (one process):

```powershell
.venv/Scripts/python.exe -m uvicorn counterfact.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs, or:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
$body = Get-Content tests/fixtures/bar_pair.json -Raw
$pair = Invoke-RestMethod http://127.0.0.1:8000/api/pairs -Method Post -ContentType 'application/json' -Body $body
$pair.members
```

The response includes original/transformed PNG URLs and reference answers. The
example changes Lab A from 40 to 80, changing the unique maximum from Lab B to Lab A.

## Contracts

- Bar charts only, 2-20 categories, unique IDs and case-insensitive unique labels.
- Printable ASCII labels of at most 40 characters. Other labels are rejected
  explicitly rather than rendering missing glyphs.
- Nonnegative finite values up to 1,000,000. Unknown fields are rejected.
- `largest_category`: unique largest category; tied maxima rejected.
- `value_lookup`: explicit value labels in images; named category must exist.
- `above_threshold`: strict `>` comparison; equality is false.
- `change_value` and `reorder` mutations; missing IDs, nonpermutations, no-op
  mutations, and pairs with no net change are rejected.
- Reference logic computes whether an answer should stay the same or change.
  Changing a value does not always change the answer.

Both charts are validated before rendering or persistence. Pair IDs include the
specification and renderer version. PNG hashes identify actual image bytes.
Byte-level reproducibility is verified within the same pinned runtime, not
asserted across all operating systems and font-library versions.

## Structure

`counterfact/schemas.py` validates inputs; `reference.py` computes answers;
`render.py` renders with Agg; `storage.py` persists generated inputs; `api.py`
serves previews. `upstream.py` contains explicitly attributed upstream helpers.
`tests/` has unit and API integration tests. `frontend/` is reserved for Phase 5.
`data/`, `.env`, virtual environments and scratch files are gitignored.

`COUNTERFACT_DATA_DIR` selects storage. The comparison and Ollama variables in
`.env.example` are reserved for later phases. Never commit keys. The API logs request
IDs, methods and status, not payloads or secrets.

## Upstream versus ours

Source: [Chartographer](https://github.com/compling-wat/Chartographer), pinned commit
`046a622103733986db6e6b1b821cbc546b779192`, Apache-2.0.

| Upstream actually reused | CounterFact implementation |
|---|---|
| `family_member_row`, `write_jsonl` from the family exporter | Strict pair schema and two controlled mutations |
| Family row field layout | Three deterministic reference-answer functions |
| Apache-2.0 source attribution | Trusted Agg renderer, SQLite store, preview API and tests |

We do not bundle upstream model predictions, image datasets, local inference
dependencies, or arbitrary generated Python execution. See `NOTICE` and `LICENSE`.
Research context: [Chartographer](https://arxiv.org/abs/2605.27311).

## Phased build plan — stop after every phase

1. Foundation and valid chart pairs (complete).
1b. Deterministic scorer, hard synthetic suite, adapter and preflight (complete).
2. Three-observation inference, explicit attempt accounting and cache (not started).
3. Budgeted validity-preserving minimizer and real recording (not started).
4. Regression CLI/export and second-model comparison (not started).
5. React/Vite/TypeScript frontend and final walkthrough (not started).

Each phase requires a report and the user's next instruction. No model outcome
is claimed before real inference. No public deployment or GitHub publication has
occurred in Phase 1. The 3-hour project limit prioritizes the minimizer over polish;
cuts must be reported, not represented as completed features.

## Checks

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m pip check
```

Tests use temporary storage and a clearly named fake HTTP transport only in adapter
unit tests. No fake outputs are stored in suite data or presented as model results.
