# CounterFact

**Change the chart. Check the answer. Shrink the failure. Keep the test.**

CounterFact is an open-source evaluation harness for vision-language models (VLMs) that read charts. It generates controlled bar-chart pairs, sends them to a real open-weight model (Gemma 4), checks each answer three times against computed ground truth, and, when it finds a repeatable failure, **reduces the failing chart to a smaller, still-valid case** and saves it as a portable regression test.

> **Status of this README.** Sections marked `[FILL]` must be completed from your own real runs before publishing. Do not paste numbers you did not observe. Features not completed are listed under [Cut and incomplete features](#cut-and-incomplete-features).

---

## Contents

1. [Why this exists](#why-this-exists)
2. [What it does](#what-it-does)
3. [How it works](#how-it-works)
4. [Models and dependencies](#models-and-dependencies)
5. [Quick start](#quick-start)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Observed results](#observed-results)
9. [Design decisions and honesty rules](#design-decisions-and-honesty-rules)
10. [Limits and non-claims](#limits-and-non-claims)
11. [Cut and incomplete features](#cut-and-incomplete-features)
12. [Testing](#testing)
13. [Repository layout](#repository-layout)
14. [Upstream attribution and license](#upstream-attribution-and-license)
15. [Agent skill](#agent-skill)

---

## Why this exists

Developers build apps on open-weight VLMs (dashboards, document readers, report analysers). When a model upgrade or prompt change breaks chart reading, the usual evidence is a large, messy failing example. It is hard to tell *what* about the image mattered, and hard to turn it into a test that keeps running.

CounterFact addresses that narrow problem for bar charts:

- It **changes** a chart in a controlled way (edit a value, reorder categories) and computes the correct answer for both versions with plain Python, not another model.
- It **checks** the real model's answer repeatedly, so one lucky or unlucky response is not treated as a verdict.
- It **shrinks** a confirmed failure by deleting non-essential categories while keeping the question valid and the failure intact, under a strict call budget.
- It **keeps** the result as a self-contained regression case that can be rerun from the command line and in CI.

Prior work it builds on: counterfactual evaluation, delta debugging, and chart-generation benchmarks. The contribution here is the combination: validity-preserving, budgeted visual failure reduction with repeated model checks and portable artifacts. See [attribution](#upstream-attribution-and-license).

## What it does

| Capability | Summary |
|---|---|
| Chart generation | Deterministic bar-chart PNGs from a JSON spec (2–20 categories). Pinned renderer, font, size, and DPI. |
| Reference answers | Versioned Python functions for three question types. No LLM is used to produce or interpret ground truth. |
| Real inference | Calls a hosted Gemma 4 endpoint through an OpenAI-compatible chat-completions API. Image sent as a base64 data URL. |
| Repeat checks | Three separately acquired observations per image. A check fails when at least 2 of 3 are wrong. |
| Cache | Observation groups stored in SQLite by a secret-free key. Reruns reuse them and make zero requests. |
| Budget | Every outbound attempt, including retries and failures, counts against a hard cap (default 60 for minimization). |
| Minimizer | Delta-debugging-style category deletion applied consistently to both sides of a pair, with fresh final confirmation. |
| Backend API | FastAPI, one in-process job queue, polling progress. |
| UI | One React page for running a suite, inspecting a failure, and minimizing it. |
| Regression cases | Saved JSON that re-renders the exact failing charts and records provenance. |

### Question types

| Type | Answer | Rule |
|---|---|---|
| `largest_category` | category label | Unique maximum required. Ties are rejected. |
| `value_lookup` | number | Named category. Value labels are drawn on the chart so scoring does not depend on estimating pixels. |
| `above_threshold` | yes / no | Strictly greater than the threshold. Equality means no. |

### Transformations

- **Change value:** edit one category's value.
- **Reorder:** permute categories, keeping each label tied to its value.

A transformation is **not assumed** to change the answer. Both answers are computed, and the expected relationship (changed or unchanged) is recorded.

## How it works

```
 suite spec (JSON)
       │
       ▼
 validate ──► render original + transformed PNG ──► reference answers (Python)
                                │
                                ▼
                 adapter: OpenAI-compatible request (temperature 0)
                                │  3 separate observations per image
                                ▼
                 scorer: strict deterministic parse ──► correct | incorrect | invalid_output
                                │
                                ▼
              check verdict (2-of-3) ──► SQLite cache (group key, slot, timestamp)
                                │
              confirmed failure │
                                ▼
          minimizer: candidate generation ─► validity filter (no model call)
                     ─► evaluate with same predicate ─► accept only smaller failing candidates
                     ─► fresh final confirmation (cache bypassed)
                                │
                                ▼
              regression case JSON  ·  API  ·  UI  ·  CLI
```

### Check semantics

- A **check** is three distinct inference observations for the same image, question, and configuration.
- A check **fails** if at least 2 of 3 observations are parsed and incorrect. It **passes** if at least 2 of 3 are correct. All three slots must be obtained before a completed verdict (no early stop at two votes).
- **Provider errors** (authentication, rate limit, timeout, server, network) are not task failures. An incomplete group is `inconclusive`.
- **Invalid output** (response that does not match the strict answer format) is stored and reported separately. It is never counted as an incorrect answer.
- For a paired robustness test, the original must pass and the transformed version must fail (`correct_to_incorrect`).
- The 2-of-3 rule is an operational reproducibility rule. It is not a statistical confidence guarantee.

### Cache

The group key is a SHA-256 over the exact image bytes, canonical question, prompt version, effective generation configuration, provider, model, and configuration revision. API keys are never part of it. Three observations are stored per group with slot, timestamp, and request provenance. One response is never duplicated into three votes. A **fresh** run bypasses the cache and records a new acquisition. Replaying cached results is a replay of earlier evidence, not new evidence that a provider still behaves the same.

### Minimizer

1. Validate the case and confirm the starting failure (three observations per needed image).
2. Generate candidates by group deletion over non-essential categories (halves, then smaller groups), then single deletions.
3. Apply each deletion to **both** charts of a pair, then regenerate images, question bindings, and reference answers.
4. **Reject invalid candidates before any model call:** fewer than two categories, ties, lost target, identity transformation, duplicate labels, validation failure.
5. Evaluate remaining candidates with the same predicate and the same 2-of-3 rule. Accept only strictly smaller candidates that keep the failure. Size is `(category_count, transformation_count)`.
6. Reserve attempts for a **fresh** final confirmation (6 for a pair, 3 for a single chart). If confirmation fails or is incomplete, the result is **not** labelled confirmed.
7. Stop with an explicit reason: `search_completed`, `budget_exhausted`, `time_limit`, `cancelled`, `no_reproducible_failure`, `confirmation_failed`, `provider_unavailable`, or `interrupted`.

Protected categories are never removed: the target category (lookup and threshold), the unique winner in the original and transformed charts (largest-category), and any category touched by the transformation. The reducer never edits the prompt to manufacture a failure.

**Budget reality:** with uncached paired checks, 60 attempts allow roughly ten six-attempt groups, including baseline and final confirmation. That is about eight search candidates with no retries, not 60 reduction steps.

The text "smallest confirmed case found" is used **only** when fresh confirmation passed. Otherwise the output says "smallest observed candidate (unconfirmed)". No global-minimality claim is made.

### Answer format and scoring

The model receives only the image and the question, with this fixed, versioned instruction:

```
Answer the question about the chart. Reply with JSON only: {"answer": <value>}.
```

Source data and expected answers never enter the model's context. Scoring is deterministic Python:

- Strict parse of `{"answer": ...}` (one outer code fence permitted, whitespace trimmed).
- Category labels: exact match after trim and case-fold.
- Numbers: small relative tolerance.
- Yes/no: normalized to a boolean.
- Anything else is `invalid_output`. No repair, no LLM judge.

> `[FILL]` If you changed the normalizer during the project (for example to remove a leading reasoning block), document the exact final rule and `SCORER_VERSION` here. Example wording: *"Scorer v2 removes exactly one fully closed leading `<thought>…</thought>` block, then applies the strict rule above. Nothing else is repaired."* Delete this note if unchanged.

## Models and dependencies

### Models

| Role | Model | Access |
|---|---|---|
| Primary model under test | `gemma-4-31b-it` (Gemma 4, open-weight, image input) | Google Gemini API OpenAI-compatible endpoint, `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Alternative route | `google/gemma-4-31b-it:free` | OpenRouter, `https://openrouter.ai/api/v1` |
| Comparison model | `[FILL]` (open-weight, image-capable) | `[FILL]` or "not run in this build" |

Gemma 4 is the model being evaluated. Provider documentation and listings show a candidate route, not guaranteed access: free-tier limits, queue capacity, and routing are provider-controlled and can change. Provider and model identity are recorded with every observation. Observations from different models or providers are never combined in one vote or one reduction.

### Key dependencies

| Package | Use |
|---|---|
| Python 3.11+ | Engine and backend |
| FastAPI + Uvicorn | HTTP API |
| httpx | Inference requests (own retry loop, no hidden SDK retries) |
| matplotlib 3.10.x (Agg) | Deterministic rendering with bundled DejaVu Sans |
| SQLite (stdlib) | Structured records and cache |
| pytest, ruff | Tests and lint |
| React + Vite + TypeScript | Frontend |

Exact versions are pinned in `pyproject.toml` and `frontend/package-lock.json`.

## Quick start

### Prerequisites

- Python 3.11 or newer (developed and tested on Python 3.14.0, Windows; `[FILL]` Linux/macOS status)
- Node.js 18+ (for the UI)
- A Gemma API key from your chosen provider, supplied through the environment only

### Install

```powershell
git clone https://github.com/parasmani-dev/counterfact.git
cd counterfact
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
copy .env.example .env
# edit .env and set GEMMA_API_KEY
```

On macOS or Linux use `source .venv/bin/activate` and `cp .env.example .env`.

### Verify the engine without any API key

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m counterfact tests/fixtures/bar_pair.json
```

Expected computed ground truth for the fixture: original largest category `Lab B`, transformed largest category `Lab A`. These are reference answers, not model output.

### Run the backend

```powershell
.venv/Scripts/python.exe -m uvicorn counterfact.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API reference.

### Run the UI

```powershell
cd frontend
npm install
npm run dev
```

Open the printed URL. The dev server proxies `/api` to the backend.

### Real-inference smoke test

```powershell
.venv/Scripts/python.exe scripts/preflight.py
```

This sends one real image request (at most three attempts including retries) and prints the raw response, parsed score, latency, and attempts. If the key is missing it prints instructions and exits with code 2.

## Configuration

All configuration is by environment variables. Keys are server-side only and are excluded from logs, cache keys, recordings, and exports. The application uses only administrator-configured endpoints; the browser cannot supply arbitrary URLs.

```dotenv
GEMMA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMMA_MODEL_ID=gemma-4-31b-it
GEMMA_API_KEY=
COMPARISON_BASE_URL=
COMPARISON_MODEL_ID=
COMPARISON_API_KEY=
INFERENCE_TIMEOUT_SECONDS=120
MINIMIZATION_TIMEOUT_SECONDS=600
MAX_MINIMIZATION_CALLS=60
MAX_RETRIES=2
MAX_CONCURRENT_JOBS=1
DATABASE_URL=sqlite:///./data/counterfact.db
ARTIFACT_DIR=./data/artifacts
```

- `MAX_MINIMIZATION_CALLS` may be lowered but never raised above 60.
- Requests use temperature 0 and a fixed versioned prompt. Other generation parameters, token limits, and any reasoning configuration are recorded with each observation. `[FILL]` list the exact `max_tokens` and reasoning/thinking setting you used.
- Temperature 0 does not guarantee identical outputs.
- Retries: at most two, for 429, 5xx, network errors, and timeouts, with exponential backoff and jitter. No retry for authentication or malformed-request errors.
- No automatic paid-model fallback and no silent provider or model switch.

## Usage

### Run a suite (UI)

1. Choose a suite and model profile. If the profile has no key, Start is disabled with the reason shown.
2. Leave **fresh acquisition** off to reuse cached observations (zero requests for complete groups). Turn it on to spend real quota and record new evidence.
3. Start. The page polls every two seconds and stops at a terminal state.
4. Select a case to see both images, the question, expected answers, all three observations per side, and the actual model and provider.
5. For an eligible failure, press **Minimize**. Watch calls used out of 60, cache hits, validity rejections, and the stop reason. Compare before and after images.
6. **Save regression** to keep the case.

### API summary

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Backend and storage readiness. No secrets. |
| `GET /api/models` | Configured profiles and key presence (yes/no only). |
| `GET /api/suites` | Available suites. |
| `POST /api/runs` | Start a run. Returns 202 and a run ID. |
| `GET /api/runs/{id}` | Poll progress and results. |
| `POST /api/runs/{id}/cancel` | Idempotent cancellation. |
| `POST /api/minimizations` | Start reduction for an eligible case. Returns 202. |
| `GET /api/minimizations/{id}` | Search steps, best candidate, confirmation. |
| `POST /api/minimizations/{id}/cancel` | Idempotent cancellation. |
| `POST /api/regressions` | Save a case with provenance. |
| `GET /api/artifacts/{hash}` | Serve a known artifact by content hash. Never a filesystem path. |

Error body: `{"error": {"code", "message", "retryable", "details"}, "request_id"}`. Status codes: 422 validation, 404 unknown ID, 409 incompatible state, 429 queue full.

Example, using a cached run:

```powershell
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/runs -H "Content-Type: application/json" -d "{\"suite_id\":\"<id>\",\"model_profile\":\"gemma\",\"fresh\":false}"
curl http://127.0.0.1:8000/api/runs/<run_id>
```

### Command line

`[FILL]` Keep this section only if Phase 6 CLI was completed and tested. Otherwise move it to *Cut and incomplete features*.

```powershell
counterfact run <suite-or-regression.json> --profile gemma [--fresh] --json out.json --junit out.xml
counterfact minimize <case.json> --profile gemma --max-calls 60 --out reduced.json
```

| Exit code | Meaning |
|---|---|
| 0 | All required checks pass. |
| 1 | Completed checks show a failure or regression. |
| 2 | Invalid input, interruption, or infrastructure failure prevents a complete conclusion. |

JSON output keeps all individual outcomes even when exit code 2 takes precedence.

## Observed results

> Only real results from your own runs belong here. The reference numbers below are placeholders. If no failure was observed, say so plainly. That is a valid result.

**Run identity** `[FILL]`

| Field | Value |
|---|---|
| Date | `[FILL]` |
| Provider and endpoint | `[FILL]` |
| Model ID | `gemma-4-31b-it` |
| Suite | `hard_v1` (`[FILL]` number of families, question-type split) |
| Fresh or cached | `[FILL]` |
| Total real outbound attempts | `[FILL]` |
| Provider errors (timeout / server / rate-limit) | `[FILL]` |
| Quota or coverage limits hit | `[FILL]` |

**Suite results** `[FILL]`

| Case | Question type | Original (correct/3) | Transformed (correct/3) | Pair status |
|---|---|---|---|---|
| `[FILL]` | | | | |

Counts to report with denominators: correct, incorrect, `invalid_output`, inconclusive. Do not report a percentage without its denominator, and do not draw an overall model-quality conclusion from a small synthetic suite.

**Minimization** `[FILL]`

| Field | Value |
|---|---|
| Starting case | `[FILL]` (categories, question type, predicate) |
| Final candidate | `[FILL]` (categories) |
| Attempts used / cap | `[FILL]` / 60 |
| Cache hits | `[FILL]` |
| Validity rejections | `[FILL]` |
| Fresh confirmation | `[FILL]` passed / failed / incomplete |
| Terminal reason | `[FILL]` |

If there was no reproducible failure, replace this table with: *"No failure observed in this run. The minimizer was verified only on deterministic fake-adapter tests, which are not model evidence."*

Before and after images: `[FILL]` add paths such as `docs/images/before.png` and `docs/images/after.png`.

## Design decisions and honesty rules

- **Real inference only for evidence.** Fake adapters exist only inside unit tests and are named as such. Their outputs never enter demo data, recordings, or this README's results.
- **No invented results.** If models pass everything, the report says no failure was observed.
- **Provider errors are not model failures.** They are tracked separately and never counted as incorrect answers.
- **Invalid output is not incorrect output.** Format failures are reported on their own.
- **No prompt tuning to induce failures.** The prompt and scorer are fixed and versioned.
- **Strict budget accounting.** Budget is reserved before dispatch. Retries count. Cancellation stops new dispatch, though requests already sent may still consume provider quota.
- **No arbitrary code execution.** Charts come from trusted templates. User-supplied plotting code is never run.
- **Model text is rendered as escaped text** in the UI, never as HTML.
- **Recorded replay is labelled.** Any view backed only by stored observations shows "Recorded run — no live inference".
- **Every synthetic input is labelled synthetic** and carries `model_evaluated: false` until a real model has been run on it.

## Limits and non-claims

- Only **bar charts** are supported. Unsupported chart types fail validation.
- Labels are printable ASCII up to 40 characters.
- The suite is **synthetic** and small. Results do not measure general chart-reading ability and do not support an overall ranking of models.
- Temperature 0 does not guarantee reproducibility. Hosted providers may route to different backends. Provider metadata is recorded where available, and the routing variation is uncontrolled.
- 2-of-3 voting is not a confidence interval.
- A reduced case is the **smallest observed** candidate under a limited search budget. It is not claimed to be globally minimal.
- A reduced failure shows that a smaller valid chart still produces the wrong answer. It does not diagnose the model's internal reasoning and does not prove or disprove understanding.
- CounterFact does not guarantee deployment safety.
- Cached replay shows earlier evidence. It does not show a provider still behaves identically today.
- Rendering byte-equality is tested within one runtime. Identical PNG bytes across operating systems and matplotlib builds are not guaranteed.
- Free-tier access can fail through rate limits, timeouts, or capacity. The tool reports these honestly and keeps partial results.
- No claim of being first at any of these techniques is made.

## Cut and incomplete features

This section is part of the honest record. Edit it to match what you actually shipped. Do not leave a cut feature marked complete.

| Feature | Status |
|---|---|
| Local Ollama inference profile | Cut. Not implemented in this build. |
| User upload of arbitrary suites | Cut. Seeded and generated suites only. |
| Public read-only hosted viewer | Cut. Local execution plus public source. |
| Background-colour transformation | Cut. |
| Separate three-screen UI | Reduced to one page with run and investigation sections. |
| Model comparison view | `[FILL]` Cut / partial / done. If cut, the UI states "not implemented in this build". |
| CLI with JSON and JUnit output | `[FILL]` Done / partial / cut. |
| Portable recording export and offline replay | `[FILL]` Done / partial / cut. |
| Second open-weight VLM run | `[FILL]` Done / blocked by availability / cut. |
| Recorded 90-second demo video | `[FILL]` |

## Testing

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m pip check
cd frontend
npm run build
npx tsc --noEmit
```

Reported at last check: `[FILL]` test count and result (Phase 2 reported 99 passing; update with the final number).

Engine tests cover: all three reference functions; changed versus unchanged answer relationships; tie and target rejection; paired category preservation; deterministic rendering; scorer cases; cache slot independence; cache replay with zero calls; partial-group completion; fresh bypass; exact budget enforcement including retries; provider error handling; invalid-output separation; secret-free cache keys; minimizer protection, validity filtering, confirmation, cancellation, and time limit; API status codes, idempotent cancel, interrupted-job recovery, and artifact path-traversal rejection.

Tests use `FakeAdapterForTests` and `FakeTransportForTests`. These are isolated-component fixtures only. They are not model evidence.

Known warning: one Starlette/AnyIO deprecation warning appears in the test run. It does not affect results.

## Repository layout

```
counterfact/          engine package (spec, reference, render, transform, score,
                      adapter, budget, store, check, runner, minimize, api)
scripts/              preflight.py, pilot.py, minimize_real.py
tests/                unit and API tests, fixtures
frontend/             React + Vite + TypeScript UI
skills/counterfact/   Agent Skill (SKILL.md)
data/                 local database, artifacts, suites, regressions (gitignored)
docs/                 demo script and images
.env.example          environment template (no secrets)
LICENSE               Apache-2.0
NOTICE                upstream attribution
```

`[FILL]` Adjust to match the real tree.

## Upstream attribution and license

CounterFact is released under the **Apache License 2.0**. See `LICENSE` and `NOTICE`.

**Upstream project:** [Chartographer](https://github.com/compling-wat/Chartographer) (Apache-2.0), pinned at commit `046a622103733986db6e6b1b821cbc546b779192`. Paper: <https://arxiv.org/abs/2605.27311>.

| Upstream (Chartographer) | CounterFact |
|---|---|
| Dependency-free `family_member_row` and `write_jsonl` export helpers, adapted for the family JSONL export | Independent renderer, validation, reference-answer logic, storage, API, inference adapter, cache, budget accounting, repeat-check policy, validity-preserving reducer, UI, and CLI |

Chartographer's rendering, validation, reference logic, storage, and API were **not** reused. Modified upstream code is marked as modified. Dataset permissions are separate from code licensing, so the demo uses CounterFact's own generated, synthetic charts rather than upstream data.

Related reading that informed the approach: [MetaRA](https://arxiv.org/abs/2605.19307) and the text metamorphic shrinking reference at <https://github.com/parag-labs/metamorph>. No code from them is used `[FILL]` confirm.

Gemma is used under Google's published terms for the Gemma models and the Gemini API. Check those terms for your own use.

## Agent skill

`skills/counterfact/SKILL.md` follows the Agent Skills open standard (`name` and `description` frontmatter, plus instructions). It tells an agent how to run `counterfact run` in CI, read the exit codes (0 pass, 1 regression, 2 infrastructure or input error), and interpret the JSON and JUnit outputs. `[FILL]` Keep only if the file exists and was checked.

## Hackathon context

Built for the MLH "Build the AI System Behind the AI" track as an open-source evaluation system for open-weight vision-language models.

- **Gemma 4:** the model under test, called with real image input through an OpenAI-compatible endpoint.
- **Open-source AI:** open-weight models, an Apache-2.0 harness, portable regression artifacts, and CI-friendly outputs.
- **Original implementation:** the repeat-check policy, budget accounting, cache provenance, and validity-preserving reducer are original to this project.

## Contributing

Issues and pull requests are welcome. Please keep the honesty rules above: no fabricated model results, no relaxed scoring to hide failures, and clearly labelled test fixtures.
