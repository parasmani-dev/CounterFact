"""All synthetic adapter responses in this file are isolated unit-test fixtures."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget, BudgetExceeded
from counterfact.check import cache_key, run_check
from counterfact.runner import EvaluationProfile, pair_status, run_suite
from counterfact.schemas import PairSpec
from counterfact.storage import PairStore
from counterfact.store import Store
from counterfact.suites import build_suite


class FakeAdapterForTests:
    def __init__(self, responses=None):
        self.responses = list(responses or ['{"answer":"A"}'] * 100)
        self.calls = 0
        self.provider = "test.example"
        self.config = SimpleNamespace(model_id="test-model", config_revision="1", api_key="secret")

    def request_params(self):
        return {
            "temperature": 0,
            "max_tokens": 1024,
            "model": self.config.model_id,
            "config_revision": self.config.config_revision,
            "endpoint": "https://test.example/v1/chat/completions",
            "api_key": self.config.api_key,
        }

    def complete(self, image, question, *, budget):
        try:
            budget.reserve(1)
        except BudgetExceeded:
            return {
                "ok": False,
                "error_type": "budget",
                "attempts_used": 0,
                "budget_exhausted": True,
            }
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, dict):
            return {"ok": False, "attempts_used": 1, "budget_exhausted": False, **response}
        return {
            "ok": True,
            "raw_text": response,
            "error_type": "none",
            "latency_ms": 1.0,
            "attempts_used": 1,
            "budget_exhausted": False,
        }


@pytest.fixture
def store(tmp_path):
    instance = Store(tmp_path)
    instance.initialize()
    return instance


def check(store, adapter, budget=None, fresh=False):
    return run_check(
        b"chart-pixels",
        "which label?",
        "A",
        "largest_category",
        adapter,
        store,
        budget if budget is not None else Budget(50),
        fresh,
    )


def count_observations(store):
    with store.connect() as connection:
        return connection.execute("SELECT COUNT(*) FROM observation").fetchone()[0]


def test_three_distinct_slots_no_early_stop_and_cache_replay(store):
    adapter = FakeAdapterForTests(['{"answer":"A"}', '{"answer":"a"}', '{"answer":"B"}'])
    first = check(store, adapter)
    assert first["verdict"] == "passes"
    assert adapter.calls == first["attempts_used"] == 3
    assert {o["slot"] for o in first["observations"]} == {0, 1, 2}
    assert len({o["id"] for o in first["observations"]}) == 3
    assert len({o["raw_text"] for o in first["observations"]}) == 3
    assert count_observations(store) == 3
    replay = check(store, adapter, Budget(0))
    assert adapter.calls == 3
    assert replay["attempts_used"] == replay["live_count"] == replay["live_requests"] == 0
    assert replay["cached_count"] == 3
    assert [o["acquired_at"] for o in replay["observations"]] == [
        o["acquired_at"] for o in first["observations"]
    ]


def test_partial_group_acquires_only_missing_slots(store):
    adapter = FakeAdapterForTests()
    partial = check(store, adapter, Budget(1))
    assert partial["verdict"] == "inconclusive"
    assert count_observations(store) == 1
    completed = check(store, adapter)
    assert completed["verdict"] == "passes"
    assert completed["cached_count"] == 1
    assert completed["live_count"] == completed["attempts_used"] == 2
    assert adapter.calls == count_observations(store) == 3


def test_fresh_bypasses_cache_preserves_acquisitions_and_updates_replay(store):
    adapter = FakeAdapterForTests()
    first = check(store, adapter)
    fresh = check(store, adapter, fresh=True)
    assert fresh["cached_count"] == 0
    assert fresh["live_count"] == 3
    assert first["group_key"] != fresh["group_key"]
    assert count_observations(store) == 6
    with store.connect() as connection:
        assert (
            connection.execute(
                "SELECT acquired_at FROM observation WHERE id=?", (first["observations"][0]["id"],)
            ).fetchone()[0]
            == first["observations"][0]["acquired_at"]
        )
    replay = check(store, adapter, Budget(0))
    assert [o["id"] for o in replay["observations"]] == [o["id"] for o in fresh["observations"]]


@pytest.mark.parametrize(
    "error", ["auth", "rate_limit", "timeout", "server", "network", "bad_request"]
)
def test_provider_error_inconclusive_and_not_an_observation(store, error):
    result = check(store, FakeAdapterForTests([{"error_type": error}]))
    assert result["verdict"] == "inconclusive"
    assert result["incorrect_count"] == 0
    assert count_observations(store) == 0
    assert result["errors"] == {error: 1}


@pytest.mark.parametrize(
    "raws,verdict,incorrect,invalid",
    [
        (["bad", "bad", '{"answer":"B"}'], "mixed", 1, 2),
        (['{"answer":"B"}', '{"answer":"B"}', "bad"], "fails", 2, 1),
        (['{"answer":"A"}', "bad", '{"answer":"A"}'], "passes", 0, 1),
        (['{"answer":"B"}', '{"answer":"B"}', '{"answer":"B"}'], "fails", 3, 0),
    ],
)
def test_invalid_output_is_separate_and_all_three_slots_required(
    store, raws, verdict, incorrect, invalid
):
    adapter = FakeAdapterForTests(raws)
    result = check(store, adapter)
    assert result["verdict"] == verdict
    assert result["incorrect_count"] == incorrect
    assert result["invalid_output_count"] == invalid
    assert adapter.calls == count_observations(store) == 3


def test_key_is_secret_free_but_tracks_configuration(store):
    adapter = FakeAdapterForTests()
    key = cache_key(b"chart-pixels", "which label?", adapter)
    adapter.config.api_key = "rotated-secret"
    assert cache_key(b"chart-pixels", "which label?", adapter) == key
    result = check(store, adapter)
    assert "rotated-secret" not in json.dumps(result)
    assert all("api_key" not in o["request_params_json"] for o in result["observations"])
    adapter.config.config_revision = "2"
    assert cache_key(b"chart-pixels", "which label?", adapter) != key
    assert cache_key(b"other-pixels", "which label?", adapter) != key


def test_budget_is_atomic_under_concurrent_reservations():
    budget = Budget(11)

    def reserve(_):
        try:
            budget.reserve(1)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        successful = sum(pool.map(reserve, range(100)))
    assert successful == budget.used == 11
    with pytest.raises(BudgetExceeded):
        budget.reserve(1)
    assert budget.report() == {"used": 11, "cap": 11}


@pytest.mark.parametrize("cap,attempts", [(0, 0), (1, 1), (2, 2), (3, 3)])
def test_adapter_reserves_every_retry_before_dispatch(cap, attempts):
    calls = []
    budget = Budget(cap)

    def handler(request):
        calls.append(request)
        assert budget.used == len(calls)  # Reservation already happened.
        return httpx.Response(500)

    class FakeTransportForTests(httpx.MockTransport):
        pass

    adapter = OpenAICompatibleAdapter(
        InferenceConfig("https://test.example/v1", "test-model", "secret", 1, 2),
        transport=FakeTransportForTests(handler),
        sleep_fn=lambda _: None,
    )
    result = adapter.complete(b"pixels", "question", budget=budget)
    assert result["attempts_used"] == budget.used == len(calls) == attempts
    assert result["budget_exhausted"] is (cap < 3)


def test_migration_preserves_pairs_and_is_idempotent(tmp_path, pair_payload):
    old = PairStore(tmp_path)
    old.initialize()
    original = old.create(PairSpec.model_validate(pair_payload))
    new = Store(tmp_path)
    new.initialize()
    new.initialize()
    assert new.get(original["id"]) == original
    old.initialize()  # Old pair API continues to work and cannot downgrade v2.
    with new.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO case_result VALUES ('c','missing','f','original','g','passes',0,3)"
            )


def test_invalid_schema_version_rejected(store):
    with store.connect() as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="schema version"):
        store.initialize()


def test_failed_migration_rolls_back_new_tables_and_version(tmp_path, pair_payload):
    old = PairStore(tmp_path)
    old.initialize()
    manifest = old.create(PairSpec.model_validate(pair_payload))
    with old.connect() as connection:
        connection.execute("CREATE TABLE run (unrelated TEXT)")
    with pytest.raises(sqlite3.OperationalError):
        Store(tmp_path).initialize()
    with old.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='observation'").fetchone()
            is None
        )
    assert old.get(manifest["id"]) == manifest


def test_observation_write_constraint_rolls_back(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.save_observation(
            {
                "group_key": "g",
                "slot": 3,
                "raw_text": "bad",
                "status": "invalid_output",
                "error_type": "none",
                "provider": "test",
                "model": "test",
                "latency_ms": 1,
                "request_params_json": "{}",
            }
        )
    assert count_observations(store) == 0


def test_runner_saves_results_replays_cache_and_reports_failures(store):
    suite = build_suite()
    original = suite.families[0].spec
    from counterfact.reference import answer

    expected = answer(original.chart, original.question)
    responses = [json.dumps({"answer": expected})] * 3 + ['{"answer":"wrong"}'] * 3
    adapter = FakeAdapterForTests(responses)
    profile = EvaluationProfile("test", adapter, store)
    first = run_suite(suite, profile, False, Budget(6), limit=1)
    assert first["cases"][0]["pair_status"] == "correct_to_incorrect"
    assert first["failures"] == [
        {"case_id": suite.families[0].family_id, "side": "transformed", "incorrect_count": 3}
    ]
    replay = run_suite(suite, profile, False, Budget(0), limit=1)
    assert replay["live_requests"] == replay["attempts_used"] == 0
    assert replay["cached_count"] == 6
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM case_result").fetchone()[0] == 4
        assert (
            connection.execute("SELECT COUNT(*) FROM run WHERE status='completed'").fetchone()[0]
            == 2
        )


def test_runner_stops_on_persistent_rate_limit_and_saves_partial_result(store):
    adapter = FakeAdapterForTests([{"error_type": "rate_limit"}])
    result = run_suite(
        build_suite(), EvaluationProfile("test", adapter, store), False, Budget(80), limit=6
    )
    assert result["status"] == "rate_limited"
    assert adapter.calls == result["attempts_used"] == 1
    assert len(result["cases"]) == 1
    assert result["cases"][0]["transformed"] is None
    with store.connect() as connection:
        assert connection.execute("SELECT status FROM run").fetchone()[0] == "rate_limited"
        assert connection.execute("SELECT COUNT(*) FROM case_result").fetchone()[0] == 1


def test_six_family_pilot_replay_needs_no_live_requests(store):
    from counterfact.reference import answer

    suite = build_suite()
    responses = []
    for family in suite.families[:6]:
        for chart in (family.spec.chart, family.spec.transformed()):
            responses.extend([json.dumps({"answer": answer(chart, family.spec.question)})] * 3)
    adapter = FakeAdapterForTests(responses)
    profile = EvaluationProfile("test", adapter, store)
    first = run_suite(suite, profile, False, Budget(80), limit=6)
    second = run_suite(suite, profile, False, Budget(0), limit=6)
    assert first["attempts_used"] == first["live_count"] == first["live_requests"] == 36
    assert second["status"] == "completed"
    assert second["cached_count"] == 36
    assert second["attempts_used"] == second["live_count"] == second["live_requests"] == 0
    assert adapter.calls == 36


def test_successful_retry_is_charged_and_reported_without_poisoning_vote(store):
    from tests.test_adapter import FakeTransportForTests

    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"A"}'}}]})

    budget = Budget(4)
    adapter = OpenAICompatibleAdapter(
        InferenceConfig("https://test.example/v1", "test", "secret", 1, 2),
        transport=FakeTransportForTests(handler),
        sleep_fn=lambda _: None,
    )
    result = check(store, adapter, budget)
    assert result["verdict"] == "passes"
    assert result["terminal_error"] == "none"
    assert result["errors"] == {"rate_limit": 1}
    assert result["attempts_used"] == len(calls) == budget.used == 4


def test_pilot_missing_key_exits_before_real_call(monkeypatch, capsys):
    from scripts.pilot import main

    monkeypatch.setattr(
        InferenceConfig, "from_env", lambda: InferenceConfig("https://test.example/v1", "test", "")
    )
    monkeypatch.setattr("sys.argv", ["pilot.py", "--limit", "6"])
    assert main() == 2
    assert "Gemma API key missing" in capsys.readouterr().out


@pytest.mark.parametrize(
    "original,transformed,expected",
    [
        ("passes", "fails", "correct_to_incorrect"),
        ("fails", "fails", "both_fail"),
        ("passes", "passes", "both_pass"),
        ("inconclusive", "passes", "inconclusive"),
        ("mixed", "passes", "mixed"),
        ("fails", "passes", "mixed"),
    ],
)
def test_pair_classification(original, transformed, expected):
    assert pair_status(original, transformed) == expected
