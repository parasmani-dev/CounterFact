"""Audit regressions: acquisition provenance, persistence, strict JSON and queue atomicity."""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from counterfact.api import create_app
from counterfact.budget import Budget
from counterfact.config import Settings
from counterfact.evidence import stored_failure
from counterfact.reference import answer
from counterfact.runner import EvaluationProfile, run_suite
from counterfact.score import SCORER_VERSION, score_response
from counterfact.store import Store
from counterfact.suites import build_suite
from scripts.rescore import rescore_stored
from tests.test_phase2 import FakeAdapterForTests


@pytest.mark.parametrize(
    "raw",
    ['{"answer":NaN}', '{"answer":Infinity}', '{"answer":-Infinity}', '{"answer":1,"answer":2}'],
)
def test_v1_rejects_nonstandard_and_ambiguous_json(raw):
    assert SCORER_VERSION == "1"
    assert score_response(raw, "value_lookup", 2)["status"] == "invalid_output"


def test_replay_references_exact_fresh_ids_and_old_runs_stay_unchanged(tmp_path):
    suite = build_suite()
    spec = suite.families[0].spec
    responses = ['{"answer":"wrong"}'] * 6
    for chart in (spec.chart, spec.transformed()):
        responses += [json.dumps({"answer": answer(chart, spec.question)})] * 3
    adapter = FakeAdapterForTests(responses)
    store = Store(tmp_path)
    profile = EvaluationProfile("gemma", adapter, store)
    old = run_suite(suite, profile, False, Budget(6), limit=1)
    fresh = run_suite(suite, profile, True, Budget(6), limit=1)
    replay = run_suite(suite, profile, False, Budget(0), limit=1)
    assert replay["cases"][0]["pair_status"] == "both_pass"
    assert stored_failure(store, suite, run_id=replay["run_id"]) is None
    assert stored_failure(store, suite, run_id=old["run_id"]) is not None
    with store.connect() as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT budget_json FROM run WHERE id=?", (replay["run_id"],)
            ).fetchone()[0]
        )
    actual_ids = {
        o["id"]
        for c in fresh["cases"]
        for s in ("original", "transformed")
        for o in c[s]["observations"]
    }
    assert set(oid for ids in metadata["observation_ids"].values() for oid in ids) == actual_ids
    rescored = rescore_stored(store, suite)
    assert rescored["observation_counts"] == {"incorrect": 6, "correct": 6}
    assert adapter.calls == 12


def test_enqueue_transaction_rolls_back_job_if_run_insert_fails(tmp_path):
    store = Store(tmp_path)
    store.initialize()
    store.start_run("hard_v1", "gemma", False, {}, run_id="duplicate")
    with pytest.raises(sqlite3.IntegrityError):
        store.enqueue_job(
            "duplicate", "run", {"suite_id": "hard_v1", "model_profile": "gemma", "fresh": False}
        )
    assert store.get_job("duplicate") is None


def test_restart_poll_retains_finished_attempt_counts(tmp_path):
    path = tmp_path / "suites/hard_v1.json"
    path.parent.mkdir()
    path.write_text(build_suite().model_dump_json())
    store = Store(tmp_path)
    store.initialize()
    store.create_job(
        "r", "run", {"suite_id": "hard_v1", "model_profile": "gemma", "fresh": False, "limit": 1}
    )
    store.start_run("hard_v1", "gemma", False, {"used": 7, "cap": 120}, run_id="r")
    store.finish_run("r", "completed", {"used": 7, "cap": 120, "cached_count": 3, "live_count": 3})
    store.update_job("r", "completed", {"cases": [], "attempts_used": 7})
    with TestClient(create_app(Settings(tmp_path), adapter_factory=FakeAdapterForTests)) as client:
        result = client.get("/api/runs/r").json()
        assert result["budget"]["used"] == 7
        assert result["cached_count"] == 3
        assert result["live_count"] == 3


def test_models_health_do_not_parse_unrelated_invalid_timeout(tmp_path, monkeypatch):
    path = tmp_path / "suites/hard_v1.json"
    path.parent.mkdir()
    path.write_text(build_suite().model_dump_json())
    monkeypatch.setenv("INFERENCE_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setenv("GEMMA_API_KEY", "test-only-secret")
    with TestClient(create_app(Settings(tmp_path))) as client:
        assert client.get("/api/health").status_code == 200
        response = client.get("/api/models")
        assert response.status_code == 200
        assert response.json()[0]["key_present"] is True
        assert "test-only-secret" not in response.text
