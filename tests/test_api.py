"""FastAPI/queue integration uses only FakeAdapterForTests, never live inference."""

import json
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from counterfact.api import create_app
from counterfact.budget import Budget
from counterfact.config import Settings
from counterfact.reference import answer
from counterfact.runner import EvaluationProfile, run_suite
from counterfact.store import Store
from counterfact.suites import build_suite
from tests.test_phase2 import FakeAdapterForTests


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "suites/hard_v1.json"
    path.parent.mkdir()
    path.write_text(build_suite().model_dump_json())
    return Settings(tmp_path)


def poll(client, path):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(0.02)
    raise AssertionError("Job did not finish")


def run_body(**kwargs):
    return {"suite_id": "hard_v1", "model_profile": "gemma", "fresh": False, "limit": 1, **kwargs}


def test_health_models_and_suites(settings):
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["phase"] == 4
        assert health.headers["x-request-id"]
        models = client.get("/api/models")
        assert models.json() == [{"name": "gemma", "key_present": True, "last_status": None}]
        assert "secret" not in models.text
        assert client.get("/api/suites").json()[0]["family_count"] == 20
        assert client.post("/api/pairs", json={}).status_code == 404


def test_404_409_422_and_secret_safe_errors(settings):
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:
        for path in ("/api/runs/missing", "/api/minimizations/missing", "/missing"):
            response = client.get(path)
            assert response.status_code == 404
            assert set(response.json()["error"]) == {"code", "message", "retryable", "details"}
            assert response.json()["request_id"]
        assert client.post("/api/runs", json=run_body(model_profile="other")).status_code == 409
        assert client.post("/api/runs", json=run_body(suite_id="missing")).status_code == 404
        response = client.post("/api/runs", json=run_body(api_key="secret-pasted-value"))
        assert response.status_code == 422
        assert "secret-pasted-value" not in response.text
        assert client.post("/api/runs", json=run_body(limit=0)).status_code == 422
        assert (
            client.post("/api/minimizations", json={"run_id": "x", "case_id": "y"}).status_code
            == 404
        )


def test_cached_run_completes_with_zero_live_calls(settings):
    spec = build_suite().families[0].spec
    responses = []
    for chart in (spec.chart, spec.transformed()):
        responses.extend([json.dumps({"answer": answer(chart, spec.question)})] * 3)
    adapter = FakeAdapterForTests(responses)
    with TestClient(create_app(settings, adapter_factory=lambda: adapter)) as client:
        first = client.post("/api/runs", json=run_body())
        assert first.status_code == 202
        rid = first.json()["run_id"]
        result = poll(client, f"/api/runs/{rid}")
        assert result["status"] == "completed"
        assert result["live_count"] == result["attempts_used"] == 6
        second = client.post("/api/runs", json=run_body()).json()["run_id"]
        cached = poll(client, f"/api/runs/{second}")
        assert cached["cached_count"] == 6
        assert cached["live_count"] == cached["attempts_used"] == 0
        assert adapter.calls == 6
        assert client.post(f"/api/runs/{rid}/cancel").status_code == 200
        assert client.post(f"/api/runs/{rid}/cancel").json()["cancel_requested"] is False


def test_queue_one_active_four_waiting_429_and_cancel_idempotent(settings):
    entered, release = threading.Event(), threading.Event()

    class BlockingFakeAdapterForTests(FakeAdapterForTests):
        def complete(self, *args, **kwargs):
            result = super().complete(*args, **kwargs)
            entered.set()
            assert release.wait(15)
            return result

    adapter = BlockingFakeAdapterForTests(['{"answer":"wrong"}'] * 100)
    with TestClient(create_app(settings, adapter_factory=lambda: adapter)) as client:
        active = client.post("/api/runs", json=run_body()).json()["run_id"]
        assert entered.wait(5)
        waiting = [client.post("/api/runs", json=run_body()).json()["run_id"] for _ in range(4)]
        full = client.post("/api/runs", json=run_body())
        assert full.status_code == 429
        assert full.json()["error"]["retryable"] is True
        assert client.get("/api/health").json()["waiting_jobs"] == 4
        assert (
            client.post(
                "/api/minimizations", json={"run_id": active, "case_id": "hard-largest_category-01"}
            ).status_code
            == 409
        )
        for rid in waiting:
            a = client.post(f"/api/runs/{rid}/cancel").json()
            b = client.post(f"/api/runs/{rid}/cancel").json()
            assert a == b
            assert b["status"] == "cancelled"
        assert client.get("/api/health").json()["waiting_jobs"] == 0
        client.post(f"/api/runs/{active}/cancel")
        release.set()
        result = poll(client, f"/api/runs/{active}")
        assert result["status"] == "cancelled"
        assert adapter.calls == 1


def test_restart_marks_old_running_and_queued_interrupted(settings):
    store = Store(settings.data_dir)
    store.initialize()
    for status in ("running", "queued"):
        store.create_job(status, "run", run_body())
        store.update_job(status, status, {})
        store.start_run(
            "hard_v1", "gemma", False, {"used": 0, "cap": 120}, run_id=status, status=status
        )
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:
        for rid in ("running", "queued"):
            assert client.get(f"/api/runs/{rid}").json()["status"] == "interrupted"
            assert store.get_run(rid)["status"] == "interrupted"


def test_artifacts_known_only_no_traversal(settings):
    store = Store(settings.data_dir)
    store.initialize()
    pair = store.create(build_suite().families[0].spec)
    digest = pair["members"][0]["image_hash"]
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:
        response = client.get(f"/api/artifacts/{digest}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        for path in (
            "/api/artifacts/" + "a" * 64,
            "/api/artifacts/..%2F.env",
            "/api/artifacts/.env",
        ):
            response = client.get(path)
            assert response.status_code == 404
            assert "filesystem" not in response.text


def test_minimization_routes_confirmed_regression_and_cancel(settings):
    store = Store(settings.data_dir)
    suite = build_suite()
    original = suite.families[0].spec
    responses = [json.dumps({"answer": answer(original.chart, original.question)})] * 3
    responses += ['{"answer":"wrong"}'] * 150
    adapter = FakeAdapterForTests(responses)
    source = run_suite(suite, EvaluationProfile("gemma", adapter, store), False, Budget(6), limit=1)
    with TestClient(create_app(settings, adapter_factory=lambda: adapter)) as client:
        response = client.post(
            "/api/minimizations",
            json={
                "run_id": source["run_id"],
                "case_id": "hard-largest_category-01",
                "side": "transformed",
            },
        )
        assert response.status_code == 202
        mid = response.json()["minimization_id"]
        result = poll(client, f"/api/minimizations/{mid}")
        assert result["confirmed"]
        assert result["end_size"][0] < result["start_size"][0]
        assert result["calls_used"] <= 60
        assert result["steps"]
        assert "secret" not in json.dumps(result)
        exported = client.post("/api/regressions", json={"minimization_id": mid})
        assert exported.status_code == 200
        assert exported.json()["confirmed"]
        digest = exported.json()["artifacts"]["after_original"]
        assert client.get(f"/api/artifacts/{digest}").status_code == 200
        a = client.post(f"/api/minimizations/{mid}/cancel").json()
        b = client.post(f"/api/minimizations/{mid}/cancel").json()
        assert a == b
        assert (
            client.post("/api/regressions", json={"minimization_id": "missing"}).status_code == 404
        )


def test_no_failure_is_409(settings):
    store = Store(settings.data_dir)
    suite = build_suite()
    responses = []
    spec = suite.families[0].spec
    for chart in (spec.chart, spec.transformed()):
        responses += [json.dumps({"answer": answer(chart, spec.question)})] * 3
    source = run_suite(
        suite,
        EvaluationProfile("gemma", FakeAdapterForTests(responses), store),
        False,
        Budget(6),
        limit=1,
    )
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:
        response = client.post(
            "/api/minimizations",
            json={"run_id": source["run_id"], "case_id": "hard-largest_category-01"},
        )
        assert response.status_code == 409


def test_storage_failure_is_safe_503(settings, monkeypatch):
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:

        def broken():
            raise sqlite3.OperationalError("database path and secrets must not leak")

        monkeypatch.setattr(client.app.state.store, "check", broken)
        response = client.get("/api/health")
        assert response.status_code == 503
        assert "must not leak" not in response.text


def test_cors_localhost_only(settings):
    with TestClient(create_app(settings, adapter_factory=FakeAdapterForTests)) as client:
        local = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert local.headers["access-control-allow-origin"] == "http://localhost:5173"
        external = client.get("/api/health", headers={"Origin": "https://example.com"})
        assert "access-control-allow-origin" not in external.headers
