"""UI evidence projections are read-only; test inference uses FakeAdapterForTests."""

import json

from fastapi.testclient import TestClient

from counterfact.api import create_app
from counterfact.budget import Budget
from counterfact.config import Settings
from counterfact.reference import answer
from counterfact.runner import EvaluationProfile, run_suite
from counterfact.store import Store
from counterfact.suites import build_suite
from tests.test_phase2 import FakeAdapterForTests


def test_ui_reads_return_full_evidence_without_dispatch_or_writes(tmp_path):
    suite = build_suite()
    path = tmp_path / "suites/hard_v1.json"
    path.parent.mkdir()
    path.write_text(suite.model_dump_json())
    spec = suite.families[0].spec
    raw = '{"answer":"<img src=x onerror=alert(1)>"}'
    adapter = FakeAdapterForTests(
        [json.dumps({"answer": answer(spec.chart, spec.question)})] * 3 + [raw] * 3
    )
    store = Store(tmp_path)
    source = run_suite(suite, EvaluationProfile("gemma", adapter, store), False, Budget(6), limit=1)
    with TestClient(create_app(Settings(tmp_path), adapter_factory=lambda: adapter)) as client:
        with store.connect() as connection:
            before = [tuple(r) for r in connection.execute("SELECT * FROM observation ORDER BY id")]
        history = client.get("/api/runs")
        assert history.status_code == 200
        assert history.json()[0]["id"] == source["run_id"]
        prefix = f"/api/runs/{source['run_id']}/cases"
        result = client.get(prefix)
        assert result.status_code == 200
        case = result.json()[0]
        assert case["eligible"] is True
        assert case["pair_status"] == "correct_to_incorrect"
        assert case["original"]["expected_answer"] == answer(spec.chart, spec.question)
        for side in ("original", "transformed"):
            assert len(case[side]["observations"]) == 3
            assert case[side]["image_hash"]
            assert client.get("/api/artifacts/" + case[side]["image_hash"]).status_code == 200
        assert case["transformed"]["observations"][0]["raw_text"] == raw
        assert client.get(prefix + "/" + case["case_id"]).json() == case
        assert client.get("/api/runs/missing/cases").status_code == 404
        assert client.get(prefix + "/missing").status_code == 404
        with store.connect() as connection:
            after = [tuple(r) for r in connection.execute("SELECT * FROM observation ORDER BY id")]
        assert before == after
        assert adapter.calls == 6
