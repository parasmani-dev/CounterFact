"""Synthetic model replies are confined to FakeAdapterForTests unit tests."""

import threading

import httpx
import pytest
from pydantic import ValidationError

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget
from counterfact.minimize import delete_categories, minimize, protected_categories
from counterfact.reference import answer
from counterfact.schemas import PairSpec
from counterfact.store import Store
from counterfact.suites import build_suite
from tests.test_adapter import FakeTransportForTests
from tests.test_phase2 import FakeAdapterForTests


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path)
    store.initialize()
    spec = next(f.spec for f in build_suite().families if f.family_id == "hard-above_threshold-01")
    return store, spec


def test_reduces_and_confirms_with_protection_and_strictly_smaller_steps(setup):
    store, spec = setup
    adapter = FakeAdapterForTests(['{"answer":true}'] * 100)
    result = minimize(spec, "wrong_answer", adapter, store, Budget(60))
    assert result["confirmed"]
    assert result["end_size"] == [2, 1]
    assert result["calls_used"] == adapter.calls <= 60
    best = PairSpec.model_validate(result["best_candidate"])
    assert "group_10" in {c.id for c in best.chart.categories}
    accepted = [s["size"] for s in result["steps"] if s["phase"] == "search" and s["accepted"]]
    assert all(
        tuple(b) < tuple(a)
        for a, b in zip([result["start_size"]] + accepted, accepted, strict=False)
    )
    assert result["steps"][-1]["phase"] == "confirmation"
    assert all(not c["cached_count"] for c in result["steps"][-1]["evidence"].values())
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM reduction_step").fetchone()[0] == len(
            result["steps"]
        )
        assert connection.execute("SELECT confirmed FROM regression_case").fetchone()[0] == 1


def test_pair_deletion_applies_same_ids_and_recomputes_reference(setup):
    store, spec = setup
    adapter = FakeAdapterForTests(
        ['{"answer":false}'] * 3 + ['{"answer":false}'] * 3 + ['{"answer":false}'] * 100
    )
    result = minimize(spec, "correct_to_incorrect", adapter, store, Budget(60))
    assert result["confirmed"]
    best = PairSpec.model_validate(result["best_candidate"])
    assert [c.id for c in best.chart.categories] == [c.id for c in best.transformed().categories]
    assert answer(best.chart, best.question) is False
    assert answer(best.transformed(), best.question) is True
    assert result["steps"][-1]["evidence"]["original"]["verdict"] == "passes"
    assert result["steps"][-1]["evidence"]["transformed"]["verdict"] == "fails"


def test_protects_both_unique_winners_and_transformation_target():
    spec = build_suite().families[0].spec
    protected = protected_categories(spec)
    assert max(spec.chart.categories, key=lambda c: c.value).id in protected
    assert max(spec.transformed().categories, key=lambda c: c.value).id in protected
    assert spec.mutations[0].category_id in protected


def test_invalid_candidate_and_identity_are_rejected_before_inference(setup):
    store, spec = setup
    with pytest.raises(ValidationError):
        delete_categories(spec, {c.id for c in spec.chart.categories if c.id != "group_10"})
    payload = spec.model_dump()
    payload["mutations"][0]["value"] = 1040.0
    adapter = FakeAdapterForTests()
    with pytest.raises(ValidationError):
        minimize(payload, "wrong_answer", adapter, store, Budget(60))
    assert adapter.calls == 0


def test_confirmation_failure_is_unconfirmed(setup):
    store, spec = setup
    adapter = FakeAdapterForTests(['{"answer":true}'] * 3 + ['{"answer":false}'] * 3)
    result = minimize(spec, "wrong_answer", adapter, store, Budget(6))
    assert not result["confirmed"]
    assert result["terminal_reason"] == "confirmation_failed"
    assert result["description"] == "smallest observed candidate (unconfirmed)"
    assert result["calls_used"] == 6


def test_no_reproducible_baseline(setup):
    store, spec = setup
    result = minimize(
        spec, "wrong_answer", FakeAdapterForTests(['{"answer":false}'] * 3), store, Budget(60)
    )
    assert result["terminal_reason"] == "no_reproducible_failure"
    assert not result["confirmed"]


@pytest.mark.parametrize("cap", [0, 1, 3, 6, 7, 12])
def test_hard_cap_includes_adapter_retries(setup, cap):
    store, spec = setup
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    adapter = OpenAICompatibleAdapter(
        InferenceConfig("https://test.example/v1", "test", "secret"),
        transport=FakeTransportForTests(handler),
        sleep_fn=lambda _: None,
    )
    result = minimize(spec, "wrong_answer", adapter, store, Budget(cap), max_calls=cap)
    assert result["calls_used"] == len(calls) <= cap
    assert not result["confirmed"]


def test_cancel_before_dispatch_and_during_retry(setup):
    store, spec = setup
    flag = threading.Event()
    flag.set()
    adapter = FakeAdapterForTests()
    result = minimize(spec, "wrong_answer", adapter, store, Budget(60), cancel_flag=flag)
    assert result["terminal_reason"] == "cancelled"
    assert adapter.calls == 0
    flag.clear()
    calls = []

    def handler(request):
        calls.append(request)
        flag.set()
        return httpx.Response(500)

    real = OpenAICompatibleAdapter(
        InferenceConfig("https://test.example/v1", "test", "secret"),
        transport=FakeTransportForTests(handler),
        sleep_fn=lambda _: None,
    )
    result = minimize(spec, "wrong_answer", real, store, Budget(60), cancel_flag=flag)
    assert result["terminal_reason"] == "cancelled"
    assert len(calls) == 1


def test_time_limit_before_dispatch(setup):
    store, spec = setup
    adapter = FakeAdapterForTests()
    result = minimize(spec, "wrong_answer", adapter, store, Budget(60), time_limit=0.000001)
    assert result["terminal_reason"] == "time_limit"
    assert adapter.calls == 0


def test_cancel_during_fresh_confirmation_is_reported_as_cancelled(setup):
    store, spec = setup
    flag = threading.Event()

    class CancellingFakeAdapterForTests(FakeAdapterForTests):
        def complete(self, *args, **kwargs):
            result = super().complete(*args, **kwargs)
            if self.calls == 4:
                flag.set()
            return result

    adapter = CancellingFakeAdapterForTests(['{"answer":true}'] * 6)
    result = minimize(spec, "wrong_answer", adapter, store, Budget(6), cancel_flag=flag)
    assert result["terminal_reason"] == "cancelled"
    assert not result["confirmed"]
    assert adapter.calls == result["calls_used"] == 4
    assert result["steps"][-1]["evidence"]["original"]["verdict"] == "inconclusive"


def test_reference_is_recomputed_for_reduced_candidate_charts(setup, monkeypatch):
    import importlib

    module = importlib.import_module("counterfact.minimize")
    store, spec = setup
    counts = []
    original_answer = module.answer

    def recording_answer(chart, question):
        counts.append(len(chart.categories))
        return original_answer(chart, question)

    monkeypatch.setattr(module, "answer", recording_answer)
    result = minimize(
        spec, "wrong_answer", FakeAdapterForTests(['{"answer":true}'] * 100), store, Budget(60)
    )
    assert result["confirmed"]
    assert 17 in counts and 2 in counts
    assert len(set(counts)) >= 4
