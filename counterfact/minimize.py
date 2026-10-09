"""Validity-preserving reduction with bounded acquisition and fresh confirmation."""

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable

from pydantic import ValidationError

from counterfact.budget import Budget, BudgetExceeded, DispatchStopped
from counterfact.check import generation_params, run_check
from counterfact.reference import answer, question_text
from counterfact.render import BarRenderer, image_hash
from counterfact.schemas import PairSpec
from counterfact.store import Store


class GuardedBudget:
    """Check limits immediately before every attempt, including adapter retries."""

    def __init__(self, parent, start, cap, guard, leave=0):
        self.parent, self.start, self.cap = parent, start, cap
        self.guard, self.leave = guard, leave

    def reserve(self, n=1):
        self.guard()
        if self.parent.used - self.start + n + self.leave > self.cap:
            raise BudgetExceeded("Fresh confirmation allowance protected")
        self.parent.reserve(n)

    @property
    def used(self):
        return self.parent.used

    def report(self):
        return self.parent.report()


def case_spec(case):
    if isinstance(case, PairSpec):
        return PairSpec.model_validate(case.model_dump()), "original"
    if isinstance(case, dict) and "spec" in case:
        spec = case["spec"]
        side = case.get("side", "original")
        if side not in ("original", "transformed"):
            raise ValueError("side must be original or transformed")
        return PairSpec.model_validate(
            spec.model_dump() if isinstance(spec, PairSpec) else spec
        ), side
    return PairSpec.model_validate(case), "original"


def protected_categories(spec):
    protected = set()
    if spec.question.target_id:
        protected.add(spec.question.target_id)
    if spec.question.type == "largest_category":
        for chart in (spec.chart, spec.transformed()):
            protected.add(max(chart.categories, key=lambda c: c.value).id)
    for mutation in spec.mutations:
        if mutation.type == "change_value":
            protected.add(mutation.category_id)
        else:
            # A reorder touches IDs whose positions actually change.
            original = [c.id for c in spec.chart.categories]
            protected.update(a for a, b in zip(original, mutation.order, strict=True) if a != b)
    return protected


def delete_categories(spec, removed):
    payload = spec.model_dump()
    payload["chart"]["categories"] = [
        c for c in payload["chart"]["categories"] if c["id"] not in removed
    ]
    for mutation in payload["mutations"]:
        if mutation["type"] == "reorder":
            mutation["order"] = [cid for cid in mutation["order"] if cid not in removed]
    return PairSpec.model_validate(payload)


def size(spec):
    return len(spec.chart.categories), len(spec.mutations)


def candidate_hash(spec):
    return hashlib.sha256(
        json.dumps(spec.model_dump(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def minimize(
    case,
    predicate,
    adapter,
    store: Store,
    budget: Budget,
    max_calls=60,
    time_limit=600,
    cancel_flag=None,
    *,
    on_step: Callable | None = None,
    minimization_id=None,
):
    if predicate not in ("wrong_answer", "correct_to_incorrect"):
        raise ValueError("Unknown failure predicate")
    if type(max_calls) is not int or not 0 <= max_calls <= 60:
        raise ValueError("max_calls must be between 0 and 60")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    spec, side = case_spec(case)
    store.initialize()
    mid = minimization_id or uuid.uuid4().hex
    start, start_used = time.monotonic(), budget.used
    cap = min(max_calls, budget.cap - start_used)
    reserve = 6 if predicate == "correct_to_incorrect" else 3
    steps, cache_hits, rejections = [], 0, 0
    best, confirmed = spec, False
    terminal = "search_completed"
    renderer = BarRenderer()

    def guard():
        cancelled = (
            cancel_flag()
            if callable(cancel_flag)
            else (cancel_flag.is_set() if cancel_flag is not None else False)
        )
        if cancelled:
            raise DispatchStopped("cancelled")
        if time.monotonic() - start >= time_limit:
            raise DispatchStopped("time_limit")

    search_budget = GuardedBudget(budget, start_used, cap, guard, reserve)
    final_budget = GuardedBudget(budget, start_used, cap, guard)

    def record(phase, candidate, valid, evidence=None, accepted=False, reason=None):
        step = {
            "sequence": len(steps),
            "phase": phase,
            "candidate_hash": candidate_hash(candidate),
            "candidate": candidate.model_dump(),
            "valid": valid,
            "size": list(size(candidate)),
            "evidence": evidence,
            "accepted": accepted,
            "reason": reason,
        }
        store.save_reduction_step(mid, step)
        steps.append(step)
        if on_step:
            on_step(step)

    def evaluate(candidate, allowance, fresh=False):
        nonlocal cache_hits
        checks = {}
        sides = ("original", "transformed") if predicate == "correct_to_incorrect" else (side,)
        for which in sides:
            guard()
            chart = candidate.chart if which == "original" else candidate.transformed()
            png = renderer.render(chart, show_value_labels=candidate.show_value_labels)
            check = run_check(
                png,
                question_text(chart, candidate.question),
                answer(chart, candidate.question),
                candidate.question.type,
                adapter,
                store,
                allowance,
                fresh=fresh,
            )
            cache_hits += check["cached_count"]
            checks[which] = check
            if check["terminal_error"] != "none":
                break
        return checks

    def preserves(checks):
        if predicate == "wrong_answer":
            return checks.get(side, {}).get("verdict") == "fails"
        return (
            checks.get("original", {}).get("verdict") == "passes"
            and checks.get("transformed", {}).get("verdict") == "fails"
        )

    def error_reason(checks):
        for check in checks.values():
            if check["terminal_error"] in ("cancelled", "time_limit"):
                return check["terminal_error"]
        if any(c["budget_exhausted"] for c in checks.values()):
            return "budget_exhausted"
        if any(c["terminal_error"] != "none" for c in checks.values()):
            return "provider_unavailable"
        return None

    try:
        baseline = evaluate(spec, search_budget)
        record("baseline", spec, True, baseline, preserves(baseline))
        problem = error_reason(baseline)
        if problem:
            terminal = problem
        elif not preserves(baseline):
            terminal = "no_reproducible_failure"
        else:
            # ddmin partitions removable IDs; accepted reductions restart at halves.
            granularity = 2
            seen = set()
            while True:
                guard()
                removable = [
                    c.id for c in best.chart.categories if c.id not in protected_categories(best)
                ]
                if not removable:
                    break
                width = max(1, math.ceil(len(removable) / granularity))
                chunks = [removable[i : i + width] for i in range(0, len(removable), width)]
                accepted_any = False
                for chunk in chunks:
                    guard()
                    token = (candidate_hash(best), tuple(chunk))
                    if token in seen:
                        continue
                    seen.add(token)
                    try:
                        candidate = delete_categories(best, set(chunk))
                    except (ValidationError, ValueError):
                        rejections += 1
                        # Hash rejected raw edits as well, without constructing an invalid spec.
                        step = {
                            "sequence": len(steps),
                            "phase": "search",
                            "candidate_hash": hashlib.sha256(repr(token).encode()).hexdigest(),
                            "removed": chunk,
                            "valid": False,
                            "evidence": None,
                            "accepted": False,
                            "reason": "invalid_candidate",
                        }
                        store.save_reduction_step(mid, step)
                        steps.append(step)
                        if on_step:
                            on_step(step)
                        continue
                    if size(candidate) >= size(best):
                        rejections += 1
                        record("search", candidate, False, reason="not_smaller")
                        continue
                    if budget.used - start_used >= cap - reserve:
                        terminal = "budget_exhausted"
                        break
                    evidence = evaluate(candidate, search_budget)
                    accept = preserves(evidence)
                    record("search", candidate, True, evidence, accept)
                    if accept:
                        best = candidate
                        accepted_any = True
                    problem = error_reason(evidence)
                    if problem:
                        terminal = problem
                        break
                    if accept:
                        break
                if terminal != "search_completed":
                    break
                if accepted_any:
                    granularity = 2
                elif width == 1:
                    break
                else:
                    granularity = min(len(removable), granularity * 2)
            if terminal not in ("provider_unavailable", "cancelled", "time_limit"):
                confirmation = evaluate(best, final_budget, fresh=True)
                confirmed = preserves(confirmation)
                record("confirmation", best, True, confirmation, confirmed)
                problem = error_reason(confirmation)
                if not confirmed:
                    terminal = (
                        problem
                        if problem in ("provider_unavailable", "cancelled", "time_limit")
                        else "confirmation_failed"
                    )
    except DispatchStopped as exc:
        terminal = exc.reason
    except KeyboardInterrupt:
        terminal = "interrupted"

    result = {
        "id": mid,
        "predicate": predicate,
        "side": side,
        "best_candidate": best.model_dump(),
        "start_size": list(size(spec)),
        "end_size": list(size(best)),
        "steps": steps,
        "calls_used": budget.used - start_used,
        "cap": cap,
        "cache_hits": cache_hits,
        "elapsed": round(time.monotonic() - start, 3),
        "validity_rejections": rejections,
        "terminal_reason": terminal,
        "confirmed": confirmed,
        "description": "smallest confirmed case found"
        if confirmed
        else "smallest observed candidate (unconfirmed)",
        "generation_config": generation_params(adapter),
        "scorer_version": "1",
    }
    artifacts = {}
    for prefix, candidate in (("before", spec), ("after", best)):
        for which in ("original", "transformed"):
            chart = candidate.chart if which == "original" else candidate.transformed()
            png = renderer.render(chart, show_value_labels=candidate.show_value_labels)
            digest = image_hash(png)
            target = store.root / "regressions" / "artifacts"
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{digest}.png").write_bytes(png)
            artifacts[f"{prefix}_{which}"] = digest
    payload = {**result, "original_case": spec.model_dump(), "artifacts": artifacts}
    result["artifacts"] = artifacts
    result["regression_id"] = store.save_regression(mid, payload, artifacts)
    return result
