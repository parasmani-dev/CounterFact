"""One durable in-process inference queue. HTTP handlers only delegate here."""

import asyncio
import os
import re
import threading
import time
import uuid

from counterfact.adapter import InferenceConfig, OpenAICompatibleAdapter
from counterfact.budget import Budget
from counterfact.evidence import stored_failure
from counterfact.minimize import DispatchStopped, GuardedBudget, minimize
from counterfact.runner import EvaluationProfile, run_suite
from counterfact.store import Store
from counterfact.suites import HardSuite, balanced_order, generate_suite


class ServiceError(Exception):
    def __init__(self, status, code, message, retryable=False):
        self.status, self.code, self.message, self.retryable = status, code, message, retryable
        super().__init__(message)


class JobService:
    def __init__(self, settings, *, adapter_factory=None):
        self.store = Store(settings.data_dir)
        self.settings = settings
        self.adapter_factory = adapter_factory or (
            lambda: OpenAICompatibleAdapter(InferenceConfig.from_env())
        )
        self.test_factory = adapter_factory is not None
        self.queue = asyncio.Queue(maxsize=4)
        self.flags, self.budgets, self.started = {}, {}, {}
        self.active = None
        self.worker_task = None

    async def start(self):
        self.store.initialize()
        self.store.interrupt_jobs()
        path = self.settings.data_dir / "suites/hard_v1.json"
        if path.exists():
            self.suite = HardSuite.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            self.suite = generate_suite(data_dir=self.settings.data_dir, output_path=path)
        self.suite = balanced_order(self.suite)
        self.worker_task = asyncio.create_task(self.worker())

    async def close(self):
        for flag in self.flags.values():
            flag.set()
        await self.queue.join()
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass

    def key_present(self):
        return self.test_factory or bool(os.getenv("GEMMA_API_KEY", "").strip())

    def models(self):
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT status FROM run WHERE profile='gemma' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return [
            {
                "name": "gemma",
                "key_present": self.key_present(),
                "last_status": row[0] if row else None,
            }
        ]

    def suites(self):
        return [
            {
                "id": self.suite.suite_id,
                "synthetic": True,
                "family_count": len(self.suite.families),
                "question_types": ["largest_category", "value_lookup", "above_threshold"],
            }
        ]

    async def enqueue(self, kind, payload):
        if self.queue.full():
            raise ServiceError(429, "queue_full", "Four jobs already waiting", True)
        if not self.key_present():
            raise ServiceError(409, "profile_unavailable", "Gemma environment key is missing")
        job_id = uuid.uuid4().hex
        self.store.enqueue_job(job_id, kind, payload)
        self.flags[job_id] = threading.Event()
        self.queue.put_nowait(job_id)
        return job_id

    async def start_run(self, body):
        if body["suite_id"] != self.suite.suite_id:
            raise ServiceError(404, "suite_not_found", "Unknown suite")
        if body["model_profile"] != "gemma":
            raise ServiceError(409, "profile_unavailable", "Only gemma is configured")
        return await self.enqueue("run", body)

    async def start_minimization(self, body):
        run = self.store.get_run(body["run_id"])
        if run is None:
            raise ServiceError(404, "run_not_found", "Unknown run")
        if run["status"] in ("running", "queued"):
            raise ServiceError(409, "run_not_finished", "Wait until the source run finishes")
        if not any(f.family_id == body["case_id"] for f in self.suite.families):
            raise ServiceError(404, "case_not_found", "Unknown case")
        selected = stored_failure(
            self.store,
            self.suite,
            run_id=body["run_id"],
            case_id=body["case_id"],
            side=body.get("side"),
        )
        if selected is None:
            raise ServiceError(409, "no_failure", "No completed strict-scored failure on this case")
        return await self.enqueue("minimization", selected)

    def poll(self, job_id, kind):
        job = self.store.get_job(job_id)
        if job is None and kind == "run":
            historical = self.store.get_run(job_id)
            if historical:
                return historical
        if job is None and kind == "minimization":
            saved = self.store.regression_for(job_id)
            if saved:
                return {"status": "completed", **saved["case"]}
        if job is None or job["kind"] != kind:
            raise ServiceError(404, "job_not_found", "Unknown job")
        result = {"id": job_id, "status": job["status"], **job["result"]}
        result["status"] = job["status"]
        if kind == "run":
            run = self.store.get_run(job_id)
            result.setdefault("cases", run["cases"] if run else [])
            if run:
                result.setdefault("budget", run["budget"])
                result.setdefault("cached_count", run["budget"].get("cached_count", 0))
                result.setdefault("live_count", run["budget"].get("live_count", 0))
            result.setdefault(
                "progress",
                {
                    "completed_families": len(result["cases"]),
                    "total_families": job["payload"].get("limit") or 20,
                },
            )
        else:
            result["steps"] = self.store.reduction_steps(job_id)
            result.setdefault("best_candidate", job["payload"]["case"]["spec"])
            accepted = [s for s in result["steps"] if s["accepted"] and s.get("candidate")]
            if accepted:
                result["best_candidate"] = accepted[-1]["candidate"]
            result.setdefault("confirmed", False)
            result.setdefault(
                "cache_hits",
                sum(
                    c["cached_count"]
                    for s in result["steps"]
                    for c in (s.get("evidence") or {}).values()
                ),
            )
            result.setdefault("validity_rejections", sum(not s["valid"] for s in result["steps"]))
            result.setdefault(
                "terminal_reason", "interrupted" if job["status"] == "interrupted" else None
            )
            result.setdefault(
                "elapsed",
                round(time.monotonic() - self.started[job_id], 3) if job_id in self.started else 0,
            )
            result.setdefault("cap", 60)
            observed_attempts = sum(
                c["attempts_used"]
                for step in result["steps"]
                for c in (step.get("evidence") or {}).values()
            )
            result.setdefault("calls_used", observed_attempts)
            result.setdefault("budget", {"used": result["calls_used"], "cap": result["cap"]})
        result.setdefault("cached_count", 0)
        result.setdefault("live_count", 0)
        budget = self.budgets.get(job_id)
        if budget:
            result["budget"] = budget.report()
            result["attempts_used"] = budget.used
            if kind == "minimization":
                result["calls_used"] = budget.used
        else:
            result.setdefault("budget", {"used": 0, "cap": 60 if kind == "minimization" else 120})
        if kind == "minimization":
            result.setdefault("calls_used", result["budget"]["used"])
        result.setdefault("attempts_used", result["budget"]["used"])
        return result

    def cancel(self, job_id, kind):
        job = self.store.get_job(job_id)
        if job is None:
            old = self.store.get_run(job_id) if kind == "run" else self.store.regression_for(job_id)
            if old:
                return {
                    "id": job_id,
                    "cancel_requested": False,
                    "status": old.get("status", "completed"),
                }
        if job is None or job["kind"] != kind:
            raise ServiceError(404, "job_not_found", "Unknown job")
        if job["status"] in ("queued", "running"):
            self.flags[job_id].set()
            if job["status"] == "queued":
                self.store.update_job(job_id, "cancelled", {"terminal_reason": "cancelled"})
                if kind == "run":
                    self.store.finish_run(job_id, "cancelled", {"used": 0, "cap": 120})
                waiting = []
                while not self.queue.empty():
                    queued = self.queue.get_nowait()
                    self.queue.task_done()
                    if queued != job_id:
                        waiting.append(queued)
                for queued in waiting:
                    self.queue.put_nowait(queued)
        return {
            "id": job_id,
            "cancel_requested": job["status"] in ("queued", "running", "cancelled"),
            "status": self.store.get_job(job_id)["status"],
        }

    def regression(self, mid):
        row = self.store.regression_for(mid)
        if row is None:
            if self.store.get_job(mid):
                raise ServiceError(409, "not_ready", "Minimization has not produced a regression")
            raise ServiceError(404, "minimization_not_found", "Unknown minimization")
        if not row["confirmed"]:
            raise ServiceError(409, "unconfirmed", "Candidate lacks fresh confirmation")
        return row

    def artifact(self, digest):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ServiceError(404, "artifact_not_found", "Unknown artifact")
        path = self.store.artifact_path(digest)
        if path is None:
            raise ServiceError(404, "artifact_not_found", "Unknown artifact")
        return path

    def execute(self, job_id):
        job = self.store.get_job(job_id)
        flag = self.flags[job_id]
        if flag.is_set():
            return {"status": "cancelled", "terminal_reason": "cancelled"}
        adapter = self.adapter_factory()
        payload = job["payload"]
        budget = self.budgets[job_id]
        if job["kind"] == "minimization":
            return minimize(
                payload["case"],
                payload["predicate"],
                adapter,
                self.store,
                budget,
                cancel_flag=flag,
                minimization_id=job_id,
            )
        cases = []

        def guard():
            if flag.is_set():
                raise DispatchStopped("cancelled")

        guarded = GuardedBudget(budget, 0, budget.cap, guard)

        def progress(case):
            cases.append(case)
            self.store.update_job(
                job_id,
                "running",
                {
                    "cases": cases,
                    "progress": {
                        "completed_families": len(cases),
                        "total_families": payload.get("limit") or 20,
                    },
                    "cached_count": sum(
                        c[s]["cached_count"]
                        for c in cases
                        for s in ("original", "transformed")
                        if c[s]
                    ),
                    "live_count": sum(
                        c[s]["live_count"]
                        for c in cases
                        for s in ("original", "transformed")
                        if c[s]
                    ),
                },
            )

        return run_suite(
            self.suite,
            EvaluationProfile("gemma", adapter, self.store),
            payload["fresh"],
            guarded,
            payload.get("limit"),
            progress,
            run_id=job_id,
            cancel_flag=flag,
        )

    async def worker(self):
        while True:
            job_id = await self.queue.get()
            try:
                self.active = job_id
                self.started[job_id] = time.monotonic()
                job = self.store.get_job(job_id)
                self.budgets[job_id] = Budget(60 if job["kind"] == "minimization" else 120)
                self.store.update_job(job_id, "running", {})
                result = await asyncio.to_thread(self.execute, job_id)
                status = result.get("status", "completed")
                if result.get("terminal_reason") == "cancelled":
                    status = "cancelled"
                if result.get("terminal_reason") == "provider_unavailable":
                    status = "provider_error"
                self.store.update_job(job_id, status, result)
                if job["kind"] == "run" and status == "cancelled":
                    self.store.finish_run(job_id, status, self.budgets[job_id].report())
            except Exception:
                self.store.update_job(
                    job_id,
                    "error",
                    {
                        "error": {
                            "code": "job_failed",
                            "message": "Job failed; partial evidence retained",
                            "retryable": False,
                            "details": None,
                        }
                    },
                )
                if self.store.get_job(job_id)["kind"] == "run":
                    self.store.finish_run(job_id, "error", self.budgets[job_id].report())
            finally:
                self.active = None
                self.queue.task_done()
