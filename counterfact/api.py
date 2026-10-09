"""Thin local FastAPI control plane for the CounterFact engine."""

import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from counterfact import __version__
from counterfact.config import Settings
from counterfact.service import JobService, ServiceError

logger = logging.getLogger("counterfact")


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RunBody(RequestBody):
    suite_id: str
    model_profile: str
    fresh: bool = False
    limit: Annotated[int, Field(ge=1, le=20)] | None = None


class MinimizeBody(RequestBody):
    run_id: str
    case_id: str
    side: Literal["original", "transformed"] | None = None


class RegressionBody(RequestBody):
    minimization_id: str


def create_app(settings: Settings | None = None, *, adapter_factory=None) -> FastAPI:
    service = JobService(settings or Settings.from_env(), adapter_factory=adapter_factory)

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        app.state.service = service
        app.state.store = service.store
        yield
        await service.close()

    app = FastAPI(title="CounterFact", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def redact(value):
        # Never return configured credentials, including a provider echo in raw evidence.
        import os

        keys = [v for k, v in os.environ.items() if k.endswith("API_KEY") and v]

        def walk(item):
            if isinstance(item, str):
                for key in keys:
                    item = item.replace(key, "[REDACTED]")
                return item
            if isinstance(item, dict):
                return {k: walk(v) for k, v in item.items() if k not in ("api_key", "API_KEY")}
            if isinstance(item, list):
                return [walk(v) for v in item]
            return item

        return walk(value)

    def error(request, status, code, message, retryable=False, details=None):
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "retryable": retryable,
                    "details": details,
                },
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    @app.middleware("http")
    async def request_id(request, call_next):
        request.state.request_id = uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            response = error(request, 500, "internal_error", "Request failed", False)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request=%s method=%s status=%s",
            request.state.request_id,
            request.method,
            response.status_code,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        details = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return error(request, 422, "invalid_input", "Invalid request", details=details)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc):
        return error(request, exc.status, exc.code, exc.message, exc.retryable)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc):
        return error(
            request,
            exc.status_code,
            "request_error",
            "Resource not found" if exc.status_code == 404 else "Request rejected",
        )

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(OSError)
    async def storage_error(request: Request, exc):
        return error(request, 503, "storage_unavailable", "Local storage is unavailable", True)

    @app.get("/api/health")
    def health():
        service.store.check()
        return {
            "status": "ok",
            "version": __version__,
            "phase": 4,
            "inference_ready": service.key_present(),
            "active_job": service.active,
            "waiting_jobs": service.queue.qsize(),
        }

    @app.get("/api/models")
    def models():
        return redact(service.models())

    @app.get("/api/suites")
    def suites():
        return service.suites()

    @app.post("/api/runs", status_code=202)
    async def start_run(body: RunBody):
        job_id = await service.start_run(body.model_dump())
        return {"run_id": job_id, "status": "queued"}

    @app.get("/api/runs/{job_id}")
    def run(job_id: str):
        return redact(service.poll(job_id, "run"))

    @app.post("/api/runs/{job_id}/cancel")
    async def cancel_run(job_id: str):
        return service.cancel(job_id, "run")

    @app.post("/api/minimizations", status_code=202)
    async def start_minimization(body: MinimizeBody):
        mid = await service.start_minimization(body.model_dump())
        return {"minimization_id": mid, "status": "queued"}

    @app.get("/api/minimizations/{job_id}")
    def minimization(job_id: str):
        return redact(service.poll(job_id, "minimization"))

    @app.post("/api/minimizations/{job_id}/cancel")
    async def cancel_minimization(job_id: str):
        return service.cancel(job_id, "minimization")

    @app.post("/api/regressions")
    def regression(body: RegressionBody):
        return redact(service.regression(body.minimization_id))

    @app.get("/api/artifacts/{digest}")
    def artifact(digest: str):
        return FileResponse(service.artifact(digest), media_type="image/png")

    return app
