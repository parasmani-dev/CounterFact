"""Phase 1 HTTP surface: health and generated inputs, with no inference endpoints."""

import logging
import re
import sqlite3
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from counterfact import __version__
from counterfact.config import Settings
from counterfact.schemas import PairSpec
from counterfact.storage import PairStore

logger = logging.getLogger("counterfact")


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    store = PairStore(config.data_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.initialize()
        app.state.store = store
        yield

    app = FastAPI(title="CounterFact", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request=%s method=%s status=%s",
            request.state.request_id,
            request.method,
            response.status_code,
        )
        return response

    def error(request: Request, status: int, code: str, message: str, details=None):
        return JSONResponse(
            status_code=status,
            content={
                "error": {"code": code, "message": message, "details": details},
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Never echo submitted payloads, which could contain accidentally pasted secrets.
        issues = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return error(request, 422, "invalid_input", "Invalid chart specification", issues)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error(request, exc.status_code, "request_error", str(exc.detail))

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(OSError)
    async def storage_error(request: Request, exc: Exception):
        logger.error(
            "storage unavailable request=%s type=%s", request.state.request_id, type(exc).__name__
        )
        return error(request, 503, "storage_unavailable", "Local storage is unavailable")

    @app.get("/api/health")
    def health():
        store.check()
        return {"status": "ok", "version": __version__, "phase": 1, "inference_ready": False}

    @app.post("/api/pairs", status_code=201)
    def create_pair(spec: PairSpec):
        return store.create(spec)

    def existing_pair(pair_id: str):
        if not re.fullmatch(r"[0-9a-f]{64}", pair_id):
            raise HTTPException(404, "Pair not found")
        pair = store.get(pair_id)
        if pair is None:
            raise HTTPException(404, "Pair not found")
        return pair

    @app.get("/api/pairs/{pair_id}")
    def get_pair(pair_id: str):
        return existing_pair(pair_id)

    @app.get("/api/pairs/{pair_id}/{variant}.png")
    def get_image(pair_id: str, variant: Literal["original", "transformed"]):
        existing_pair(pair_id)
        path = config.data_dir / "pairs" / pair_id / f"{variant}.png"
        if not path.is_file():
            raise HTTPException(404, "Image not found")
        return FileResponse(path, media_type="image/png")

    return app
