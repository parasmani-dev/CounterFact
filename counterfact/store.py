"""Schema-v2 migration and durable observations/runs alongside existing chart pairs."""

import json
import uuid
from datetime import UTC, datetime

from counterfact.storage import PairStore


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class Store(PairStore):
    def initialize(self) -> None:
        super().initialize()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (1, 2):
                raise RuntimeError("Unsupported database schema version")
            if version == 1:
                connection.execute(
                    "CREATE TABLE observation (id TEXT PRIMARY KEY, group_key TEXT NOT NULL, "
                    "slot INTEGER NOT NULL CHECK(slot BETWEEN 0 AND 2), raw_text TEXT NOT NULL, "
                    "status TEXT NOT NULL "
                    "CHECK(status IN ('correct','incorrect','invalid_output')), "
                    "error_type TEXT NOT NULL CHECK(error_type='none'), provider TEXT NOT NULL, "
                    "model TEXT NOT NULL, latency_ms REAL NOT NULL, acquired_at TEXT NOT NULL, "
                    "request_params_json TEXT NOT NULL, UNIQUE(group_key,slot))"
                )
                connection.execute(
                    "CREATE TABLE run (id TEXT PRIMARY KEY, suite_id TEXT NOT NULL, "
                    "profile TEXT NOT NULL, fresh INTEGER NOT NULL CHECK(fresh IN (0,1)), "
                    "status TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT, "
                    "budget_json TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE case_result (id TEXT PRIMARY KEY, run_id TEXT NOT NULL "
                    "REFERENCES run(id), case_id TEXT NOT NULL, side TEXT NOT NULL "
                    "CHECK(side IN ('original','transformed')), group_key TEXT NOT NULL, "
                    "verdict TEXT NOT NULL, cached_count INTEGER NOT NULL, "
                    "live_count INTEGER NOT NULL)"
                )
                connection.execute("PRAGMA user_version=2")
            # Check the full schema rather than accepting a corrupt version marker.
            for table in ("observation", "run", "case_result"):
                connection.execute(f"SELECT COUNT(*) FROM {table}")

    def cached_slots(self, base_key: str) -> dict[int, dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM observation WHERE group_key=? OR group_key LIKE ? "
                "ORDER BY acquired_at DESC,id DESC",
                (base_key, base_key + ":%"),
            ).fetchall()
        slots = {}
        for row in rows:
            slots.setdefault(row["slot"], dict(row))
        return slots

    def save_observation(self, observation: dict) -> dict:
        row = {**observation, "id": uuid.uuid4().hex, "acquired_at": timestamp()}
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO observation(id,group_key,slot,raw_text,status,error_type,provider,"
                "model,latency_ms,acquired_at,request_params_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                tuple(
                    row[key]
                    for key in (
                        "id",
                        "group_key",
                        "slot",
                        "raw_text",
                        "status",
                        "error_type",
                        "provider",
                        "model",
                        "latency_ms",
                        "acquired_at",
                        "request_params_json",
                    )
                ),
            )
        return row

    def start_run(self, suite_id: str, profile: str, fresh: bool, budget: dict) -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO run VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    suite_id,
                    profile,
                    int(fresh),
                    "running",
                    timestamp(),
                    None,
                    json.dumps(budget, sort_keys=True),
                ),
            )
        return run_id

    def save_case(self, run_id: str, case_id: str, side: str, result: dict, budget: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO case_result VALUES (?,?,?,?,?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    run_id,
                    case_id,
                    side,
                    result["group_key"],
                    result["verdict"],
                    result["cached_count"],
                    result["live_count"],
                ),
            )
            connection.execute(
                "UPDATE run SET budget_json=? WHERE id=?",
                (json.dumps(budget, sort_keys=True), run_id),
            )

    def finish_run(self, run_id: str, status: str, budget: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE run SET status=?,finished_at=?,budget_json=? WHERE id=?",
                (status, timestamp(), json.dumps(budget, sort_keys=True), run_id),
            )
