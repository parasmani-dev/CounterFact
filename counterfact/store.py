"""Schema-v4 migrations and durable observations/runs alongside existing chart pairs."""

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
            if version not in (1, 2, 3, 4):
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
            if version < 3:
                connection.execute(
                    "CREATE TABLE reduction_step (id TEXT PRIMARY KEY, "
                    "minimization_id TEXT NOT NULL, "
                    "sequence INTEGER NOT NULL, payload_json TEXT NOT NULL, "
                    "UNIQUE(minimization_id,sequence))"
                )
                connection.execute(
                    "CREATE TABLE regression_case (id TEXT PRIMARY KEY, "
                    "minimization_id TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, "
                    "confirmed INTEGER NOT NULL CHECK(confirmed IN (0,1)), "
                    "case_json TEXT NOT NULL, artifact_json TEXT NOT NULL)"
                )
                connection.execute("PRAGMA user_version=3")
            if version < 4:
                connection.execute(
                    "CREATE TABLE job (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                    "status TEXT NOT NULL, payload_json TEXT NOT NULL, result_json TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
                )
                connection.execute("PRAGMA user_version=4")
            # Check the full schema rather than accepting a corrupt version marker.
            for table in (
                "observation",
                "run",
                "case_result",
                "reduction_step",
                "regression_case",
                "job",
            ):
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

    def start_run(
        self,
        suite_id: str,
        profile: str,
        fresh: bool,
        budget: dict,
        *,
        run_id: str | None = None,
        status: str = "running",
    ) -> str:
        run_id = run_id or uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO run VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    suite_id,
                    profile,
                    int(fresh),
                    status,
                    timestamp(),
                    None,
                    json.dumps(budget, sort_keys=True),
                ),
            )
        return run_id

    def save_case(self, run_id: str, case_id: str, side: str, result: dict, budget: dict) -> None:
        case_result_id = uuid.uuid4().hex
        with self.connect() as connection:
            row = connection.execute("SELECT budget_json FROM run WHERE id=?", (run_id,)).fetchone()
            metadata = {**json.loads(row[0]), **budget}
            metadata.setdefault("observation_ids", {})[case_result_id] = [
                o["id"] for o in result["observations"] if o is not None
            ]
            connection.execute(
                "INSERT INTO case_result VALUES (?,?,?,?,?,?,?,?)",
                (
                    case_result_id,
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
                (json.dumps(metadata, sort_keys=True), run_id),
            )

    def finish_run(self, run_id: str, status: str, budget: dict) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT budget_json FROM run WHERE id=?", (run_id,)).fetchone()
            metadata = {**json.loads(row[0]), **budget} if row else budget
            connection.execute(
                "UPDATE run SET status=?,finished_at=?,budget_json=? WHERE id=?",
                (status, timestamp(), json.dumps(metadata, sort_keys=True), run_id),
            )

    def save_reduction_step(self, minimization_id: str, step: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO reduction_step VALUES (?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    minimization_id,
                    step["sequence"],
                    json.dumps(step, sort_keys=True),
                ),
            )

    def save_regression(self, minimization_id: str, payload: dict, artifacts: dict) -> str:
        regression_id = uuid.uuid4().hex
        target = self.root / "regressions" / regression_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "case.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO regression_case VALUES (?,?,?,?,?,?)",
                (
                    regression_id,
                    minimization_id,
                    timestamp(),
                    int(payload["confirmed"]),
                    json.dumps(payload, sort_keys=True),
                    json.dumps(artifacts, sort_keys=True),
                ),
            )
        return regression_id

    def enqueue_job(self, job_id: str, kind: str, payload: dict) -> None:
        """Commit the queued job and its run together; failures leave neither row."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO job VALUES (?,?,?,?,?,?,?)",
                (job_id, kind, "queued", json.dumps(payload), "{}", timestamp(), timestamp()),
            )
            if kind == "run":
                connection.execute(
                    "INSERT INTO run VALUES (?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        payload["suite_id"],
                        payload["model_profile"],
                        int(payload["fresh"]),
                        "queued",
                        timestamp(),
                        None,
                        json.dumps({"used": 0, "cap": 120}),
                    ),
                )

    def create_job(self, job_id: str, kind: str, payload: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO job VALUES (?,?,?,?,?,?,?)",
                (job_id, kind, "queued", json.dumps(payload), "{}", timestamp(), timestamp()),
            )

    def update_job(self, job_id: str, status: str, result: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE job SET status=?,result_json=?,updated_at=? "
                "WHERE id=? AND status!='interrupted'",
                (status, json.dumps(result), timestamp(), job_id),
            )

    def get_job(self, job_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "kind": row["kind"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def interrupt_jobs(self) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE job SET status='interrupted',updated_at=? "
                "WHERE status IN ('running','queued')",
                (timestamp(),),
            )
            connection.execute(
                "UPDATE run SET status='interrupted',finished_at=? "
                "WHERE status IN ('running','queued')",
                (timestamp(),),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
            cases = connection.execute(
                "SELECT case_id,side,verdict,cached_count,live_count "
                "FROM case_result WHERE run_id=?",
                (run_id,),
            ).fetchall()
        if row is None:
            return None
        return {
            "id": row["id"],
            "status": row["status"],
            "suite_id": row["suite_id"],
            "model_profile": row["profile"],
            "fresh": bool(row["fresh"]),
            "budget": json.loads(row["budget_json"]),
            "cases": [dict(c) for c in cases],
        }

    def reduction_steps(self, mid: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM reduction_step WHERE minimization_id=? ORDER BY sequence",
                (mid,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def regression_for(self, mid: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM regression_case WHERE minimization_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (mid,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "confirmed": bool(row["confirmed"]),
            "case": json.loads(row["case_json"]),
            "artifacts": json.loads(row["artifact_json"]),
        }

    def artifact_path(self, digest: str):
        from counterfact.render import image_hash

        with self.connect() as connection:
            pairs = connection.execute("SELECT id,manifest FROM pairs").fetchall()
            regressions = connection.execute("SELECT artifact_json FROM regression_case").fetchall()
        for row in pairs:
            for member in json.loads(row["manifest"])["members"]:
                if member["image_hash"] == digest:
                    path = self.root / "pairs" / row["id"] / f"{member['variant']}.png"
                    if path.is_file() and image_hash(path.read_bytes()) == digest:
                        return path
        if any(digest in json.loads(r[0]).values() for r in regressions):
            path = self.root / "regressions" / "artifacts" / f"{digest}.png"
            if path.is_file() and image_hash(path.read_bytes()) == digest:
                return path
        return None
