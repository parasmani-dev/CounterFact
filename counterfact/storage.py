"""SQLite stores only generated input pairs in Phase 1, never model evidence."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from counterfact.reference import answer, question_text
from counterfact.render import RENDERER_VERSION, BarRenderer, image_hash
from counterfact.schemas import PairSpec
from counterfact.upstream import family_member_row, write_jsonl


class PairStore:
    def __init__(self, data_dir: Path):
        self.root = data_dir
        self.database = data_dir / "counterfact.db"

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, 4):
                raise RuntimeError("Unsupported database schema version")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pairs ("
                "id TEXT PRIMARY KEY, spec TEXT NOT NULL, manifest TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))"
            )
            if version == 0:
                connection.execute("PRAGMA user_version=1")

    def check(self) -> None:
        with self.connect() as connection:
            connection.execute("SELECT COUNT(*) FROM pairs").fetchone()

    def get(self, pair_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT manifest FROM pairs WHERE id=?", (pair_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def create(self, spec: PairSpec) -> dict:
        spec = PairSpec.model_validate(spec.model_dump())
        serialized = json.dumps(spec.model_dump(), sort_keys=True, separators=(",", ":"))
        pair_id = hashlib.sha256((RENDERER_VERSION + serialized).encode()).hexdigest()
        existing = self.get(pair_id)
        if existing is not None:
            return existing
        target = self.root / "pairs" / pair_id
        target.mkdir(parents=True, exist_ok=True)
        text = question_text(spec.chart, spec.question)
        members, rows = [], []
        renderer = BarRenderer()
        for variant, chart in (("original", spec.chart), ("transformed", spec.transformed())):
            png = renderer.render(chart, show_value_labels=spec.show_value_labels)
            expected = answer(chart, spec.question)
            (target / f"{variant}.png").write_bytes(png)
            (target / f"{variant}.json").write_text(
                chart.model_dump_json(indent=2), encoding="utf-8"
            )
            members.append(
                {
                    "variant": variant,
                    "image_hash": image_hash(png),
                    "image_url": f"/api/pairs/{pair_id}/{variant}.png",
                    "expected_answer": expected,
                }
            )
            rows.append(
                family_member_row(
                    base_row={},
                    member_source={"question": text, "answer": expected},
                    variant=variant,
                    chart_id=spec.id,
                    question_id=f"{spec.id}-q1",
                    question_col="question",
                    answer_col="answer",
                    image_rel=f"{variant}.png",
                    chart_data_rel=f"{variant}.json",
                    source_dataset_split="counterfact_synthetic",
                    source_row_index=0,
                )
            )
        manifest = {
            "id": pair_id,
            "kind": "synthetic_input_pair",
            "renderer_version": RENDERER_VERSION,
            "spec": spec.model_dump(),
            "question": text,
            "expected_relation": (
                "same"
                if members[0]["expected_answer"] == members[1]["expected_answer"]
                else "different"
            ),
            "members": members,
            "model_evaluated": False,
        }
        write_jsonl(target / "chartographer-family.jsonl", rows)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO pairs(id,spec,manifest) VALUES(?,?,?)",
                (pair_id, serialized, json.dumps(manifest)),
            )
        return manifest
