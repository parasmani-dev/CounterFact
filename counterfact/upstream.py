"""Chartographer export helpers adapted from Apache-2.0 source.

Upstream: compling-wat/Chartographer, commit 046a622103733986db6e6b1b821cbc546b779192
Source: src/pipeline/datasets/export_chart_question_families.py
Changes: extract only dependency-free family_member_row and write_jsonl helpers;
modernize type annotations. Used by CounterFact's pair export, not its renderer.
See NOTICE and LICENSE.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def family_member_row(
    *,
    base_row: dict[str, Any],
    member_source: dict[str, Any],
    variant: str,
    chart_id: str,
    question_id: str,
    question_col: str,
    answer_col: str,
    image_rel: str,
    chart_data_rel: str,
    source_dataset_split: str,
    source_row_index: int,
) -> dict[str, Any]:
    return {
        "chart_id": chart_id,
        "question_id": question_id,
        "variant": variant,
        "image": image_rel,
        "question": str(member_source.get(question_col, base_row.get(question_col, ""))),
        "answer": str(member_source.get(answer_col, base_row.get(answer_col, ""))),
        "chart_data": chart_data_rel,
        "source_dataset_split": source_dataset_split,
        "source_row_index": source_row_index,
    }
