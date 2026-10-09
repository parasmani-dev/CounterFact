import io
import json
import sqlite3

import matplotlib
import pytest
from PIL import Image
from pydantic import ValidationError

from counterfact.render import BarRenderer, image_hash
from counterfact.schemas import PairSpec
from counterfact.storage import PairStore


def test_png_is_repeatable_and_transformed_pixels_differ(pair_payload):
    spec = PairSpec.model_validate(pair_payload)
    renderer = BarRenderer()
    original = renderer.render(spec.chart)
    repeated = renderer.render(spec.chart)
    transformed = renderer.render(spec.transformed())
    assert original == repeated
    assert image_hash(original) != image_hash(transformed)
    assert Image.open(io.BytesIO(original)).size == (1000, 600)
    assert matplotlib.get_backend().casefold() == "agg"


def test_value_label_flag_changes_png_and_pair_identity(pair_payload, tmp_path):
    hidden_payload = json.loads(json.dumps(pair_payload))
    hidden_payload["show_value_labels"] = False
    visible_payload = json.loads(json.dumps(pair_payload))
    visible_payload["show_value_labels"] = True
    hidden = PairSpec.model_validate(hidden_payload)
    visible = PairSpec.model_validate(visible_payload)
    assert image_hash(BarRenderer().render(hidden.chart, show_value_labels=False)) != image_hash(
        BarRenderer().render(visible.chart, show_value_labels=True)
    )
    store = PairStore(tmp_path)
    store.initialize()
    assert store.create(hidden)["id"] != store.create(visible)["id"]


def test_label_visibility_defaults_follow_question_type(pair_payload):
    assert PairSpec.model_validate(pair_payload).show_value_labels is False
    pair_payload["question"] = {"type": "value_lookup", "target_id": "lab_a"}
    pair_payload["mutations"] = [{"type": "reorder", "order": ["lab_c", "lab_a", "lab_b"]}]
    assert PairSpec.model_validate(pair_payload).show_value_labels is True
    pair_payload["show_value_labels"] = False
    with pytest.raises(ValidationError, match="value_lookup requires"):
        PairSpec.model_validate(pair_payload)


def test_label_is_literal_text_not_latex(pair_payload):
    pair_payload["chart"]["categories"][0]["label"] = "$not_latex{"
    png = BarRenderer().render(PairSpec.model_validate(pair_payload).chart)
    assert png.startswith(b"\x89PNG")


def test_store_is_idempotent_and_survives_reopen(tmp_path, pair_payload):
    store = PairStore(tmp_path)
    store.initialize()
    spec = PairSpec.model_validate(pair_payload)
    manifest = store.create(spec)
    assert store.create(spec) == manifest
    reopened = PairStore(tmp_path)
    reopened.initialize()
    assert reopened.get(manifest["id"]) == manifest
    with sqlite3.connect(store.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 1
    assert manifest["kind"] == "synthetic_input_pair"
    assert manifest["model_evaluated"] is False
    assert "prediction" not in json.dumps(manifest)


def test_chartographer_family_export_contains_inputs_only(tmp_path, pair_payload):
    store = PairStore(tmp_path)
    store.initialize()
    result = store.create(PairSpec.model_validate(pair_payload))
    folder = tmp_path / "pairs" / result["id"]
    rows = [
        json.loads(line)
        for line in (folder / "chartographer-family.jsonl").read_text().splitlines()
    ]
    assert [row["answer"] for row in rows] == ["Lab B", "Lab A"]
    for row in rows:
        assert (folder / row["image"]).is_file()
        assert (folder / row["chart_data"]).is_file()
        assert row["source_dataset_split"] == "counterfact_synthetic"


def test_schema_version_mismatch_fails_explicitly(tmp_path):
    store = PairStore(tmp_path)
    store.initialize()
    with sqlite3.connect(store.database) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="schema version"):
        store.initialize()
