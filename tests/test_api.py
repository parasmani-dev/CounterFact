import sqlite3

from fastapi.testclient import TestClient

from counterfact.api import create_app
from counterfact.config import Settings


def test_health_create_fetch_and_image(tmp_path, pair_payload):
    with TestClient(create_app(Settings(tmp_path))) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["inference_ready"] is False
        assert health.headers["x-request-id"]
        created = client.post("/api/pairs", json=pair_payload)
        assert created.status_code == 201
        pair = created.json()
        assert pair["expected_relation"] == "different"
        assert client.get(f"/api/pairs/{pair['id']}").json() == pair
        for member in pair["members"]:
            image = client.get(member["image_url"])
            assert image.status_code == 200
            assert image.headers["content-type"] == "image/png"


def test_invalid_payload_never_echoes_secrets(tmp_path, pair_payload):
    pair_payload["api_key"] = "test-only-sensitive-value"
    with TestClient(create_app(Settings(tmp_path))) as client:
        response = client.post("/api/pairs", json=pair_payload)
        assert response.status_code == 422
        assert "test-only-sensitive-value" not in response.text
        assert response.json()["error"]["code"] == "invalid_input"


def test_missing_pairs_and_invalid_image_variants(tmp_path, pair_payload):
    with TestClient(create_app(Settings(tmp_path))) as client:
        assert client.get("/api/pairs/not-a-hash").status_code == 404
        assert client.get("/api/pairs/" + "a" * 64).status_code == 404
        pair = client.post("/api/pairs", json=pair_payload).json()
        assert client.get(f"/api/pairs/{pair['id']}/secret.png").status_code == 422


def test_storage_failure_is_safe_503(tmp_path, monkeypatch):
    with TestClient(create_app(Settings(tmp_path))) as client:

        def broken():
            raise sqlite3.OperationalError("database path and details must not leak")

        monkeypatch.setattr(client.app.state.store, "check", broken)
        response = client.get("/api/health")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "storage_unavailable"
        assert "must not leak" not in response.text
