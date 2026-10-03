"""Fast API contract tests for the NOVA backend."""
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health_reports_native_state():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["job_schema"] == "0.9"
    assert data["native_profile"] == "creator-16f-base"
    assert data["native_profile"] == "creator-16f-base"
    assert data["version"] == "0.9.1"
    assert "native_ready" in data


def test_job_contract_accepts_director_and_sampler():
    response = client.post(
        "/v1/jobs",
        json={
            "prompt": "A red circle moving through a blue room",
            "project_name": "Contract test",
            "sampler": "ddim",
            "sampling_steps": 20,
            "director": {
                "subject": "red circle",
                "action": "moves forward",
                "environment": "blue room",
                "camera_movement": "dolly in",
                "lighting": "soft",
                "style": "cinematic",
                "motion": "natural",
            },
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["request"]["sampler"] == "ddim"
    assert data["request"]["sampling_steps"] == 20
    assert data["request"]["frames"] == 16
    assert data["request"]["height"] == 64
    assert data["request"]["width"] == 64
    assert data["progress"] == 5
    assert data["stage"] == "queued"
    assert data["request"]["continuity"]["enabled"] is True


def test_invalid_sampler_is_rejected():
    response = client.post(
        "/v1/jobs",
        json={"prompt": "test", "sampler": "invalid"},
    )
    assert response.status_code == 422


def test_preflight_reports_continuity_and_native_shape_contract():
    response = client.post(
        "/v1/preflight",
        json={
            "prompt": "A cinematic character walking forward",
            "continuity": {
                "enabled": True,
                "continuity_id": "demo-world",
                "shot_index": 2,
                "context": "Same wardrobe and location as shot 1",
                "locked_elements": ["character", "wardrobe", "location"],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert isinstance(data["warnings"], list)
    assert data["errors"] == []
