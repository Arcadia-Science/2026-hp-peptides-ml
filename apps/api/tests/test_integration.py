from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.getenv("INTEGRATION_BASE_URL", "http://localhost:8000")
DATASET = os.getenv("INTEGRATION_DATASET", "ext_val")


@pytest.fixture(scope="session")
def client():
    try:
        client = httpx.Client(base_url=BASE_URL, timeout=120.0)
        response = client.get("/healthz")
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"integration API not available at {BASE_URL}: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def datapoint(client):
    response = client.get("/datapoints", params={"dataset": DATASET, "limit": 1})
    response.raise_for_status()
    payload = response.json()
    datapoints = payload.get("datapoints", [])
    if not datapoints:
        pytest.skip(f"no datapoints returned for dataset '{DATASET}'")
    return datapoints[0]


@pytest.fixture(scope="session")
def geometry_payload(client, datapoint):
    response = client.get(
        f"/datapoints/{DATASET}/{datapoint['molecule_id']}",
        params={"include_geometry": "true"},
    )
    response.raise_for_status()
    payload = response.json()
    return {"pos": payload["pos"], "z": payload["z"]}


@pytest.mark.integration
def test_datasets_includes_dataset(client):
    response = client.get("/datasets")
    response.raise_for_status()
    datasets = [item["name"] for item in response.json().get("datasets", [])]
    assert DATASET in datasets


@pytest.mark.integration
def test_predict_vib(client, geometry_payload):
    response = client.post("/predict/vib", json=geometry_payload)
    response.raise_for_status()
    payload = response.json()
    assert len(payload.get("freq", [])) > 0
    assert len(payload.get("freq", [])) == len(payload.get("ir_intensity", []))
    assert len(payload.get("freq", [])) == len(payload.get("raman_activity", []))


@pytest.mark.integration
def test_predict_raman(client, geometry_payload):
    response = client.post("/predict/raman", json=geometry_payload)
    response.raise_for_status()
    payload = response.json()
    assert len(payload.get("x", [])) == len(payload.get("y", []))
    assert len(payload.get("x", [])) > 0
    assert len(payload.get("png_base64", "")) > 0


@pytest.mark.integration
def test_predict_uv(client, geometry_payload):
    response = client.post("/predict/uv", json=geometry_payload)
    response.raise_for_status()
    payload = response.json()
    assert len(payload.get("uv", [])) > 0
