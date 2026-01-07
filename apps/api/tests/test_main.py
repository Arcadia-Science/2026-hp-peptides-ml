from __future__ import annotations

from app import db


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_datasets(client, monkeypatch):
    async def fake_fetch(query: str, *args):
        return [
            {
                "name": "qm9s",
                "description": "QM9 subset",
                "source_uri": "s3://example/qm9s",
            }
        ]

    monkeypatch.setattr(db, "fetch", fake_fetch)
    response = client.get("/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["datasets"][0]["name"] == "qm9s"


def test_list_datapoints(client, monkeypatch):
    async def fake_fetch(query: str, *args):
        return [
            {
                "dataset": "qm9s",
                "molecule_id": 1,
                "smiles": "C",
                "n_atoms": 1,
                "environment": None,
            }
        ]

    monkeypatch.setattr(db, "fetch", fake_fetch)
    response = client.get("/datapoints", params={"dataset": "qm9s"})
    assert response.status_code == 200
    data = response.json()
    assert data["datapoints"][0]["molecule_id"] == 1


def test_get_datapoint_not_found(client, monkeypatch):
    async def fake_fetchrow(query: str, *args):
        return None

    monkeypatch.setattr(db, "fetchrow", fake_fetchrow)
    response = client.get("/datapoints/qm9s/999")
    assert response.status_code == 404


def test_predict_nmr_adds_message(client):
    payload = {"pos": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "z": [6, 1]}
    response = client.post("/predict/nmr", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sc"] == [1.0]
    assert data["sh"] == [2.0]
    assert "message" in data
