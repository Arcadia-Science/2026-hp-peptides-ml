from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from app import cache, db, main


class DummyModelClient:
    async def close(self) -> None:
        return None

    async def post(self, path: str, payload: dict) -> dict:
        if path == "/predict/nmr":
            return {"sc": [1.0], "sh": [2.0]}
        if path == "/predict/charge":
            return {"charge": [0.1]}
        return {"ok": True}


@pytest.fixture()
def client(monkeypatch):
    async def noop() -> None:
        return None

    monkeypatch.setattr(db, "init_db", noop)
    monkeypatch.setattr(db, "close_db", noop)
    monkeypatch.setattr(cache, "init_cache", noop)
    monkeypatch.setattr(cache, "close_cache", noop)
    monkeypatch.setattr(main, "ModelClient", lambda: DummyModelClient())

    with TestClient(main.app) as client:
        yield client
