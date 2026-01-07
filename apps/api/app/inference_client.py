import httpx

from app.config import settings


class ModelClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=settings.model_url, timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def post(self, path: str, payload: dict) -> dict:
        response = await self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
