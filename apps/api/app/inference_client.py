import asyncio

import httpx

from app.config import settings


class ModelClient:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.model_max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=settings.model_url,
            timeout=settings.model_timeout_seconds,
            limits=httpx.Limits(
                max_connections=settings.model_max_connections,
                max_keepalive_connections=settings.model_max_keepalive_connections,
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def post(self, path: str, payload: dict) -> dict:
        delay = settings.model_retry_base_delay_seconds
        for attempt in range(1, settings.model_retry_attempts + 1):
            try:
                async with self._semaphore:
                    response = await self._client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ):
                if attempt == settings.model_retry_attempts:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, settings.model_retry_max_delay_seconds)
