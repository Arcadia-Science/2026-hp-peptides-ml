import json

import redis.asyncio as redis

from app.config import settings

_client: redis.Redis | None = None


async def init_cache() -> None:
    global _client
    _client = redis.from_url(settings.redis_url, decode_responses=True)


async def close_cache() -> None:
    if _client is not None:
        await _client.close()


async def get_json(key: str):
    if _client is None:
        return None
    raw = await _client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def set_json(key: str, value, ttl_seconds: int) -> None:
    if _client is None:
        return
    await _client.set(key, json.dumps(value), ex=ttl_seconds)
