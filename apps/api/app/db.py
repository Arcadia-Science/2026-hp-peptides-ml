import asyncpg

from app.config import settings

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    _pool = await asyncpg.create_pool(dsn=settings.database_url, min_size=1, max_size=10)


async def close_db() -> None:
    if _pool:
        await _pool.close()


async def fetch(query: str, *args):
    if not _pool:
        raise RuntimeError("Database pool is not initialized")
    async with _pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args):
    if not _pool:
        raise RuntimeError("Database pool is not initialized")
    async with _pool.acquire() as conn:
        return await conn.fetchrow(query, *args)
