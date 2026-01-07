import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    model_url: str
    parquet_dir: str
    cache_ttl_seconds: int


settings = Settings(
    database_url=os.getenv("DATABASE_URL", "postgresql://detanet:detanet@localhost:5432/detanet"),
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    model_url=os.getenv("MODEL_URL", "http://localhost:8001"),
    parquet_dir=os.getenv("PARQUET_DIR", "/data/processed"),
    cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
)
