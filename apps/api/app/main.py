import httpx
from fastapi import FastAPI, HTTPException, Query

from app import cache, db
from app.cache import get_json, set_json
from app.config import settings
from app.data_store import read_geometry
from app.inference_client import ModelClient
from app.models import GeometryRequest, NmrAggregateRequest

app = FastAPI(title="DetaNet API", version="0.1.0")


@app.on_event("startup")
async def startup() -> None:
    await db.init_db()
    await cache.init_cache()
    app.state.model_client = ModelClient()


@app.on_event("shutdown")
async def shutdown() -> None:
    await db.close_db()
    await cache.close_cache()
    await app.state.model_client.close()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/datasets")
async def list_datasets() -> dict:
    rows = await db.fetch("SELECT name, description, source_uri FROM datasets ORDER BY name")
    return {"datasets": [dict(row) for row in rows]}


@app.get("/datapoints")
async def list_datapoints(
    dataset: str = Query(...),
    smiles: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    if smiles:
        rows = await db.fetch(
            """
            SELECT dataset, molecule_id, smiles, n_atoms, environment
            FROM molecules
            WHERE dataset=$1 AND smiles=$2
            ORDER BY molecule_id
            LIMIT $3 OFFSET $4
            """,
            dataset,
            smiles,
            limit,
            offset,
        )
    else:
        rows = await db.fetch(
            """
            SELECT dataset, molecule_id, smiles, n_atoms, environment
            FROM molecules
            WHERE dataset=$1
            ORDER BY molecule_id
            LIMIT $2 OFFSET $3
            """,
            dataset,
            limit,
            offset,
        )
    return {"datapoints": [dict(row) for row in rows]}


@app.get("/datapoints/{dataset}/{molecule_id}")
async def get_datapoint(dataset: str, molecule_id: int, include_geometry: bool = False) -> dict:
    cache_key = f"dp:{dataset}:{molecule_id}:{int(include_geometry)}"
    cached = await get_json(cache_key)
    if cached:
        return cached

    row = await db.fetchrow(
        """
        SELECT dataset, molecule_id, qm9_id, smiles, n_atoms, environment, source_path, source_row
        FROM molecules
        WHERE dataset=$1 AND molecule_id=$2
        """,
        dataset,
        molecule_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    payload = dict(row)
    if include_geometry:
        pos, z = read_geometry(settings.parquet_dir, row["source_path"], row["source_row"])
        payload["pos"] = pos
        payload["z"] = z

    await set_json(cache_key, payload, settings.cache_ttl_seconds)
    return payload


async def resolve_geometry(request: GeometryRequest) -> tuple[list[list[float]], list[int]]:
    if request.pos is not None and request.z is not None:
        return request.pos, request.z

    if request.dataset is None or request.molecule_id is None:
        raise HTTPException(status_code=400, detail="Missing dataset or molecule_id")

    row = await db.fetchrow(
        """
        SELECT source_path, source_row
        FROM molecules
        WHERE dataset=$1 AND molecule_id=$2
        """,
        request.dataset,
        request.molecule_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    return read_geometry(settings.parquet_dir, row["source_path"], row["source_row"])


def make_cache_key(prefix: str, request: GeometryRequest) -> str | None:
    if request.dataset is None or request.molecule_id is None:
        return None
    return f"pred:{prefix}:{request.dataset}:{request.molecule_id}"


async def predict(path: str, request: GeometryRequest, cache_prefix: str) -> dict:
    cache_key = make_cache_key(cache_prefix, request)
    if cache_key:
        cached = await get_json(cache_key)
        if cached:
            return cached

    pos, z = await resolve_geometry(request)
    try:
        result = await app.state.model_client.post(path, {"pos": pos, "z": z})
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Model request failed"
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Model service unavailable") from exc

    if cache_key:
        await set_json(cache_key, result, settings.cache_ttl_seconds)
    return result


@app.post("/predict/charge")
async def predict_charge(request: GeometryRequest) -> dict:
    return await predict("/predict/charge", request, "charge")


@app.post("/predict/vib")
async def predict_vib(request: GeometryRequest) -> dict:
    return await predict("/predict/vib", request, "vib")


@app.post("/predict/raman")
async def predict_raman(request: GeometryRequest) -> dict:
    return await predict("/predict/raman", request, "raman")


@app.post("/predict/uv")
async def predict_uv(request: GeometryRequest) -> dict:
    return await predict("/predict/uv", request, "uv")


@app.post("/predict/nmr")
async def predict_nmr(request: GeometryRequest) -> dict:
    result = await predict("/predict/nmr", request, "nmr")
    result["message"] = "Provide indexc/indexh to /predict/nmr/aggregate for environment aggregation."
    return result


@app.post("/predict/nmr/aggregate")
async def predict_nmr_aggregate(request: NmrAggregateRequest) -> dict:
    try:
        return await app.state.model_client.post("/predict/nmr/aggregate", request.model_dump())
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Model request failed"
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Model service unavailable") from exc
