from __future__ import annotations

import json
import os
import re
import sqlite3
import zlib
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd


_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def _decode_bytes(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return value


def _scalarize(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_bytes(value.item())
        if value.size == 1:
            return _decode_bytes(value.reshape(-1)[0])
    return _decode_bytes(value)


def _sanitize_key(key: str) -> str:
    key = key.strip().lower()
    key = _NON_ALNUM.sub("_", key).strip("_")
    return key or "field"


def decode_raman_blob(blob: bytes) -> dict:
    payload = zlib.decompress(blob)
    return json.loads(payload)


def _raman_num_atoms_from_blob(blob: bytes) -> Optional[int]:
    if blob is None:
        return None
    try:
        data = decode_raman_blob(blob)
    except Exception:
        return None
    atoms = data.get("atoms")
    if atoms is None:
        return None
    try:
        return len(atoms)
    except TypeError:
        return None


def load_raman_db(
    path: Path,
    *,
    decode_blob: bool = True,
    expand_blob: bool = True,
    sanitize_columns: bool = True,
    columns: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """Load Raman-ChEMBL SQLite DB into a DataFrame."""
    path = Path(path)
    if columns is not None:
        cols = [str(col) for col in columns]
        if not cols:
            raise ValueError("columns must contain at least one column name.")
        col_expr = ", ".join(f'"{col.replace(chr(34), chr(34) * 2)}"' for col in cols)
        query = f"SELECT {col_expr} FROM molecule"
    else:
        query = "SELECT * FROM molecule"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
        if decode_blob and "blob_data" in df.columns:
            df = df.copy()
            df["blob_data"] = df["blob_data"].apply(
                lambda x: decode_raman_blob(x) if isinstance(x, (bytes, bytearray)) else x
            )
            if expand_blob:
                expanded = df["blob_data"].apply(pd.Series)
                if sanitize_columns:
                    expanded = expanded.rename(columns=_sanitize_key)
                df = pd.concat([df.drop(columns=["blob_data"]), expanded], axis=1)
        return df

    with sqlite3.connect(path) as con:
        if chunksize:
            frames = []
            for chunk in pd.read_sql_query(query, con, chunksize=int(chunksize)):
                frames.append(_postprocess(chunk))
            return pd.concat(frames, ignore_index=True)
        df = pd.read_sql_query(query, con)
    return _postprocess(df)


def load_raman_smiles_atoms(
    path: Path,
    *,
    limit: Optional[int] = None,
    chunksize: int = 2000,
    workers: int = 0,
) -> pd.DataFrame:
    """Load Raman SMILES and atom counts without expanding the full blob payload."""
    path = Path(path)
    query = 'SELECT id, SMILES, blob_data FROM molecule'
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    def _compute_num_atoms(series: pd.Series) -> list:
        blobs = series.tolist()
        if workers and workers > 1:
            import multiprocessing as mp

            with mp.Pool(processes=workers) as pool:
                return list(pool.imap(_raman_num_atoms_from_blob, blobs, chunksize=200))
        return [_raman_num_atoms_from_blob(b) for b in blobs]

    frames = []
    with sqlite3.connect(path) as con:
        for chunk in pd.read_sql_query(query, con, chunksize=int(chunksize)):
            chunk = chunk.rename(columns={"SMILES": "smiles"})
            chunk["num_atoms"] = _compute_num_atoms(chunk["blob_data"])
            chunk = chunk.drop(columns=["blob_data"])
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=["id", "smiles", "num_atoms"])
    return pd.concat(frames, ignore_index=True)


def iter_spice_rows(
    path: Path,
    *,
    keys: Optional[Iterable[str]] = None,
    max_groups: Optional[int] = None,
    max_confs: Optional[int] = None,
) -> Iterator[dict]:
    """Yield per-conformation rows from the SPICE HDF5 file."""
    path = Path(path)
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("h5py is required to read SPICE HDF5 files.") from exc

    with h5py.File(path, "r") as handle:
        group_iter = enumerate(handle.items())
        for group_idx, (mol_key, group) in group_iter:
            if max_groups is not None and group_idx >= max_groups:
                break
            n_conf = None
            if "conformations" in group:
                n_conf = int(group["conformations"].shape[0])
            else:
                for item in group.values():
                    if hasattr(item, "shape") and item.shape:
                        n_conf = int(item.shape[0])
                        break
            n_conf = n_conf or 1

            selected = list(group.keys()) if keys is None else list(keys)
            data = {}
            for key in selected:
                if key not in group:
                    continue
                data[key] = group[key][()]

            for conf_idx in range(n_conf):
                if max_confs is not None and conf_idx >= max_confs:
                    break
                row = {"mol_id": str(mol_key), "conf_id": int(conf_idx)}
                for key, arr in data.items():
                    arr = _scalarize(arr)
                    if isinstance(arr, np.ndarray) and arr.shape:
                        if arr.shape[0] == n_conf:
                            row[key] = _scalarize(arr[conf_idx])
                        else:
                            row[key] = arr
                    else:
                        row[key] = arr
                yield row


def load_spice_hdf5(
    path: Path,
    *,
    keys: Optional[Iterable[str]] = None,
    max_groups: Optional[int] = None,
    max_confs: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """Load SPICE HDF5 data into a DataFrame."""
    path = Path(path)
    rows = iter_spice_rows(path, keys=keys, max_groups=max_groups, max_confs=max_confs)
    if chunksize:
        frames = []
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= int(chunksize):
                frames.append(pd.DataFrame.from_records(batch))
                batch = []
        if batch:
            frames.append(pd.DataFrame.from_records(batch))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame.from_records(list(rows))


def _ensure_qm7x_cache(path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{path.stem}.hdf5"
    if target.exists():
        return target
    import lzma
    import shutil
    tmp = target.with_suffix(f".tmp.{os.getpid()}")
    with lzma.open(path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp, target)
    return target


def _open_qm7x_file(path: Path, *, cache_dir: Optional[Path] = None):
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("h5py is required to read QM7-X HDF5 files.") from exc

    if path.suffix == ".xz":
        if cache_dir is not None:
            cached = _ensure_qm7x_cache(path, Path(cache_dir))
            return h5py.File(cached, "r"), None
        import lzma
        import shutil
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        tmp.close()
        with lzma.open(path, "rb") as src, open(tmp.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return h5py.File(tmp.name, "r"), tmp.name
    return h5py.File(path, "r"), None


def _open_spice_file(path: Path, *, cache_dir: Optional[Path] = None):
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("h5py is required to read SPICE HDF5 files.") from exc

    if path.suffixes[-2:] == [".hdf5", ".gz"]:
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            target_name = path.name[:-3]  # strip .gz
            cached = cache_dir / target_name
            if not cached.exists():
                import gzip
                import shutil
                tmp = cached.with_suffix(f".tmp.{os.getpid()}")
                with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(tmp, cached)
            return h5py.File(cached, "r"), None
        import gzip
        import shutil
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        tmp.close()
        with gzip.open(path, "rb") as src, open(tmp.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return h5py.File(tmp.name, "r"), tmp.name
    return h5py.File(path, "r"), None


def iter_qm7x_rows(
    dataset_dir: Path,
    *,
    keys: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    max_mols: Optional[int] = None,
    max_confs: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    ) -> Iterator[dict]:
    """Yield per-conformation rows from QM7-X HDF5 shards (.xz or .hdf5)."""
    dataset_dir = Path(dataset_dir)
    files = sorted(
        list(dataset_dir.glob("*.hdf5")) + list(dataset_dir.glob("*.xz"))
    )
    for file_idx, path in enumerate(files):
        if max_files is not None and file_idx >= max_files:
            break
        if not path.exists():
            continue
        handle, tmp_path = _open_qm7x_file(path, cache_dir=cache_dir)
        try:
            mol_iter = enumerate(handle.items())
            for mol_idx, (mol_key, mol_group) in mol_iter:
                if max_mols is not None and mol_idx >= max_mols:
                    break
                conf_iter = enumerate(mol_group.items())
                for conf_idx, (conf_key, conf) in conf_iter:
                    if max_confs is not None and conf_idx >= max_confs:
                        break
                    row = {"mol_id": str(mol_key), "conf_id": str(conf_key)}
                    selected = list(conf.keys()) if keys is None else list(keys)
                    for key in selected:
                        if key not in conf:
                            if key == "smiles":
                                row["smiles"] = None
                            continue
                        row[key] = _scalarize(conf[key][()])
                    yield row
        finally:
            handle.close()
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def load_qm7x(
    dataset_dir: Path,
    *,
    keys: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    max_mols: Optional[int] = None,
    max_confs: Optional[int] = None,
    chunksize: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load QM7-X into a DataFrame."""
    dataset_dir = Path(dataset_dir)
    rows = iter_qm7x_rows(
        dataset_dir,
        keys=keys,
        max_files=max_files,
        max_mols=max_mols,
        max_confs=max_confs,
        cache_dir=cache_dir,
    )
    if chunksize:
        frames = []
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= int(chunksize):
                frames.append(pd.DataFrame.from_records(batch))
                batch = []
        if batch:
            frames.append(pd.DataFrame.from_records(batch))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame.from_records(list(rows))


def iter_qm7x_atom_counts(
    dataset_dir: Path,
    *,
    max_files: Optional[int] = None,
    max_mols: Optional[int] = None,
    max_confs: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Iterator[dict]:
    """Yield rows with atom counts only (fast, avoids loading full arrays)."""
    dataset_dir = Path(dataset_dir)
    files = sorted(
        list(dataset_dir.glob("*.hdf5")) + list(dataset_dir.glob("*.xz"))
    )
    for file_idx, path in enumerate(files):
        if max_files is not None and file_idx >= max_files:
            break
        if not path.exists():
            continue
        handle, tmp_path = _open_qm7x_file(path, cache_dir=cache_dir)
        try:
            mol_iter = enumerate(handle.items())
            for mol_idx, (mol_key, mol_group) in mol_iter:
                if max_mols is not None and mol_idx >= max_mols:
                    break
                conf_iter = enumerate(mol_group.items())
                for conf_idx, (conf_key, conf) in conf_iter:
                    if max_confs is not None and conf_idx >= max_confs:
                        break
                    if "atNUM" not in conf:
                        continue
                    try:
                        num_atoms = int(conf["atNUM"].shape[0])
                    except Exception:
                        num_atoms = None
                    yield {
                        "mol_id": str(mol_key),
                        "conf_id": str(conf_key),
                        "num_atoms": num_atoms,
                        "source_file": path.name,
                    }
        finally:
            handle.close()
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def load_qm7x_atom_counts(
    dataset_dir: Path,
    *,
    max_files: Optional[int] = None,
    max_mols: Optional[int] = None,
    max_confs: Optional[int] = None,
    chunksize: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    workers: int = 0,
) -> pd.DataFrame:
    """Load QM7-X atom counts without materializing full coordinate/atom arrays."""
    if workers and workers > 1:
        files = sorted(
            list(Path(dataset_dir).glob("*.hdf5")) + list(Path(dataset_dir).glob("*.xz"))
        )
        if max_files is not None:
            files = files[:max_files]
        args = [
            (path, cache_dir, max_mols, max_confs)
            for path in files
            if path.exists()
        ]
        if not args:
            return pd.DataFrame()
        import multiprocessing as mp

        with mp.Pool(processes=workers) as pool:
            rows = pool.map(_qm7x_atom_counts_for_file, args)
        flat = [row for part in rows for row in part]
        return pd.DataFrame.from_records(flat)

    rows = iter_qm7x_atom_counts(
        dataset_dir,
        max_files=max_files,
        max_mols=max_mols,
        max_confs=max_confs,
        cache_dir=cache_dir,
    )
    if chunksize:
        frames = []
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= int(chunksize):
                frames.append(pd.DataFrame.from_records(batch))
                batch = []
        if batch:
            frames.append(pd.DataFrame.from_records(batch))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame.from_records(list(rows))


def _qm7x_atom_counts_for_file(args) -> list[dict]:
    path, cache_dir, max_mols, max_confs = args
    rows = []
    handle, tmp_path = _open_qm7x_file(Path(path), cache_dir=cache_dir)
    try:
        mol_iter = enumerate(handle.items())
        for mol_idx, (mol_key, mol_group) in mol_iter:
            if max_mols is not None and mol_idx >= max_mols:
                break
            conf_iter = enumerate(mol_group.items())
            for conf_idx, (conf_key, conf) in conf_iter:
                if max_confs is not None and conf_idx >= max_confs:
                    break
                if "atNUM" not in conf:
                    continue
                try:
                    num_atoms = int(conf["atNUM"].shape[0])
                except Exception:
                    num_atoms = None
                rows.append(
                    {
                        "mol_id": str(mol_key),
                        "conf_id": str(conf_key),
                        "num_atoms": num_atoms,
                        "source_file": Path(path).name,
                    }
                )
    finally:
        handle.close()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return rows


def _spice_num_atoms(group) -> Optional[int]:
    if "atomic_numbers" in group:
        try:
            return int(group["atomic_numbers"].shape[0])
        except Exception:
            return None
    if "conformations" in group:
        try:
            return int(group["conformations"].shape[1])
        except Exception:
            return None
    if "atoms" in group:
        try:
            return int(len(group["atoms"]))
        except Exception:
            return None
    return None


def _spice_smiles(group, mol_key: str) -> Optional[str]:
    if "smiles" in group:
        raw = group["smiles"][()]
        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                return None
            return _scalarize(raw.reshape(-1)[0])
        return _scalarize(raw)
    return str(mol_key)


def _spice_subset_name(path: Path) -> str:
    parent = path.parent.name
    name = path.name.lower()
    if parent == "pubchem" and name.startswith("solvated-pubchem"):
        return "solvated-pubchem"
    return parent


def _spice_atom_counts_for_file(args) -> list[dict]:
    path, cache_dir, max_mols = args
    rows = []
    handle, tmp_path = _open_spice_file(Path(path), cache_dir=cache_dir)
    subset = _spice_subset_name(Path(path))
    try:
        mol_iter = enumerate(handle.items())
        for mol_idx, (mol_key, group) in mol_iter:
            if max_mols is not None and mol_idx >= max_mols:
                break
            num_atoms = _spice_num_atoms(group)
            smiles = _spice_smiles(group, str(mol_key))
            rows.append(
                {
                    "mol_id": str(mol_key),
                    "smiles": smiles,
                    "num_atoms": num_atoms,
                    "subset": subset,
                    "source_file": Path(path).name,
                }
            )
    finally:
        handle.close()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return rows


def iter_spice_dir_atom_counts(
    dataset_dir: Path,
    *,
    max_files: Optional[int] = None,
    max_mols: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Iterator[dict]:
    dataset_dir = Path(dataset_dir)
    files = sorted(
        list(dataset_dir.rglob("*.hdf5")) + list(dataset_dir.rglob("*.hdf5.gz"))
    )
    for file_idx, path in enumerate(files):
        if max_files is not None and file_idx >= max_files:
            break
        if "_cache" in path.parts:
            continue
        handle, tmp_path = _open_spice_file(path, cache_dir=cache_dir)
        subset = _spice_subset_name(path)
        try:
            mol_iter = enumerate(handle.items())
            for mol_idx, (mol_key, group) in mol_iter:
                if max_mols is not None and mol_idx >= max_mols:
                    break
                num_atoms = _spice_num_atoms(group)
                smiles = _spice_smiles(group, str(mol_key))
                yield {
                    "mol_id": str(mol_key),
                    "smiles": smiles,
                    "num_atoms": num_atoms,
                    "subset": subset,
                    "source_file": path.name,
                }
        finally:
            handle.close()
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def load_spice_dir_atom_counts(
    dataset_dir: Path,
    *,
    max_files: Optional[int] = None,
    max_mols: Optional[int] = None,
    chunksize: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    workers: int = 0,
) -> pd.DataFrame:
    if workers and workers > 1:
        files = sorted(
            list(Path(dataset_dir).rglob("*.hdf5")) + list(Path(dataset_dir).rglob("*.hdf5.gz"))
        )
        files = [p for p in files if "_cache" not in p.parts]
        if max_files is not None:
            files = files[:max_files]
        args = [(path, cache_dir, max_mols) for path in files]
        if not args:
            return pd.DataFrame()
        import multiprocessing as mp

        with mp.Pool(processes=workers) as pool:
            rows = pool.map(_spice_atom_counts_for_file, args)
        flat = [row for part in rows for row in part]
        return pd.DataFrame.from_records(flat)

    rows = iter_spice_dir_atom_counts(
        dataset_dir,
        max_files=max_files,
        max_mols=max_mols,
        cache_dir=cache_dir,
    )
    if chunksize:
        frames = []
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= int(chunksize):
                frames.append(pd.DataFrame.from_records(batch))
                batch = []
        if batch:
            frames.append(pd.DataFrame.from_records(batch))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame.from_records(list(rows))
