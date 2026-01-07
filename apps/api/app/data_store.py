from pathlib import Path

import pyarrow.parquet as pq


def read_geometry(parquet_dir: str, source_path: str, source_row: int) -> tuple[list[list[float]], list[int]]:
    parquet_path = Path(parquet_dir) / source_path
    pf = pq.ParquetFile(parquet_path)

    row_offset = source_row
    for rg_index in range(pf.num_row_groups):
        num_rows = pf.metadata.row_group(rg_index).num_rows
        if row_offset < num_rows:
            table = pf.read_row_group(rg_index, columns=["pos", "z"])
            row = table.slice(row_offset, 1)
            pos = row.column("pos")[0].as_py()
            z = row.column("z")[0].as_py()
            return pos, z
        row_offset -= num_rows

    raise IndexError(f"Row {source_row} out of range for {parquet_path}")
