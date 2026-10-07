"""Stack an h5ad of raw counts with a thinned copy of itself, for a scaling point beyond
the largest public file at hand.

    .venv/bin/python benches/double_counts.py data/embryo_1m_counts.h5ad data/embryo_2m_counts.h5ad

The copy keeps every cell's annotation and draws each count from Binomial(count, 0.85),
so the second half has the same structure at a slightly lower depth and is not a
byte-for-byte duplicate. The file is written block by block in CSR, never held whole.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from anndata.io import read_elem, write_elem


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    rng = np.random.default_rng(0)
    with h5py.File(source) as src, h5py.File(target, "w") as dst:
        n_cells, n_genes = (int(v) for v in src["X"].attrs["shape"])
        indptr = src["X/indptr"][...].astype(np.int64)
        nnz = int(indptr[-1])
        x = dst.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.array([2 * n_cells, n_genes], dtype=np.int64)
        data = x.create_dataset(
            "data", shape=(2 * nnz,), maxshape=(None,), dtype=np.float32, chunks=(1 << 20,)
        )
        indices = x.create_dataset(
            "indices", shape=(2 * nnz,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,)
        )
        out_indptr = np.empty(2 * n_cells + 1, dtype=np.int64)
        block = 50_000
        written = 0
        for copy in range(2):
            for start in range(0, n_cells, block):
                end = min(start + block, n_cells)
                lo, hi = int(indptr[start]), int(indptr[end])
                values = src["X/data"][lo:hi]
                cols = src["X/indices"][lo:hi]
                if copy == 1:
                    values = rng.binomial(values.astype(np.int64), 0.85).astype(np.float32)
                    keep = values > 0
                    # Row offsets after dropping the zeros.
                    rows = np.repeat(np.arange(start, end), np.diff(indptr[start : end + 1]))
                    rows, values, cols = rows[keep], values[keep], cols[keep]
                    counts = np.bincount(rows - start, minlength=end - start)
                else:
                    counts = np.diff(indptr[start : end + 1])
                data[written : written + len(values)] = values
                indices[written : written + len(values)] = cols
                row0 = copy * n_cells + start
                out_indptr[row0 + 1 : row0 + 1 + (end - start)] = written + np.cumsum(counts)
                written += len(values)
                print(f"copy {copy} rows {end}/{n_cells}", flush=True)
        out_indptr[0] = 0
        data.resize((written,))
        indices.resize((written,))
        x.create_dataset("indptr", data=out_indptr)
        obs = read_elem(src["obs"])
        doubled = obs.copy()
        doubled.index = [f"{i}-2" for i in obs.index]
        import pandas as pd

        write_elem(dst, "obs", pd.concat([obs, doubled]))
        write_elem(dst, "var", read_elem(src["var"]))
        print(f"wrote {target}: {2 * n_cells} cells, {written} stored values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
