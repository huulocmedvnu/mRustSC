#!/usr/bin/env python3
"""Turn a CELLxGENE `.h5ad` into a counts-only file the streaming pipeline can read.

    .venv/bin/python benches/prepare_counts.py in.h5ad out.h5ad [--cells N]

CELLxGENE files carry the normalised matrix in `X` and the raw counts in `raw.X`, over
the full feature set. The streaming path reads `X` in row blocks, so this copies
`raw.X` into `X` of a new file block by block (never holding the matrix whole), keeps
the `cell_type` and `donor_id` columns of `obs`, and names the genes by `feature_name`.
`--cells N` keeps the first N rows, for a quick check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

BLOCK = 50_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--cells", type=int)
    args = parser.parse_args()

    with h5py.File(args.source, "r") as src:
        obs = read_elem(src["obs"])
        raw = src.get("raw", src)
        var = read_elem(raw["var"])
        x = raw["X"]
        n_obs, n_vars = (int(v) for v in x.attrs["shape"])
        if args.cells:
            n_obs = min(n_obs, args.cells)
        indptr = x["indptr"][: n_obs + 1].astype(np.int64)
        nnz = int(indptr[-1])

        keep_obs = [c for c in ("cell_type", "author_celltype", "donor_id", "assay") if c in obs]
        obs = obs.iloc[:n_obs][keep_obs].copy()
        for c in obs.columns:
            obs[c] = obs[c].astype(str)
        names = (
            var["feature_name"].astype(str).to_numpy()
            if "feature_name" in var
            else var.index.to_numpy()
        )
        var_out = pd.DataFrame(index=pd.Index(names, dtype=str))
        var_out.index = anndata.utils.make_index_unique(var_out.index)

        skeleton = anndata.AnnData(obs=obs, var=var_out)
        skeleton.write_h5ad(args.target)

        with h5py.File(args.target, "r+") as dst:
            if "X" in dst:
                del dst["X"]
            g = dst.create_group("X")
            g.attrs["encoding-type"] = "csr_matrix"
            g.attrs["encoding-version"] = "0.1.0"
            g.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)
            # scipy wants indptr and indices in one dtype; int32 covers up to 2^31 stored values.
            g.create_dataset("indptr", data=indptr.astype(np.int32 if nnz < 2**31 else np.int64))
            data = g.create_dataset(
                "data", shape=(nnz,), dtype=np.float32, chunks=(min(nnz, 1 << 20),)
            )
            indices = g.create_dataset(
                "indices",
                shape=(nnz,),
                dtype=np.int32 if nnz < 2**31 else np.int64,
                chunks=(min(nnz, 1 << 20),),
            )
            for start in range(0, n_obs, BLOCK):
                stop = min(start + BLOCK, n_obs)
                lo, hi = int(indptr[start]), int(indptr[stop])
                data[lo:hi] = x["data"][lo:hi].astype(np.float32)
                indices[lo:hi] = x["indices"][lo:hi].astype(np.int32)
                print(f"  rows {stop:>9,} / {n_obs:,}", end="\r", flush=True)
    print(f"\nwrote {args.target} ({n_obs:,} x {n_vars:,}, {nnz:,} stored values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
