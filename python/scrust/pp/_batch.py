"""Removing unwanted variation. Owned by feat/regress-combat.

Plumbing only: this module resolves `obs` column names to a numeric design,
hands the matrix to the Rust core, and writes the dense result back to
`adata.X`. The arithmetic — including the intercept column and the one-hot
encoding's effect on the fit — belongs to the core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scrust._shared import (
    _LABEL_DTYPE,
    _VALUE_DTYPE,
    _csr_args,
    _default_device,
    _extension,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from anndata import AnnData

__all__ = ["combat", "regress_out"]


def regress_out(
    adata: AnnData, keys: str | Sequence[str], *, device: str | None = None, inplace: bool = True
) -> np.ndarray | None:
    """Regress each gene on `keys` and keep the residuals, as `scanpy.pp.regress_out`."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    if not keys:
        raise ValueError("regress_out needs at least one obs column to regress on")
    residuals = np.asarray(
        _extension().regress_out(
            *_csr_args(adata.X), _design(adata, keys), _resolve_device(device)
        ),
        dtype=_VALUE_DTYPE,
    )
    if not inplace:
        return residuals
    adata.X = residuals
    return None


def combat(
    adata: AnnData,
    key: str = "batch",
    *,
    covariates: Sequence[str] | None = None,
    device: str | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Empirical Bayes batch correction, as `scanpy.pp.combat`."""
    covariates = list(covariates or [])
    if key in covariates:
        raise ValueError(f"the batch key {key!r} cannot also be a covariate")
    if len(covariates) != len(set(covariates)):
        raise ValueError("covariates must be unique")

    labels = _categories(adata, key)
    corrected = np.asarray(
        _extension().combat(
            *_csr_args(adata.X),
            labels.codes.astype(_LABEL_DTYPE, copy=False),
            len(labels.categories),
            _design(adata, covariates) if covariates else None,
            _resolve_device(device),
        ),
        dtype=_VALUE_DTYPE,
    )
    if not inplace:
        return corrected
    adata.X = corrected
    return None


def _resolve_device(device: str | None) -> str:
    """The caller's device, or `settings.device` when they named none."""
    return _default_device() if device is None else device


def _design(adata: AnnData, keys: Sequence[str]) -> np.ndarray:
    """The `obs` columns as a numeric `(n_obs, k)` design.

    No intercept: the core adds the one its model needs, which is a different
    column for `regress_out` (a constant) and for `combat` (the batch
    indicators).
    """
    columns = [block for key in keys for block in _columns(adata, key)]
    if not columns:
        return np.zeros((adata.n_obs, 0), dtype=_VALUE_DTYPE)
    return np.ascontiguousarray(np.column_stack(columns), dtype=_VALUE_DTYPE)


def _columns(adata: AnnData, key: str) -> list[np.ndarray]:
    """One column for a numeric annotation, one per level but the first for a
    categorical one.

    Dropping a level is what keeps the design full rank once the core adds its
    own constant or batch columns; the fit spans the same space either way, so
    the residuals are scanpy's whatever parameterisation is chosen.
    """
    import pandas as pd

    if key not in adata.obs:
        raise KeyError(f"adata.obs has no column {key!r}")
    series = adata.obs[key]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return [series.to_numpy(dtype=_VALUE_DTYPE)]

    levels = pd.Categorical(series).remove_unused_categories()
    codes = levels.codes
    if (codes < 0).any():
        raise ValueError(f"adata.obs[{key!r}] has missing values, which cannot be regressed on")
    return [(codes == level).astype(_VALUE_DTYPE) for level in range(1, len(levels.categories))]


def _categories(adata: AnnData, key: str) -> pd.Categorical:
    """The batch annotation as categories, in the order scanpy groups them."""
    import pandas as pd

    if key not in adata.obs:
        raise ValueError(f"could not find the key {key!r} in adata.obs")
    labels = pd.Categorical(adata.obs[key]).remove_unused_categories()
    if (labels.codes < 0).any():
        raise ValueError(f"adata.obs[{key!r}] has missing values, which cannot be a batch")
    return labels
