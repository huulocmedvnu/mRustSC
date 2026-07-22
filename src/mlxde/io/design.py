"""Construction of treatment-coded design matrices from sample metadata."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlxde.contracts import DesignMatrix

INTERCEPT_NAME = "intercept"


def _require_column(sample_metadata: pd.DataFrame, column: str, role: str) -> pd.Series:
    if column not in sample_metadata.columns:
        available = list(map(str, sample_metadata.columns))
        raise KeyError(f"unknown {role} column {column!r}; available: {available}")
    return sample_metadata[column]


def _factor_levels(condition: pd.Series, reference_level: str | None) -> tuple[str, list[str]]:
    """Reference level and the remaining levels, in a deterministic order."""
    levels = sorted({str(value) for value in condition})
    if not levels:
        raise ValueError("condition column is empty")
    if reference_level is None:
        return levels[0], levels[1:]
    if reference_level not in levels:
        raise ValueError(f"unknown reference level {reference_level!r}; available: {levels}")
    return reference_level, [level for level in levels if level != reference_level]


def _covariate_column(sample_metadata: pd.DataFrame, column: str) -> np.ndarray:
    values = _require_column(sample_metadata, column, "covariate")
    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError(f"covariate {column!r} must be numeric, got dtype {values.dtype}")
    numeric = values.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"covariate {column!r} contains missing or infinite values")
    return numeric


def build_design_matrix(
    sample_metadata: pd.DataFrame,
    condition_column: str,
    reference_level: str | None = None,
    covariate_columns: Sequence[str] = (),
) -> DesignMatrix:
    """Treatment-coded model matrix for ``condition_column``.

    Column 0 is the intercept, holding the mean of the reference level; each
    remaining level of the factor gets an indicator column named
    ``f"{condition_column}[{level}]"``, so a caller asks for its comparison by
    name via :meth:`DesignMatrix.contrast`. Numeric covariates are appended
    unchanged, under their own column names.
    """
    condition = _require_column(sample_metadata, condition_column, "condition").astype(str)
    reference, contrast_levels = _factor_levels(condition, reference_level)

    columns = [np.ones(len(sample_metadata))]
    names = [INTERCEPT_NAME]
    for level in contrast_levels:
        columns.append((condition == level).to_numpy(dtype=np.float64))
        names.append(f"{condition_column}[{level}]")
    for column in covariate_columns:
        columns.append(_covariate_column(sample_metadata, column))
        names.append(column)

    matrix = np.column_stack(columns)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError(
            f"design matrix is rank deficient: columns {names} are linearly dependent "
            f"for {matrix.shape[0]} samples (reference level {reference!r})"
        )
    return DesignMatrix(matrix=matrix, coefficient_names=tuple(names))
