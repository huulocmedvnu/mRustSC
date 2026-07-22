"""Reading count matrices and their sample metadata from delimited text files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlxde.contracts import CountMatrix

_SEPARATOR_BY_SUFFIX = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".txt": "\t"}


def table_separator(path: Path) -> str:
    """Field separator implied by ``path``'s suffix.

    Shared with the writer so a table written by this package reads back with
    the same dialect it was written in.
    """
    suffix = path.suffix.lower()
    if suffix not in _SEPARATOR_BY_SUFFIX:
        raise ValueError(
            f"unsupported table format {path.suffix!r} for {path}; "
            f"expected one of {sorted(_SEPARATOR_BY_SUFFIX)}"
        )
    return _SEPARATOR_BY_SUFFIX[suffix]


def read_table(path: Path, id_column: str, role: str) -> pd.DataFrame:
    """Read a delimited table indexed by its identifier column.

    ``id_column`` is used when the file carries such a header; otherwise the
    first column is taken as the identifier, which is what the contract
    promises and what most upstream tools emit.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{role} file not found: {path}")

    table = pd.read_csv(path, sep=table_separator(path))
    if table.columns.empty:
        raise ValueError(f"{role} file {path} has no columns")

    identifiers = id_column if id_column in table.columns else table.columns[0]
    return table.set_index(identifiers)


def _validate_unique(identifiers: pd.Index, role: str, path: Path) -> None:
    duplicates = identifiers[identifiers.duplicated()].unique()
    if len(duplicates) > 0:
        raise ValueError(f"duplicate {role} in {path}: {sorted(map(str, duplicates))}")


def align_metadata(sample_metadata: pd.DataFrame, sample_ids: pd.Index) -> pd.DataFrame:
    """Reorder ``sample_metadata`` rows to follow ``sample_ids``.

    Samples are never dropped or silently reordered: any mismatch between the
    two id sets is a data error, because it would otherwise pair counts with
    the covariates of a different sample.
    """
    missing_metadata = sample_ids.difference(sample_metadata.index)
    if len(missing_metadata) > 0:
        raise ValueError(
            f"samples present in the counts but missing from the metadata: "
            f"{sorted(map(str, missing_metadata))}"
        )
    extra_metadata = sample_metadata.index.difference(sample_ids)
    if len(extra_metadata) > 0:
        raise ValueError(
            f"samples present in the metadata but missing from the counts: "
            f"{sorted(map(str, extra_metadata))}"
        )
    return sample_metadata.loc[sample_ids]


def _validate_counts(counts: pd.DataFrame, path: Path) -> np.ndarray:
    non_numeric = [
        str(column)
        for column, dtype in counts.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype)
    ]
    if non_numeric:
        raise ValueError(f"non-numeric count columns in {path}: {sorted(non_numeric)}")

    values = counts.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"counts in {path} contain missing or infinite values")
    if (values < 0).any():
        raise ValueError(f"counts in {path} contain negative values")
    return values


class CsvCountReader:
    """``CountReader`` for a counts table plus a sample metadata table.

    The counts file holds gene ids in its first column and one column per
    sample; the metadata file holds sample ids in its first column and one
    covariate per remaining column.
    """

    def __init__(
        self, gene_id_column: str = "gene_id", sample_id_column: str = "sample_id"
    ) -> None:
        self._gene_id_column = gene_id_column
        self._sample_id_column = sample_id_column

    def read(self, counts_path: Path, metadata_path: Path) -> CountMatrix:
        counts_path = Path(counts_path)
        metadata_path = Path(metadata_path)

        counts = read_table(counts_path, self._gene_id_column, "counts")
        metadata = read_table(metadata_path, self._sample_id_column, "metadata")

        _validate_unique(counts.index, "gene ids", counts_path)
        _validate_unique(counts.columns, "sample columns", counts_path)
        _validate_unique(metadata.index, "sample ids", metadata_path)

        values = _validate_counts(counts, counts_path)
        aligned_metadata = align_metadata(metadata, counts.columns)

        return CountMatrix(
            counts=values,
            gene_ids=counts.index.to_numpy(dtype=object).astype(str),
            sample_ids=counts.columns.to_numpy(dtype=object).astype(str),
            sample_metadata=aligned_metadata,
        )
