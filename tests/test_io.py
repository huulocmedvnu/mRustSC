"""Tests for the io layer: readers, writers and the design matrix builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlxde.contracts import CountReader, DifferentialExpressionResult, ResultWriter
from mlxde.io.design import build_design_matrix
from mlxde.io.readers import CsvCountReader
from mlxde.io.writers import CsvResultWriter

SAMPLE_IDS = ["s1", "s2", "s3", "s4"]
COUNTS = np.array([[10, 12, 30, 33], [0, 1, 2, 3], [5, 5, 5, 5]], dtype=np.int64)
GENE_IDS = ["g1", "g2", "g3"]


def write_counts(path: Path, counts: np.ndarray = COUNTS) -> Path:
    table = pd.DataFrame(counts, index=pd.Index(GENE_IDS, name="gene_id"), columns=SAMPLE_IDS)
    table.to_csv(path)
    return path


def write_metadata(path: Path, sample_ids: list[str] = SAMPLE_IDS) -> Path:
    table = pd.DataFrame(
        {"condition": ["control", "control", "treated", "treated"]},
        index=pd.Index(sample_ids, name="sample_id"),
    )
    table.to_csv(path)
    return path


@pytest.fixture
def counts_path(tmp_path: Path) -> Path:
    return write_counts(tmp_path / "counts.csv")


@pytest.fixture
def metadata_path(tmp_path: Path) -> Path:
    return write_metadata(tmp_path / "metadata.csv")


def test_reader_satisfies_protocol() -> None:
    assert isinstance(CsvCountReader(), CountReader)
    assert isinstance(CsvResultWriter(), ResultWriter)


def test_read_round_trip(counts_path: Path, metadata_path: Path) -> None:
    matrix = CsvCountReader().read(counts_path, metadata_path)

    assert matrix.counts.shape == (3, 4)
    assert matrix.counts.dtype == np.float64
    assert list(matrix.gene_ids) == GENE_IDS
    assert list(matrix.sample_ids) == SAMPLE_IDS
    np.testing.assert_array_equal(matrix.counts, COUNTS.astype(np.float64))
    assert list(matrix.sample_metadata.index) == SAMPLE_IDS
    assert list(matrix.sample_metadata.columns) == ["condition"]


def test_read_aligns_shuffled_metadata(tmp_path: Path, counts_path: Path) -> None:
    shuffled = ["s3", "s1", "s4", "s2"]
    metadata = pd.DataFrame(
        {"condition": ["treated", "control", "treated", "control"]},
        index=pd.Index(shuffled, name="sample_id"),
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path)

    matrix = CsvCountReader().read(counts_path, metadata_path)

    assert list(matrix.sample_metadata.index) == SAMPLE_IDS
    assert list(matrix.sample_metadata["condition"]) == ["control", "control", "treated", "treated"]


def test_read_supports_tsv(tmp_path: Path) -> None:
    counts_path = tmp_path / "counts.tsv"
    pd.DataFrame(COUNTS, index=pd.Index(GENE_IDS, name="gene_id"), columns=SAMPLE_IDS).to_csv(
        counts_path, sep="\t"
    )
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame(
        {"condition": ["control", "control", "treated", "treated"]},
        index=pd.Index(SAMPLE_IDS, name="sample_id"),
    ).to_csv(metadata_path, sep="\t")

    matrix = CsvCountReader().read(counts_path, metadata_path)

    assert matrix.counts.shape == (3, 4)


def test_missing_counts_file_raises(tmp_path: Path, metadata_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="counts file not found"):
        CsvCountReader().read(tmp_path / "absent.csv", metadata_path)


def test_missing_metadata_file_raises(tmp_path: Path, counts_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metadata file not found"):
        CsvCountReader().read(counts_path, tmp_path / "absent.csv")


def test_unsupported_suffix_raises(tmp_path: Path, metadata_path: Path) -> None:
    counts_path = tmp_path / "counts.h5ad"
    counts_path.write_text("irrelevant")
    with pytest.raises(ValueError, match="unsupported table format"):
        CsvCountReader().read(counts_path, metadata_path)


def test_sample_missing_from_metadata_raises(tmp_path: Path, counts_path: Path) -> None:
    metadata_path = write_metadata(tmp_path / "metadata.csv", sample_ids=["s1", "s2", "s3", "sX"])
    with pytest.raises(ValueError, match=r"missing from the metadata.*'s4'"):
        CsvCountReader().read(counts_path, metadata_path)


def test_sample_missing_from_counts_raises(tmp_path: Path, metadata_path: Path) -> None:
    counts_path = tmp_path / "counts.csv"
    pd.DataFrame(
        COUNTS[:, :3], index=pd.Index(GENE_IDS, name="gene_id"), columns=["s1", "s2", "s3"]
    ).to_csv(counts_path)
    with pytest.raises(ValueError, match=r"missing from the counts.*'s4'"):
        CsvCountReader().read(counts_path, metadata_path)


def test_duplicate_gene_ids_raise(tmp_path: Path, metadata_path: Path) -> None:
    counts_path = tmp_path / "counts.csv"
    pd.DataFrame(
        COUNTS, index=pd.Index(["g1", "g1", "g3"], name="gene_id"), columns=SAMPLE_IDS
    ).to_csv(counts_path)
    with pytest.raises(ValueError, match="duplicate gene ids"):
        CsvCountReader().read(counts_path, metadata_path)


def test_duplicate_sample_ids_raise(tmp_path: Path, counts_path: Path) -> None:
    metadata_path = write_metadata(tmp_path / "metadata.csv", sample_ids=["s1", "s1", "s3", "s4"])
    with pytest.raises(ValueError, match="duplicate sample ids"):
        CsvCountReader().read(counts_path, metadata_path)


def test_non_numeric_counts_raise(tmp_path: Path, metadata_path: Path) -> None:
    counts_path = tmp_path / "counts.csv"
    table = pd.DataFrame(COUNTS, index=pd.Index(GENE_IDS, name="gene_id"), columns=SAMPLE_IDS)
    table["s2"] = ["a", "b", "c"]
    table.to_csv(counts_path)
    with pytest.raises(ValueError, match="non-numeric count columns"):
        CsvCountReader().read(counts_path, metadata_path)


def test_negative_counts_raise(tmp_path: Path, metadata_path: Path) -> None:
    negative = COUNTS.copy()
    negative[1, 1] = -5
    counts_path = write_counts(tmp_path / "counts.csv", counts=negative)
    with pytest.raises(ValueError, match="negative values"):
        CsvCountReader().read(counts_path, metadata_path)


def test_missing_counts_values_raise(tmp_path: Path, metadata_path: Path) -> None:
    counts_path = tmp_path / "counts.csv"
    table = pd.DataFrame(
        COUNTS.astype(float), index=pd.Index(GENE_IDS, name="gene_id"), columns=SAMPLE_IDS
    )
    table.iloc[0, 0] = np.nan
    table.to_csv(counts_path)
    with pytest.raises(ValueError, match="missing or infinite"):
        CsvCountReader().read(counts_path, metadata_path)


def two_level_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["control", "control", "treated", "treated"],
            "age": [30.0, 40.0, 50.0, 65.0],
        },
        index=SAMPLE_IDS,
    )


def test_design_two_level_factor() -> None:
    design = build_design_matrix(two_level_metadata(), "condition")

    assert design.coefficient_names == ("intercept", "condition[treated]")
    np.testing.assert_array_equal(
        design.matrix, np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=float)
    )
    np.testing.assert_array_equal(design.contrast("condition[treated]"), [0.0, 1.0])


def test_design_reference_level_choice() -> None:
    design = build_design_matrix(two_level_metadata(), "condition", reference_level="treated")

    assert design.coefficient_names == ("intercept", "condition[control]")
    np.testing.assert_array_equal(design.matrix[:, 1], [1.0, 1.0, 0.0, 0.0])


def test_design_three_level_factor() -> None:
    metadata = pd.DataFrame({"treatment": ["low", "high", "control", "low"]}, index=SAMPLE_IDS)

    design = build_design_matrix(metadata, "treatment", reference_level="control")

    assert design.coefficient_names == ("intercept", "treatment[high]", "treatment[low]")
    np.testing.assert_array_equal(
        design.matrix,
        np.array([[1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]], dtype=float),
    )


def test_design_with_numeric_covariate() -> None:
    design = build_design_matrix(two_level_metadata(), "condition", covariate_columns=["age"])

    assert design.coefficient_names == ("intercept", "condition[treated]", "age")
    np.testing.assert_array_equal(design.matrix[:, 2], [30.0, 40.0, 50.0, 65.0])
    assert design.n_samples == 4
    assert design.n_coefficients == 3


def test_design_unknown_reference_level_raises() -> None:
    with pytest.raises(ValueError, match="unknown reference level 'absent'"):
        build_design_matrix(two_level_metadata(), "condition", reference_level="absent")


def test_design_unknown_condition_column_raises() -> None:
    with pytest.raises(KeyError, match="unknown condition column"):
        build_design_matrix(two_level_metadata(), "genotype")


def test_design_non_numeric_covariate_raises() -> None:
    metadata = two_level_metadata()
    metadata["batch"] = ["a", "b", "a", "b"]
    with pytest.raises(TypeError, match="must be numeric"):
        build_design_matrix(metadata, "condition", covariate_columns=["batch"])


def test_design_rank_deficient_raises() -> None:
    metadata = two_level_metadata()
    # `batch` is perfectly confounded with `condition`, so their effects are
    # not separately identifiable.
    metadata["batch"] = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="rank deficient"):
        build_design_matrix(metadata, "condition", covariate_columns=["batch"])


def make_result() -> DifferentialExpressionResult:
    return DifferentialExpressionResult(
        gene_ids=np.array(["g1", "g2", "g3"]),
        base_mean=np.array([100.0, 20.0, 3.0]),
        log2_fold_change=np.array([1.5, -0.25, 3.0]),
        log2_fold_change_standard_error=np.array([0.2, 0.3, 0.4]),
        statistic=np.array([7.5, -0.83, 7.5]),
        p_value=np.array([1e-8, 0.4, 1e-4]),
        adjusted_p_value=np.array([3e-8, 0.4, 1.5e-4]),
    )


def test_writer_sorts_by_adjusted_p_value(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    result = make_result()

    CsvResultWriter().write(result, path)
    written = pd.read_csv(path)

    assert list(written["gene_id"]) == ["g1", "g3", "g2"]
    assert written["adjusted_p_value"].is_monotonic_increasing


def test_writer_round_trips_values(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    result = make_result()

    CsvResultWriter().write(result, path)
    written = pd.read_csv(path).set_index("gene_id")
    expected = result.to_dataframe().set_index("gene_id")

    pd.testing.assert_frame_equal(written, expected.loc[written.index])


def test_writer_creates_missing_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "results.tsv"

    CsvResultWriter().write(make_result(), path)

    assert list(pd.read_csv(path, sep="\t")["gene_id"]) == ["g1", "g3", "g2"]
