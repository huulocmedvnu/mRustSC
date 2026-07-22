"""CLI behaviour, exercised against fake layers.

The io, pipeline and report layers are developed on other branches, so every
collaborator is replaced by a recording double here. What is under test is the
adapter itself: argument parsing, the order and arguments of the delegations,
the printed text and the exit codes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mlxde.cli.main import app
from mlxde.contracts import CountMatrix, DesignMatrix, DifferentialExpressionResult

runner = CliRunner()


@dataclass
class Recorder:
    """Every delegation the CLI makes, in the order it makes them."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def arguments(self, name: str) -> dict[str, Any]:
        return next(arguments for called, arguments in self.calls if called == name)


def make_count_matrix() -> CountMatrix:
    metadata = pd.DataFrame({"condition": ["control", "control", "treated", "treated"]})
    return CountMatrix(
        counts=np.arange(12, dtype=float).reshape(3, 4),
        gene_ids=np.array(["gene_0", "gene_1", "gene_2"]),
        sample_ids=np.array(["s0", "s1", "s2", "s3"]),
        sample_metadata=metadata,
    )


def make_result() -> DifferentialExpressionResult:
    """Three genes, of which one is significant at alpha=0.05 and |log2FC| >= 1."""
    return DifferentialExpressionResult(
        gene_ids=np.array(["gene_0", "gene_1", "gene_2"]),
        base_mean=np.array([10.0, 20.0, 30.0]),
        log2_fold_change=np.array([3.0, 0.5, 0.1]),
        log2_fold_change_standard_error=np.array([0.1, 0.1, 0.1]),
        statistic=np.array([30.0, 5.0, 1.0]),
        p_value=np.array([1e-6, 1e-3, 0.5]),
        adjusted_p_value=np.array([1e-5, 0.02, 0.9]),
    )


def install_fake_layers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    design: DesignMatrix | None = None,
    reader_error: Exception | None = None,
) -> Recorder:
    """Replace the not-yet-existing collaborators by recording doubles."""
    recorder = Recorder()
    count_matrix = make_count_matrix()
    result = make_result()
    coefficients = (
        design
        if design is not None
        else DesignMatrix(
            matrix=np.column_stack([np.ones(4), [0.0, 0.0, 1.0, 1.0]]),
            coefficient_names=("intercept", "condition[treated]"),
        )
    )

    class FakeCsvCountReader:
        def read(self, counts_path: Path, metadata_path: Path) -> CountMatrix:
            recorder.calls.append(
                ("read", {"counts_path": counts_path, "metadata_path": metadata_path})
            )
            if reader_error is not None:
                raise reader_error
            return count_matrix

    class FakeCsvResultWriter:
        def write(self, written: DifferentialExpressionResult, path: Path) -> None:
            recorder.calls.append(("write", {"result": written, "path": path}))

    class FakePipeline:
        def run(
            self, matrix: CountMatrix, used_design: DesignMatrix, contrast: np.ndarray
        ) -> DifferentialExpressionResult:
            recorder.calls.append(
                ("pipeline.run", {"counts": matrix, "design": used_design, "contrast": contrast})
            )
            return result

    def fake_build_design_matrix(
        sample_metadata: pd.DataFrame,
        condition_column: str,
        reference_level: str | None = None,
    ) -> DesignMatrix:
        recorder.calls.append(
            (
                "build_design_matrix",
                {
                    "sample_metadata": sample_metadata,
                    "condition_column": condition_column,
                    "reference_level": reference_level,
                },
            )
        )
        return coefficients

    def fake_build_default_pipeline(backend_name: str | None = None) -> FakePipeline:
        recorder.calls.append(("build_default_pipeline", {"backend_name": backend_name}))
        return FakePipeline()

    def fake_summarize(summarized: DifferentialExpressionResult, alpha: float = 0.05) -> str:
        recorder.calls.append(("summarize", {"result": summarized, "alpha": alpha}))
        return "<report>"

    modules = {
        "mlxde.io.readers": {"CsvCountReader": FakeCsvCountReader},
        "mlxde.io.writers": {"CsvResultWriter": FakeCsvResultWriter},
        "mlxde.io.design": {"build_design_matrix": fake_build_design_matrix},
        "mlxde.report.summary": {"summarize": fake_summarize},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr("mlxde.factory.build_default_pipeline", fake_build_default_pipeline)
    return recorder


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    counts_path = tmp_path / "counts.csv"
    metadata_path = tmp_path / "metadata.csv"
    counts_path.write_text("gene_id,s0\n")
    metadata_path.write_text("sample_id,condition\n")
    return counts_path, metadata_path, tmp_path / "results.csv"


def invoke_run(inputs: tuple[Path, Path, Path], *extra: str):
    counts_path, metadata_path, output_path = inputs
    return runner.invoke(
        app,
        [
            "run",
            "--counts",
            str(counts_path),
            "--metadata",
            str(metadata_path),
            "--condition",
            "condition",
            "--output",
            str(output_path),
            *extra,
        ],
    )


def test_run_delegates_to_every_layer_in_order(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    recorder = install_fake_layers(monkeypatch)
    counts_path, metadata_path, output_path = inputs

    result = invoke_run(inputs, "--reference", "control", "--backend", "numpy")

    assert result.exit_code == 0, result.output
    assert recorder.names[:4] == [
        "read",
        "build_design_matrix",
        "build_default_pipeline",
        "pipeline.run",
    ]
    assert "write" in recorder.names
    assert recorder.arguments("read") == {
        "counts_path": counts_path,
        "metadata_path": metadata_path,
    }
    design_arguments = recorder.arguments("build_design_matrix")
    assert design_arguments["condition_column"] == "condition"
    assert design_arguments["reference_level"] == "control"
    assert recorder.arguments("build_default_pipeline") == {"backend_name": "numpy"}
    assert recorder.arguments("write")["path"] == output_path


def test_run_contrasts_the_non_reference_level(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    recorder = install_fake_layers(monkeypatch)

    assert invoke_run(inputs).exit_code == 0

    contrast = recorder.arguments("pipeline.run")["contrast"]
    assert np.array_equal(contrast, np.array([0.0, 1.0]))


def test_run_defaults_backend_and_reference_to_none(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    recorder = install_fake_layers(monkeypatch)

    assert invoke_run(inputs).exit_code == 0

    assert recorder.arguments("build_default_pipeline") == {"backend_name": None}
    assert recorder.arguments("build_design_matrix")["reference_level"] is None


def test_run_reports_tested_and_significant_counts(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    install_fake_layers(monkeypatch)

    result = invoke_run(inputs, "--alpha", "0.05", "--min-log2fc", "1.0")

    assert result.exit_code == 0
    assert "1 of 3 tested genes are significant" in result.output
    assert "<report>" in result.output


def test_run_thresholds_change_the_significant_count(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    install_fake_layers(monkeypatch)

    result = invoke_run(inputs, "--alpha", "0.5")

    assert result.exit_code == 0
    assert "2 of 3 tested genes are significant" in result.output


def test_missing_input_file_exits_non_zero_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_layers(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--counts",
            str(tmp_path / "absent.csv"),
            "--metadata",
            str(tmp_path / "absent_metadata.csv"),
            "--condition",
            "condition",
            "--output",
            str(tmp_path / "results.csv"),
        ],
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "absent.csv" in _text(result)


def test_reader_failure_becomes_a_message(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    install_fake_layers(monkeypatch, reader_error=ValueError("counts and metadata disagree"))

    result = invoke_run(inputs)

    assert result.exit_code == 1
    assert "counts and metadata disagree" in _text(result)
    assert "Traceback" not in _text(result)


def test_unknown_coefficient_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path]
) -> None:
    design_without_condition = DesignMatrix(
        matrix=np.ones((4, 1)), coefficient_names=("intercept",)
    )
    recorder = install_fake_layers(monkeypatch, design=design_without_condition)

    result = invoke_run(inputs)

    assert result.exit_code == 1
    assert "condition" in _text(result)
    assert "Traceback" not in _text(result)
    assert "pipeline.run" not in recorder.names


def test_backends_lists_numpy() -> None:
    result = runner.invoke(app, ["backends"])

    assert result.exit_code == 0
    assert "numpy" in result.output.split()


@pytest.mark.parametrize("arguments", [["--help"], ["run", "--help"], ["backends", "--help"]])
def test_help_is_available(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0


def _text(result) -> str:
    """Everything the CLI printed, whether the runner splits stderr or not."""
    try:
        return result.output + result.stderr
    except ValueError:
        return result.output
