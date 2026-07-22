"""Command line entry point.

A thin adapter over the library: it parses arguments, delegates to the layers
below and formats their answer. It owns no statistics, no file format and no
modelling rule of its own.

Collaborators are imported inside the commands, the way the composition root
does, so that importing this module — and therefore ``--help`` — never depends
on an optional layer being installed or importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from mlxde.contracts import DesignMatrix

app = typer.Typer(
    add_completion=False,
    help="Differential expression analysis for RNA-seq count data.",
)


@app.command()
def run(
    counts: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="CSV of raw counts, genes by samples."),
    ],
    metadata: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="CSV of per-sample metadata."),
    ],
    condition: Annotated[str, typer.Option(help="Metadata column holding the condition.")],
    output: Annotated[Path, typer.Option(dir_okay=False, help="CSV to write the results to.")],
    reference: Annotated[
        str | None, typer.Option(help="Condition level used as the baseline.")
    ] = None,
    backend: Annotated[
        str | None, typer.Option(help="Compute backend; the best available one by default.")
    ] = None,
    alpha: Annotated[float, typer.Option(help="Adjusted p-value threshold.")] = 0.05,
    min_log2fc: Annotated[
        float, typer.Option("--min-log2fc", help="Minimum absolute log2 fold change.")
    ] = 0.0,
) -> None:
    """Run a differential expression analysis and write the result table."""
    try:
        summary = _analyse(
            counts_path=counts,
            metadata_path=metadata,
            condition=condition,
            reference=reference,
            output_path=output,
            backend_name=backend,
            alpha=alpha,
            min_log2_fold_change=min_log2fc,
        )
    except Exception as error:  # a user of the CLI gets a message, not a traceback
        typer.secho(f"Error: {_message(error)}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from error
    typer.echo(summary)


@app.command()
def backends() -> None:
    """List the compute backends available on this machine."""
    from mlxde.backend import available_backends

    for name in available_backends():
        typer.echo(name)


def _analyse(
    *,
    counts_path: Path,
    metadata_path: Path,
    condition: str,
    reference: str | None,
    output_path: Path,
    backend_name: str | None,
    alpha: float,
    min_log2_fold_change: float,
) -> str:
    """Wire the layers together for one run and return the text to print."""
    from mlxde.factory import build_default_pipeline
    from mlxde.io.design import build_design_matrix
    from mlxde.io.readers import CsvCountReader
    from mlxde.io.writers import CsvResultWriter
    from mlxde.report.summary import summarize

    count_matrix = CsvCountReader().read(counts_path, metadata_path)
    design = build_design_matrix(count_matrix.sample_metadata, condition, reference_level=reference)
    contrast = design.contrast(_treatment_coefficient(design, condition))

    result = build_default_pipeline(backend_name).run(count_matrix, design, contrast)
    CsvResultWriter().write(result, output_path)

    n_tested = len(result.gene_ids)
    n_significant = len(result.significant(alpha, min_log2_fold_change))
    return (
        f"{summarize(result, alpha)}\n"
        f"{n_significant} of {n_tested} tested genes are significant "
        f"(adjusted p <= {alpha}, |log2FC| >= {min_log2_fold_change}); "
        f"results written to {output_path}"
    )


def _message(error: Exception) -> str:
    """Human-readable text of an exception; ``str(KeyError)`` would add quotes."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error) or type(error).__name__


def _treatment_coefficient(design: DesignMatrix, condition: str) -> str:
    """Name of the single non-reference level of ``condition`` in the design."""
    prefix = f"{condition}["
    candidates = [name for name in design.coefficient_names if name.startswith(prefix)]
    if len(candidates) != 1:
        raise KeyError(
            f"expected exactly one non-reference level of {condition!r} in the design, "
            f"found {candidates or 'none'}; available coefficients: {design.coefficient_names}"
        )
    return candidates[0]
