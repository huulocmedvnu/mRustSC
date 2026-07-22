"""Writing differential expression results to delimited text files."""

from __future__ import annotations

from pathlib import Path

from mlxde.contracts import DifferentialExpressionResult
from mlxde.io.readers import table_separator


class CsvResultWriter:
    """``ResultWriter`` emitting the result table ordered by adjusted p-value."""

    def write(self, result: DifferentialExpressionResult, path: Path) -> None:
        path = Path(path)
        separator = table_separator(path)
        # Stable sort so genes tied on the adjusted p-value keep their input order.
        table = result.to_dataframe().sort_values("adjusted_p_value", kind="stable")
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep=separator, index=False)
