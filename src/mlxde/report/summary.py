"""Human-readable summary of a differential expression result."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlxde.contracts import DifferentialExpressionResult

_MAX_STRONGEST_HITS = 10


def summarize(result: DifferentialExpressionResult, alpha: float = 0.05) -> str:
    """Return a short report; the caller decides whether to print, log or store it."""
    table = result.to_dataframe()
    n_tested = int(np.isfinite(table["adjusted_p_value"].to_numpy()).sum())
    n_filtered = len(table) - n_tested

    significant = result.significant(alpha=alpha)
    n_up = int((significant["log2_fold_change"] > 0).sum())
    n_down = int((significant["log2_fold_change"] < 0).sum())

    lines = [
        "Differential expression summary",
        "------------------------------",
        f"Genes in result:      {len(table)}",
        f"Genes tested:         {n_tested}",
        f"Genes filtered out:   {n_filtered}",
        f"Significant at FDR <= {alpha:g}: {len(significant)}",
        f"  up-regulated:       {n_up}",
        f"  down-regulated:     {n_down}",
    ]
    lines.extend(_strongest_hits(significant, alpha))
    return "\n".join(lines)


def _strongest_hits(significant: pd.DataFrame, alpha: float) -> list[str]:
    if significant.empty:
        return ["", f"No significant genes at FDR <= {alpha:g}."]

    # `significant` is already ordered by adjusted p-value, strongest evidence first.
    hits = significant.head(_MAX_STRONGEST_HITS)
    lines = ["", f"Strongest hits (up to {_MAX_STRONGEST_HITS}):"]
    lines.extend(
        f"  {row.gene_id}  log2FC={row.log2_fold_change:+.3f}  padj={row.adjusted_p_value:.3g}"
        for row in hits.itertuples()
    )
    return lines
