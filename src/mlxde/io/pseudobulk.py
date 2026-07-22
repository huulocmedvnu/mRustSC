"""Pool single cells into pseudobulk replicates.

Per-cell counts violate the negative binomial GLM's assumption of independent
replicates, so single-cell differential expression is run on sums of cells. This
module only reshapes counts; it knows nothing about how the cells were labelled.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mlxde.contracts import CountMatrix


def build_pseudobulk(
    counts: pd.DataFrame,
    cell_labels: pd.Series,
    conditions: Mapping[str, str],
    n_replicates: int = 5,
    seed: int = 0,
) -> CountMatrix:
    """Sum each labelled cell population into `n_replicates` disjoint pools.

    `counts` is cells x genes, `cell_labels` maps each cell to a population, and
    `conditions` maps the populations of interest to the condition they belong
    to. Cells outside `conditions` are ignored.
    """
    if n_replicates < 2:
        raise ValueError(f"need at least 2 replicates per condition, got {n_replicates}")
    missing = set(conditions) - set(cell_labels.unique())
    if missing:
        raise KeyError(f"no cells labelled {sorted(missing)}")

    rng = np.random.default_rng(seed)
    pools: dict[str, np.ndarray] = {}
    condition_of_sample: dict[str, str] = {}
    for population, condition in conditions.items():
        cells = np.asarray(cell_labels.index[cell_labels == population])
        if len(cells) < n_replicates:
            raise ValueError(
                f"{population!r} has {len(cells)} cells, too few for {n_replicates} replicates"
            )
        for replicate, pool in enumerate(np.array_split(rng.permutation(cells), n_replicates)):
            sample_id = f"{condition}_{replicate}"
            pools[sample_id] = counts.loc[pool].to_numpy().sum(axis=0)
            condition_of_sample[sample_id] = condition

    sample_ids = np.array(list(pools))
    return CountMatrix(
        counts=np.column_stack([pools[sample_id] for sample_id in sample_ids]),
        gene_ids=np.asarray(counts.columns),
        sample_ids=sample_ids,
        sample_metadata=pd.DataFrame(
            {"condition": [condition_of_sample[sample_id] for sample_id in sample_ids]},
            index=sample_ids,
        ),
    )
