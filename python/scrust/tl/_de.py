"""Differential expression, mirroring `scanpy.tl.rank_genes_groups`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scrust._shared import _LABEL_DTYPE, _csr_args, _extension

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

__all__ = ["filter_rank_genes_groups", "rank_genes_groups"]

# scanpy's dtype per `uns["rank_genes_groups"]` field; its plotting reads these.
_DE_FIELD_DTYPES = {
    "names": "O",
    "scores": "float32",
    "logfoldchanges": "float32",
    "pvals": "float64",
    "pvals_adj": "float64",
}

_SUPPORTED_METHODS = ("wilcoxon", "t-test", "t-test_overestim_var", "logreg")
_TIE_CORRECT = False

# scanpy reports neither p-values nor fold changes for `logreg`: a coefficient is
# not a test statistic, so its `uns` entry holds only `names` and `scores`.
_LOGREG_FIELDS = ("names", "scores")

# sklearn's `max_iter`, which scanpy does not override.
_LOGREG_MAX_ITER = 100


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    groups: str | Sequence[str] = "all",
    reference: str = "rest",
    method: str = "wilcoxon",
    device: str = "auto",
) -> None:
    """Rank genes by differential expression, writing `uns["rank_genes_groups"]`."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {_SUPPORTED_METHODS}, got {method!r}")

    group_names, label_names = _group_order(adata, groupby, groups, reference)
    codes = _labels(adata, groupby, label_names)
    # "rest" is the core's None, not a sentinel index.
    reference_index = None if reference == "rest" else label_names.index(reference)

    # The core labels every cell it is given, so cells outside the selected
    # groups are dropped here rather than encoded as an out-of-range label.
    compared = codes >= 0
    matrix = adata.X[compared] if not compared.all() else adata.X

    result = _compare(
        matrix,
        codes[compared].astype(_LABEL_DTYPE),
        len(label_names),
        reference_index,
        method,
        device,
    )

    # The core returns statistics in gene order; scanpy's slot is ranked by score.
    rows = [label_names.index(name) for name in group_names]
    gene_names = adata.var_names.to_numpy()
    scores = np.asarray(result["scores"])
    # scanpy reverses an ascending argsort, which flips the order of tied
    # scores; sorting the negated array instead would not match it.
    order = {row: np.argsort(scores[row])[::-1] for row in rows}
    columns = {
        "names": [gene_names[order[row]] for row in rows],
        "scores": [scores[row][order[row]] for row in rows],
        "logfoldchanges": [
            np.asarray(result["log2_fold_changes"])[row][order[row]] for row in rows
        ],
        "pvals": [np.asarray(result["p_values"])[row][order[row]] for row in rows],
        "pvals_adj": [np.asarray(result["adjusted_p_values"])[row][order[row]] for row in rows],
    }
    fields = {key: columns[key] for key in _LOGREG_FIELDS} if method == "logreg" else columns
    adata.uns["rank_genes_groups"] = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "method": method,
            "use_raw": False,
            "layer": None,
            "corr_method": "benjamini-hochberg",
        },
        **{
            key: _record_array(columns, group_names, _DE_FIELD_DTYPES[key])
            for key, columns in fields.items()
        },
    }


def _compare(
    matrix, codes: np.ndarray, n_labels: int, reference_index: int | None, method: str, device: str
) -> dict:
    """Dispatch one method to the core, which returns per-group statistics in gene order."""
    extension = _extension()
    csr = _csr_args(matrix)
    if method == "wilcoxon":
        return extension.rank_genes_groups_wilcoxon(
            *csr, codes, n_labels, reference_index, _TIE_CORRECT, device
        )
    if method == "logreg":
        # One multinomial fit over every labelled cell, as scanpy does: a named
        # reference takes part as a further class, not as a comparison.
        return extension.rank_genes_groups_logreg(*csr, codes, n_labels, _LOGREG_MAX_ITER, device)
    t_test = {
        "t-test": extension.rank_genes_groups_t_test,
        "t-test_overestim_var": extension.rank_genes_groups_t_test_overestim_var,
    }[method]
    return t_test(*csr, codes, n_labels, reference_index, device)


def _group_order(
    adata: AnnData, groupby: str, groups: str | Sequence[str], reference: str
) -> tuple[list[str], list[str]]:
    """Return the groups to report on, and the groups the core must label.

    A named `reference` is labelled too even when it is not reported, because the
    test compares against its cells.
    """
    if groupby not in adata.obs:
        raise KeyError(f"adata.obs has no {groupby!r}")
    categories = [str(name) for name in adata.obs[groupby].astype("category").cat.categories]

    if isinstance(groups, str):
        if groups != "all":
            raise ValueError("groups must be 'all' or a sequence of group names")
        group_names = list(categories)
    else:
        group_names = [str(name) for name in groups]

    unknown = set(group_names + ([reference] if reference != "rest" else [])) - set(categories)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)} in adata.obs[{groupby!r}]")

    label_names = list(group_names)
    if reference != "rest" and reference not in label_names:
        label_names.append(reference)
    return group_names, label_names


def _labels(adata: AnnData, groupby: str, label_names: Sequence[str]) -> np.ndarray:
    """Encode the grouping as codes into `label_names`, `-1` for excluded cells."""
    positions = {name: index for index, name in enumerate(label_names)}
    observed = adata.obs[groupby].astype(str)
    return observed.map(lambda name: positions.get(name, -1)).to_numpy(dtype=np.int32)


def _record_array(columns: Sequence[np.ndarray], group_names: Sequence[str], dtype: str):
    """Build the one-field-per-group structured array scanpy's accessors expect."""
    return np.rec.fromarrays(
        [np.asarray(column, dtype=dtype) for column in columns],
        dtype=[(name, dtype) for name in group_names],
    )


def filter_rank_genes_groups(
    adata: AnnData,
    *,
    key: str = "rank_genes_groups",
    groupby: str | None = None,
    key_added: str = "rank_genes_groups_filtered",
    min_in_group_fraction: float = 0.25,
    max_out_group_fraction: float = 0.5,
    min_fold_change: float = 2.0,
) -> None:
    """Blank out genes failing the expression-fraction filters, as scanpy does.

    The result keeps the shape of `uns[key]` and replaces the names of the genes
    that fail with `NaN`, which is what scanpy's plotting expects to find.
    """
    # Imported here, and the two helpers defined here, because another branch
    # owns the rest of this file: this keeps the diff inside one function.
    import pandas as pd

    def expressed_fraction(expression) -> np.ndarray:
        """Fraction of cells with a stored, non-zero count, per gene."""
        n_expressing = (
            expression.getnnz(axis=0)
            if hasattr(expression, "getnnz")
            else np.count_nonzero(expression, axis=0)
        )
        return n_expressing / expression.shape[0]

    def log_fold_change(inside, outside) -> np.ndarray:
        """Log2 fold change, undoing the log the counts were stored in.

        The 1e-9 is scanpy's guard against a gene expressed in neither group.
        """
        base = adata.uns.get("log1p", {}).get("base")
        expm1 = np.expm1 if base is None else (lambda values: np.expm1(values * np.log(base)))
        return np.log2(
            (expm1(np.ravel(inside.mean(0))) + 1e-9) / (expm1(np.ravel(outside.mean(0))) + 1e-9)
        )

    def frame(columns: dict):
        """One column per group, in the order `names` stores them."""
        return pd.DataFrame(columns, index=gene_names.index, columns=gene_names.columns)

    if key not in adata.uns:
        raise KeyError(f"adata.uns has no {key!r}; run rank_genes_groups first")
    result = adata.uns[key]
    params = result["params"]
    if groupby is None:
        groupby = params["groupby"]
    # The stored statistics describe one particular comparison, so scanpy only
    # reuses them when that is the comparison being filtered, and recomputes
    # them from X otherwise.
    same_params = params["groupby"] == groupby and params["reference"] == "rest"
    use_logfolds = same_params and "logfoldchanges" in result
    use_fractions = same_params and "pts_rest" in result

    gene_names = pd.DataFrame(result["names"])
    in_fractions, out_fractions, fold_changes = {}, {}, {}
    for group in gene_names.columns:
        genes = gene_names[group].to_numpy()
        if use_fractions:
            in_fractions[group] = result["pts"][group].loc[genes].to_numpy()
            out_fractions[group] = result["pts_rest"][group].loc[genes].to_numpy()
        if not (use_fractions and use_logfolds):
            in_group = (adata.obs[groupby] == group).to_numpy()
            expression = adata[:, genes].X
            inside, outside = expression[in_group], expression[~in_group]
        if not use_fractions:
            in_fractions[group] = expressed_fraction(inside)
            out_fractions[group] = expressed_fraction(outside)
        if not use_logfolds:
            fold_changes[group] = log_fold_change(inside, outside)

    fold_change = pd.DataFrame(result["logfoldchanges"]) if use_logfolds else frame(fold_changes)
    kept = (
        (frame(in_fractions) > min_in_group_fraction)
        & (frame(out_fractions) < max_out_group_fraction)
        & (fold_change > min_fold_change)
    )
    adata.uns[key_added] = {**result, "names": gene_names[kept].to_records(index=False)}
